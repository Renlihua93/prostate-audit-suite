#!/usr/bin/env python3
"""Local loading and batch embedding for pathology foundation encoders.

H-optimus-0 ships a timm state dict, Phikon/Phikon-v2 are HF transformers
models; each has its own normalization constants, so they live here together.

Encoders expect 224x224 inputs at 0.5 microns per pixel.
"""

from __future__ import annotations
from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT  # noqa: E402

from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

SPECS = {
    "phikon": {
        "kind": "hf", "id": "owkin/phikon", "dim": 768,
        "mean": IMAGENET[0], "std": IMAGENET[1],
        "local": [ROOT / "models/phikon"],
    },
    "phikon_v2": {
        "kind": "hf", "id": "owkin/phikon-v2", "dim": 1024,
        "mean": IMAGENET[0], "std": IMAGENET[1],
        "local": [Path(f"{MODEL_ROOT}/HF-models/HF-models/phikon-v2"),
                  ROOT / "models/phikon-v2"],
    },
    "hoptimus0": {
        "kind": "timm", "id": "bioptimus/H-optimus-0", "dim": 1536,
        "arch": "vit_giant_patch14_reg4_dinov2",
        "mean": (0.707223, 0.578729, 0.703617),
        "std": (0.211883, 0.230117, 0.177517),
        "local": [Path(f"{MODEL_ROOT}/H-optimus-0"), ROOT / "models/h-optimus-0"],
    },
    "resnet50": {
        "kind": "torchvision", "id": "resnet50", "dim": 2048,
        "mean": IMAGENET[0], "std": IMAGENET[1],
        "local": [Path(f"{MODEL_ROOT}/HF-models/HF-models/resnet50"),
                  ROOT / "models/resnet50"],
    },
}


def _find_local(spec) -> Path | None:
    marker = {"hf": "config.json", "timm": "pytorch_model.bin",
              "torchvision": "model.safetensors"}[spec["kind"]]
    for p in spec["local"]:
        if (p / marker).exists():
            return p
    return None


class Encoder:
    def __init__(self, name: str, device: str, dtype: torch.dtype | None = None):
        if name not in SPECS:
            raise SystemExit(f"unknown encoder {name}; choose from {sorted(SPECS)}")
        self.name = name
        self.device = device
        self.dtype = dtype or torch.float32
        self.dim = SPECS[name]["dim"]
        spec = SPECS[name]
        self.mean = np.array(spec["mean"], dtype=np.float32)
        self.std = np.array(spec["std"], dtype=np.float32)
        local = _find_local(spec)

        if local is None and spec["kind"] != "hf":
            raise SystemExit(f"{name}: put the weights under one of {spec['local']}")

        if spec["kind"] == "timm":
            import timm
            model = timm.create_model(spec["arch"], pretrained=False,
                                      img_size=224, num_classes=0,
                                      init_values=1e-5, dynamic_img_size=False)
            sd = torch.load(local / "pytorch_model.bin", map_location="cpu",
                            weights_only=True)
            model.load_state_dict(sd, strict=True)
            del sd
            self._forward = lambda xb: self.model(xb)
        elif spec["kind"] == "torchvision":
            from safetensors.torch import load_file
            from torchvision.models import resnet50
            model = resnet50(weights=None)
            sd = {k: v for k, v in
                  load_file(str(local / "model.safetensors")).items()
                  if not k.startswith("fc.")}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if unexpected or set(missing) - {"fc.weight", "fc.bias"}:
                raise SystemExit(f"{name}: unexpected keys {unexpected}, "
                                 f"missing {missing}")
            del sd
            model.fc = torch.nn.Identity()
            self._forward = lambda xb: self.model(xb)
        else:
            from transformers import AutoModel
            src = str(local) if local is not None else spec["id"]
            model = AutoModel.from_pretrained(src)
            self._forward = lambda xb: self.model(xb).last_hidden_state[:, 0]

        self.model = model.eval().to(device)
        if dtype is not None:
            self.model = self.model.to(dtype)

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        arr = np.stack([np.asarray(im.convert("RGB").resize((224, 224),
                                                            Image.BILINEAR),
                                   dtype=np.float32) / 255.0
                        for im in images])
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device,
                                                          dtype=self.dtype)

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        xb = self.preprocess(images)
        if self.device == "cuda" and self.dtype == torch.float32:
            with torch.autocast("cuda", dtype=torch.float16):
                out = self._forward(xb)
        else:
            out = self._forward(xb)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 3:
            out = out[:, 0]
        return out.float().cpu().numpy()

    def embed_paths(self, paths: list[Path], batch: int = 16,
                    log_every: int = 0) -> np.ndarray:
        import time
        embs = []
        t0 = time.time()
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p) for p in paths[i:i + batch]]
            for im in imgs:
                im.load()
            embs.append(self.embed(imgs))
            for im in imgs:
                im.close()
            if log_every and i % (batch * log_every) == 0:
                print(f"  {i}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
        return np.concatenate(embs)


def load_encoder(name: str, device: str | None = None, fp16: bool = True) -> Encoder:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (fp16 and device == "cuda") else None
    return Encoder(name, device, dtype)
