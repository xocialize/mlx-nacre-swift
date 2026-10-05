#!/usr/bin/env python3
# Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
"""Convert the released Nacre v1 weights to the MLX layout the Swift port loads.

    /Volumes/Satechi/Development/mlxengine-image/WIP/nacre/.venv/bin/python oracle/convert_weights.py [--fp16]

Per-tensor exact: conv weights [O, I, kh, kw] -> [O, kh, kw, I] (OHWI), everything else unchanged. Two
tensors in the release are NOT parameters of the Swift module tree and are dropped here, by name:
  * `*.attn.relative_position_index` — a buffer, rebuilt from the window size in Swift;
  * `quantize.embedding.weight` — the VQ codebook; Nacre decodes continuous latents and never snaps.
One file per network (`nacre_v1_mlx.safetensors`, `vq_f4_mlx.safetensors`) so each lane downloads
only what it needs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("/Volumes/Satechi/Development/mlxengine-image/WIP/nacre/release/nacre-v1")
OUT = Path(__file__).resolve().parent / "weights"
DROP = (".attn.relative_position_index", "quantize.embedding.weight")


def convert(src: Path, dst: Path, dtype: torch.dtype) -> None:
    sd = load_file(src)
    out = {}
    for k, v in sd.items():
        if k.endswith(DROP) or k == "quantize.embedding.weight":
            continue
        if v.ndim == 4:
            v = v.permute(0, 2, 3, 1)
        out[k] = v.contiguous().to(dtype)
    save_file(out, dst, metadata={"layout": "mlx-ohwi", "source": src.name, "dtype": str(dtype)})
    print(f"{src.name} -> {dst.name}: {len(out)} tensors ({len(sd) - len(out)} dropped)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    dt, tag = (torch.float16, "fp16") if a.fp16 else (torch.float32, "fp32")
    convert(SRC / "nacre_v1.safetensors", OUT / f"nacre_v1_mlx_{tag}.safetensors", dt)
    convert(SRC / "vq_f4.safetensors", OUT / f"vq_f4_mlx_{tag}.safetensors", dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
