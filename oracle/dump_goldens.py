#!/usr/bin/env python3
# Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
"""Dump per-component parity goldens for the Swift port from the PyTorch reference (the ORACLE).

Run with the nacre venv (it owns the reference implementation):

    /Volumes/Satechi/Development/mlxengine-image/WIP/nacre/.venv/bin/python oracle/dump_goldens.py

Everything runs on the CPU in float32 with the released weights (`nacre_v1.safetensors`,
`vq_f4.safetensors`). Arrays are written channel-LAST (NHWC) so the Swift side compares directly.
The noise the sampler consumes is drawn HERE and saved, then injected on both sides — parity never
depends on two RNGs agreeing.

Cases (each its own LQ size, so position-dependent machinery is exercised at more than one grid):
  s64   LQ 64²   (latent 64² — the training grid; deepest level 8², a single window, shift OFF there)
  s128  LQ 128²  (deepest level 16² — shifted windows + the two-axis mask active at EVERY level)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

NACRE = Path("/Volumes/Satechi/Development/mlxengine-image/WIP/nacre")
sys.path.insert(0, str(NACRE))
from nacre.diffusion.residual_shift import ResidualShiftDiffusion  # noqa: E402
from nacre.models.unet import SwinUNet  # noqa: E402
from nacre.models.vq import VQAutoencoder  # noqa: E402

REL = NACRE / "release/nacre-v1"
OUT = Path(__file__).resolve().parent / "goldens"
SRC = Path("/Volumes/Satechi/Development/mlxengine-image/corpus/sr-bench/DIV2K_valid_HR/0801.png")
torch.set_grad_enabled(False)
torch.set_num_threads(8)


def nhwc(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().permute(0, 2, 3, 1).contiguous().numpy()


def save(case: str, name: str, arr: np.ndarray) -> None:
    (OUT / case).mkdir(parents=True, exist_ok=True)
    np.save(OUT / case / f"{name}.npy", np.ascontiguousarray(arr.astype(np.float32)))


def main() -> int:
    cfg = json.loads((REL / "config.json").read_text())
    net = SwinUNet(**cfg["model"]).eval()
    net.load_state_dict(load_file(REL / "nacre_v1.safetensors"), strict=True)
    vq = VQAutoencoder()
    vq.load_state_dict(load_file(REL / "vq_f4.safetensors"), strict=True)
    diff = ResidualShiftDiffusion(**cfg["diffusion"])

    # Schedule scalars, float64 — the Swift side recomputes these in Double and must match exactly.
    sched = {k: getattr(diff, k).tolist() for k in
             ("sqrt_etas", "etas", "posterior_mean_coef1", "posterior_mean_coef2",
              "posterior_log_variance_clipped")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "schedule.json").write_text(json.dumps(sched, indent=1))

    src = np.asarray(Image.open(SRC).convert("RGB"), dtype=np.float32) / 255.0
    g = torch.Generator().manual_seed(20261005)
    for case, side in (("s64", 64), ("s128", 128)):
        # A real LQ: area-downsample a natural crop 4x (content, not noise, so every rung sees
        # realistic statistics). The degradation itself is irrelevant to parity.
        crop = torch.from_numpy(src[200:200 + 4 * side, 300:300 + 4 * side]).permute(2, 0, 1)[None]
        lq = F.interpolate(crop, scale_factor=0.25, mode="area").clamp(0, 1)
        save(case, "lq01", nhwc(lq))

        up = F.interpolate(lq, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
        save(case, "bicubic_up", nhwc(up))
        y0 = vq.encode(up * 2 - 1)
        save(case, "y0", nhwc(y0))
        cond = lq * 2 - 1

        # --- substrate taps on the first UNet input path (t = 3) -----------------------------
        t3 = torch.full((1,), 3, dtype=torch.long)
        from nacre.models.nn_blocks import timestep_embedding
        temb = timestep_embedding(t3, cfg["model"]["model_channels"])
        emb = net.time_embed(temb)
        save(case, "temb", temb.numpy()); save(case, "emb", emb.numpy())

        noises = [torch.randn(y0.shape, generator=g) for _ in range(4)]  # prior + 3 chain draws
        for i, n in enumerate(noises):
            save(case, f"noise{i}", nhwc(n))
        x_t = diff.prior_sample(y0, noises[0])
        save(case, "x_T", nhwc(x_t))
        xin = diff.scale_input(x_t, t3)
        save(case, "x_in", nhwc(xin))

        h = torch.cat([xin, net.feature_extractor(cond)], dim=1)
        h = net.input_blocks[0](h, emb); save(case, "in0", nhwc(h))
        rb = net.input_blocks[1][0](h, emb); save(case, "in1_res", nhwc(rb))
        sw = net.input_blocks[1][1](rb); save(case, "in1_swin", nhwc(sw))
        blk = net.input_blocks[1][1].blocks[1]   # the SHIFTED block of that stage
        pre = net.input_blocks[1][1].patch_embed.proj(rb)
        save(case, "swin_b0_in", nhwc(pre))
        b0 = net.input_blocks[1][1].blocks[0](pre); save(case, "swin_b0_out", nhwc(b0))
        save(case, "swin_b1_out", nhwc(blk(b0)))

        x0_pred = net(xin, t3, lq=cond); save(case, "unet_t3", nhwc(x0_pred))

        # --- the full chain with injected noise ---------------------------------------------
        draws = iter(noises[1:])
        x = x_t
        per_step = []
        for step in range(diff.num_timesteps - 1, -1, -1):
            t = torch.full((1,), step, dtype=torch.long)
            mean, _, log_var, _ = diff.p_mean_variance(net, x, y0, t, cond=cond)
            x = mean if step == 0 else mean + torch.exp(0.5 * log_var) * next(draws)
            per_step.append(x)
            save(case, f"chain_t{step}", nhwc(x))
        save(case, "z0", nhwc(x))
        img = ((vq.decode(x) + 1) / 2).clamp(0, 1)
        save(case, "out01", nhwc(img))
        print(f"{case}: lq {tuple(lq.shape)} -> out {tuple(img.shape)}")
    print("goldens ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
