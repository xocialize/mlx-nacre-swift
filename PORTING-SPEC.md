# mlx-nacre-swift — porting spec

Port of **Nacre v1** (`hf://xocialize/nacre-v1`, Apache-2.0; receipt AB-R-0396) from the reference PyTorch package
`mlxengine-image/WIP/nacre` to Swift/MLX. Oracle = that reference, CPU fp32, released weights; goldens dumped by
`oracle/dump_goldens.py` (cases `s64` = LQ 64², latent 64² — the training grid; `s128` = LQ 128², shifted windows and
the two-axis mask active at every level). Weights converted per-tensor exact by `oracle/convert_weights.py` (conv OHWI;
`relative_position_index` buffers and the unused VQ codebook dropped by name).

Gates run from the CLI lane (`swift run -c release nacre-smoke <mode>`); every stamp below was written after the run.

| Phase | Gate | Status |
|---|---|---|
| S0 key contract | module flattened keys == converted file keys, 0 missing / 0 unused, both networks (`keys`) | **PASSED 2026-10-05** — unet 540/540 (118,593,583 params), vq 204/204 (55,298,206). Nested `[[Module]]` arrays reproduce `input_blocks.N.M` natively, no remap |
| S1 substrate | bicubic ×4 (torch kernel), timestep embedding, time MLP, prior sample, input_blocks[0], ResBlock, Swin stage, Swin block unshifted + SHIFTED, full UNet at t=3, VQ encode, VQ decode — each from the oracle's own input (`gate`) | **PASSED 2026-10-05** (CPU, fp32), both cases. Per-op rel ≤ 1e-5 except VQ encode: rel 1.06e-4 (s64) / 2.2e-5 (s128) — tolerance 2e-4 for that one row, because torch's own fp32 encode differs from its fp64 encode by 3.8e-5 rel (measured) and the Swift encode sits within ~3× of that |
| S2 core pipeline | 4-step chain per step + z0 + decode + `upscale()` e2e with injected noise | **PASSED 2026-10-05** — CPU e2e **130.9 / 131.1 dB** vs torch (s64 / s128); GPU fp32 with `MLX_ENABLE_TF32=0` **127.9 / 127.6 dB**; GPU fp32 with MLX's default TF32 66.0 / 66.8 dB (AB-L-0175 — still above 8-bit quantisation, but the parity lane is TF32-off) |
| S2b GPU + eyeball | one real image (RealSR Canon_007 LR, 300×200 → 1200×800) on Metal | **PASSED 2026-10-05** — fp32 coherent (town, rocks, water); 4.7 s whole-image |
| S3a dtype | fp16 lane | **PASSED 2026-10-05 after two fixes.** First fp16 decode was all-black (7.1 dB): VQ GroupNorm statistics and the VQ attention scores overflow fp16. Both now fp32 (UNet GroupNorm already was). fp16 UNet + fp16 VQ vs fp32 lane: **58.1 dB** on the S2b image (8-bit PNGs); UNet-only fp16 57.9, VQ-only fp16 58.7 |
| S3b reductions | CPU-lane accuracy | Two-level GroupNorm mean (blocks of 1,024 rows) — MLX's CPU stream sums sequentially (AB-L-0180); lifted the CPU e2e from 117.5 → 130.9 dB |
| S3c memory + tiling | peak vs LQ size; tiled vs whole-image on a real image vs ground truth | **MEASURED 2026-10-05** (M5 Max, release, fp16 whole-image): LQ 64² 2.09 GB / 0.15 s · 128² 2.92 GB / 0.46 s · 192² 5.36 GB / 1.40 s · 256² 8.76 GB / 2.97 s. A 500×400 LQ (RealSR Nikon_010): whole-image **29.7 GB / 15.4 s**; tiled 128 / overlap 32 **2.97 GB / 9.4 s**, and *better* vs GT (PSNR-Y 23.99 vs 23.33, LPIPS 0.206 vs 0.222; MUSIQ 57.2 vs 60.3), no visible seams (blend-centre gradient 3.56 vs 4.41 global — overlap bands slightly smoother). VQ attention chunked above 4,096 tokens in 1,024-query slices (exact) |
| S4 package | MLXNacre `imageUpscale` wrapper, C0–C14, MAT, CAN, split footprint | **PASSED 2026-10-05.** Offline suite 16/16 (`swift test --build-system swiftbuild --filter MLXNacreTests`): manifest C7/C8 Apache-2.0 both layers + permissive-only admitted, provenance `xocialize/`, split footprints per lane, `QuantConfigured`/`BudgetAware` (fp32 → fp16 under 8 GB), canonical `imageUpscale` surface, registration, weight sources per lane, Codable portable-knobs-only, **CAN-1..3**, **MAT-1..5** per lane + store layout + `needsDownload`, **C14 INF** (passes, and fails on a graph flipped to training). Live through the REAL `MLXServeEngine` (`nacre-smoke engine`, release, M5 Max): 500×400 → 2000×1600 **fp16 10.0 s, phys peak 4.50 GB** (MLX 2.11); **fp32 14.0 s, 5.53 GB**; raw BGRA in/out + `scale: 2` honoured (1000×800); **fresh store**: the engine materialised ONLY the fp16 lane (2 × safetensors + config, 347 MB) from `xocialize/nacre-v1-mlx`, then ran (9.9 s); **live cancel** at 3 s → `CancellationError` (unwrapped) in **0.158 s** |
| S5 publish | repo + MLX-layout weights + registry row | **PASSED 2026-10-05.** Attribution complete (every CC BY training image named; full licence re-audit of all 51,759 images clean — AB-R-0396). PUBLIC: github.com/xocialize/mlx-nacre-swift (v0.1.1), hf.co/xocialize/nacre-v1 (release), hf.co/xocialize/nacre-v1-mlx (MLX lanes). Anonymous verification: credential-free curl of the fp16 lane = local sha256 (both files); credential-free clone at v0.1.0 FAILED to resolve (empty `Tests/NacreMLXTests` never tracked) → v0.1.1 adds the 5 core tests there (S0 counts, schedule vs oracle float64, two-axis mask, bicubic partition of unity, C14 at construction); suite 21/21 |
| S6 in-app | Val in a real app (Nacre Demo, `Demos/Nacre Demo`) + AB-T-0019 phys re-baseline | **PASSED 2026-10-05** (v0.1.2, fp16 activation corrected in v0.1.3). Release build, app's own `MLXServeEngine` (governor 0.7, GPU pool cap 2 GB), `ValidationHarness` isolate, one fresh process per number, RealSR Nikon_010 500×400: **fp16 ×3** floor 0.42 GB (baseline 0.05; MLX active 0.33 GB = the weights), kernel lifetime peak 4.57–4.62 GB, run 9.7–10.0 s; **fp32 ×2** floor 0.77–0.82 GB, peak 5.54–5.58 GB, run 13.4 s. Declared split corrected 0.80 + 3.85 → **0.40 + 4.30** (fp16; v0.1.2's 4.25 came from the first three runs and the next three peaked at 4.70 GB = 4.28 GB activation — fp16 ×6 total, 4.57–4.70 GB), 1.05 + 4.5 → **0.78 + 4.80** (fp32) — the CLI had read "resident" after the run (+0.30–0.36 GB non-MLX residue), right total / wrong split. The harness's 150 ms sampler under-read the peak by 0.2–0.45 GB every run; activation is declared from the kernel's `ledger_phys_footprint_peak`. **[CAN]** in-app: cancel at 3.2 s (tile 7/20) → `CancellationError` in 0.07 s, rerun clean. **[MAT]** cold fp16 lane into a throwaway store: 347.9 MB in 18.5 s, `.downloading` phase seen, marker written. Also: a tiling config with `overlap ≥ tile/2` hit the tiler's `precondition` (kills the host app) — now refused with `NacreError.badInput` at run |

## Decisions

- **Tiled by default** (128 LQ, overlap 32, fp16). Peak is set by the tile, and the measured fidelity is better than
  whole-image (the model trained on 64-LQ crops; the VQ's global attention over very large latents drifts).
- **fp16 shipping lane**, with every reduction (GroupNorm statistics, VQ attention scores + softmax) in fp32.
- **Parity lane = CPU fp32, or GPU fp32 with `MLX_ENABLE_TF32=0`.**

## Traps met

1. `roll(_:shift:axes:)` with one shift and two axes compiles and then aborts at runtime ("one shift value per axis")
   — roll one axis at a time.
2. fp16 VQ = black output (GroupNorm sums of squares and 512-wide q·k overflow fp16).
3. M5 GPU fp32 is TF32-class by default (AB-L-0175) — the GPU parity lane must set `MLX_ENABLE_TF32=0`.
4. MLX CPU reductions are sequential (AB-L-0180) — 262k-element GroupNorm groups need a two-level sum on the CPU lane.
