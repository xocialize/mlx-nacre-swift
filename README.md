# mlx-nacre-swift

Swift/MLX port of **Nacre v1** — Xocialize's provenance-clean, ResShift-class 4× real-world super-resolution model —
packaged for [MLXEngine](https://github.com/xocialize/mlx-engine-swift) as an `imageUpscale` provider.

Nacre is a 118.6 M-parameter SwinUNet driving a **4-step residual-shift diffusion** in the latent space of the CompVis
VQ-f4 autoencoder. It is permissive all the way down: Apache-2.0 code and denoiser weights trained from scratch on
Wikimedia Commons CC0 / public-domain / CC BY images only; the autoencoder is CompVis' MIT release. On neutral
benchmarks it is at parity with the (non-commercial) ResShift v3: fidelity level or better, MUSIQ ahead, CLIPIQA
0–0.04 behind.

| product | what |
|---|---|
| `NacreMLX` | the engine-agnostic core: `Nacre` (whole-image and tiled 4× restore), `SwinUNet`, `VQAutoencoder`, the residual-shift schedule |
| `MLXNacre` | `NacreUpscalePackage` — the MLXEngine `imageUpscale` package (`NacreUpscalePackage.registration`) |
| `nacre-smoke` | the CLI gate lane: `keys` · `gate` · `gate-gpu` · `run` · `tiled` · `mem` · `engine` · `cancel` |

```swift
let id = try await engine.register(NacreUpscalePackage.registration, configuration: NacreConfiguration())
let out = try await engine.run(ImageUpscaleRequest(image: png), package: id) as! ImageUpscaleResponse
```

**Defaults:** fp16 lane (every reduction in fp32 inside), tiled 128-px LQ tiles with 32-px overlap (peak memory is set
by the tile, not the image — ~4.7 GB process peak at the default tile, measured in-app; declared per tile 64–256), fixed seed (reproducible). 500×400 → 2000×1600 in ~10 s on an M5 Max.

**Generative caveat:** Nacre invents plausible detail. Like every generative upscaler measured, it can alter text that
was already legible — route legible small text to a fidelity upscaler.

Weights: `xocialize/nacre-v1-mlx` (MLX layout, fp16 + fp32 lanes), materialised by the engine on first use. Parity,
measurements and decisions: [PORTING-SPEC.md](PORTING-SPEC.md). Licence: Apache-2.0 (see NOTICE for the CompVis MIT
autoencoder and the training-image attributions).
