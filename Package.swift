// swift-tools-version: 6.2
import PackageDescription

// mlx-nacre-swift — Nacre v1 (Xocialize, Apache-2.0: a clean-room ResShift-class 4× real-world super-resolution model —
// a 118.6 M SwinUNet denoiser driving a 4-step residual-shift diffusion in the latent space of the CompVis VQ-f4
// autoencoder (MIT); trained only on Wikimedia Commons CC0 / PD / CC BY images) ported PyTorch → Swift/MLX for
// MLXEngine as the provenance-clean GENERATIVE stills upscale tier. The mlx-heart-swift shape:
//   • NacreMLX — engine-agnostic Swift/MLX core, isomorphic to the reference `nacre/` package (models/unet.py, swin.py,
//     nn_blocks.py, vq.py; diffusion/residual_shift.py): NHWC. No MLXToolKit dependency.
//   • MLXNacre — the MLXEngine `imageUpscale` ModelPackage over that core.
// Parity: oracle/dump_goldens.py dumps per-stage goldens from the reference PyTorch code on the CPU with the released
// weights and injected noise (PORTING-SPEC.md S0–S7). Weights: oracle/convert_weights.py (per-tensor exact, OHWI).
let package = Package(
    name: "mlx-nacre-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "NacreMLX", targets: ["NacreMLX"]),
        .executable(name: "nacre-smoke", targets: ["NacreSmoke"]),   // CLI gate modes
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    ],
    targets: [
        .target(
            name: "NacreMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "NacreMLXTests",
            dependencies: [
                "NacreMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "NacreSmoke",
            dependencies: [
                "NacreMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/Smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
