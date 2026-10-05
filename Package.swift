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
        .library(name: "MLXNacre", targets: ["MLXNacre"]),
        .executable(name: "nacre-smoke", targets: ["NacreSmoke"]),   // CLI gate modes
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.63.0"),
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
        // MLXEngine `imageUpscale` wrapper over the local core.
        .target(
            name: "MLXNacre",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "NacreMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXNacreTests",
            dependencies: [
                "MLXNacre",
                "NacreMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformanceNN", package: "mlx-engine-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]   // the loaded MLX graphs are not Sendable-audited (C14 walker)
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
                "MLXNacre",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Sources/Smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
