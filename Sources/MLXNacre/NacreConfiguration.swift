// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
import Foundation
import MLXToolKit
import NacreMLX

/// Init-time configuration for `NacreUpscalePackage` (C9). Stable for the session.
///
/// Nacre ships two precision lanes from one repo (`xocialize/nacre-v1-mlx`, MLX layout, per-tensor exact):
///
///   quant   files                                                   weights   role
///   .fp16   nacre_v1_mlx_fp16 + vq_f4_mlx_fp16 (.safetensors)      ≈ 348 MB  shipping lane — every reduction (GroupNorm
///                                                                             statistics, VQ attention scores + softmax)
///                                                                             in fp32 inside; 58 dB vs the fp32 lane
///   .fp32   nacre_v1_mlx_fp32 + vq_f4_mlx_fp32 (.safetensors)      ≈ 696 MB  parity lane (TF32 on M5 unless the HOST sets
///                                                                             MLX_ENABLE_TF32=0 — AB-L-0175)
///
/// bf16 resolves to fp16. Nacre is a GENERATIVE model: the output depends on `seed` (fixed by default, so a given
/// image + configuration is reproducible).
public struct NacreConfiguration: PackageConfiguration, ModelStorable, QuantConfigured, BudgetAware, FootprintConfigured {
    /// `.fp16` (default) or `.fp32`; anything else is treated as `.fp16`.
    public var quant: Quant
    /// LQ tile side for the tiled path (whole pipeline per tile, linear-ramp blend); `nil` = 128 (PORTING-SPEC S3c:
    /// peak set by the tile; better fidelity than whole-image). 16…256 (`maxTile`); any size — each tile is padded
    /// internally. The activation footprint follows the tile, so it is declared per tile (`peakActivationBytesHint`).
    public var tileSize: Int?
    /// LQ overlap between tiles; `nil` = 32. Must be < tile/2.
    public var tileOverlap: Int?
    /// Diffusion seed. Same image + seed + tiling ⇒ same output.
    public var seed: UInt64

    /// Absolute path to a directory holding the lane's files. Honored OVER the store — dev-mode escape hatch.
    public var weightsDirectory: URL?
    /// Where the engine materializes weights — stamped from its `ModelStore.root` (`ModelStorable`).
    public var modelsRootDirectory: URL?
    /// Real headroom at load, stamped by the governor (`BudgetAware`): below `fp32MinBudgetBytes` an fp32
    /// configuration loads the fp16 lane instead.
    public var availableBudgetBytes: UInt64?

    public static let repo = "xocialize/nacre-v1-mlx"
    /// The floor for the fp32 lane at the DEFAULT tile; the real test is `fp32MinBudgetBytes(tile:)`.
    public static let fp32MinBudgetBytes: UInt64 = 8_000_000_000

    /// Headroom the fp32 lane needs at `tile`: its weights + that tile's declared activation, never below the
    /// default-tile floor. Below it, an fp32 configuration loads the fp16 lane (tile 256 fp32 ≈ 16.6 GB).
    public static func fp32MinBudgetBytes(tile: Int) -> UInt64 {
        max(fp32MinBudgetBytes, 780_000_000 + declaredActivationBytes(quant: .fp32, tile: tile))
    }

    public init(quant: Quant = .fp16,
                tileSize: Int? = nil,
                tileOverlap: Int? = nil,
                seed: UInt64 = 20260923,
                weightsDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil,
                availableBudgetBytes: UInt64? = nil) {
        self.quant = quant
        self.tileSize = tileSize
        self.tileOverlap = tileOverlap
        self.seed = seed
        self.weightsDirectory = weightsDirectory
        self.modelsRootDirectory = modelsRootDirectory
        self.availableBudgetBytes = availableBudgetBytes
    }

    public var effectiveQuant: Quant { quant == .fp32 ? .fp32 : .fp16 }

    // MARK: footprint per tile (FootprintConfigured)

    /// The largest tile with a measured envelope. Bigger tiles cost more memory AND fidelity (the model trained on
    /// 64-px LQ crops; whole-image lost to tiled vs GT in S3c), so the API stops where the measurements stop.
    public static let maxTile = 256

    /// Declared activation per measured tile, IN-APP (Nacre Demo, Release, app engine, GPU pool cap 2 GB, one fresh
    /// process per number, kernel lifetime phys peak − post-load floor; 2026-10-05, AB-T-0019), declared as
    /// max × 1.2 + 256 MB (the image-fleet N6 convention):
    ///
    ///   tile   fp16 measured max → declared     fp32 measured max → declared
    ///    64        3.25 GB (×2) →  4.16 GB          4.01 GB (×2) →  5.07 GB
    ///   128        4.28 GB (×6) →  5.40 GB          4.77 GB (×3) →  5.98 GB
    ///   192        7.08 GB (×2) →  8.76 GB          9.13 GB (×4) → 11.22 GB   (fp32 spread 7.72–9.13: real, not contention)
    ///   256        9.52 GB (×2) → 11.69 GB         12.97 GB (×2) → 15.83 GB
    ///
    /// A tile between two rows takes the next row up; below 64 takes the 64 row.
    static let activationByTile: [(tile: Int, fp16: UInt64, fp32: UInt64)] = [
        (64, 4_160_000_000, 5_070_000_000),
        (128, 5_400_000_000, 5_980_000_000),
        (192, 8_760_000_000, 11_220_000_000),
        (256, 11_690_000_000, 15_830_000_000),
    ]

    public static func declaredActivationBytes(quant: Quant, tile: Int) -> UInt64 {
        let row = activationByTile.first { tile <= $0.tile } ?? activationByTile[activationByTile.count - 1]
        return quant == .fp32 ? row.fp32 : row.fp16
    }

    /// Weights are per lane only — the quant row's `residentBytes` stands.
    public var residentBytesHint: UInt64? { nil }

    /// The tile's activation, so a configuration with a bigger tile is admitted against what it will really use.
    public var peakActivationBytesHint: UInt64? {
        Self.declaredActivationBytes(quant: effectiveQuant, tile: tileSize ?? Nacre.defaultTile)
    }

    public static func files(for quant: Quant) -> [String] {
        let lane = quant == .fp32 ? "fp32" : "fp16"
        return ["nacre_v1_mlx_\(lane).safetensors", "vq_f4_mlx_\(lane).safetensors", "config.json"]
    }

    public func resolvedWeightsDirectory(storeRoot: URL?) -> URL? {
        weightsDirectory ?? ModelStore(root: storeRoot).directory(for: Self.repo)
    }

    private enum CodingKeys: String, CodingKey { case quant, tileSize, tileOverlap, seed }
}

/// Fresh-machine sources (contract 1.24: the ENGINE downloads them into the store before `load()`). One role per lane,
/// matching only that lane's files.
extension NacreConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        let lane = effectiveQuant == .fp32 ? "fp32" : "fp16"
        return [WeightSource(role: "nacre-v1-\(lane)", repo: Self.repo, revision: "main",
                             matching: Self.files(for: effectiveQuant))]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let dir = weightsDirectory {
            let complete = Self.files(for: effectiveQuant).allSatisfy {
                FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
            }
            return complete ? [] : weightSources
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}

extension NacreConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolvedWeightsDirectory(storeRoot: modelsRootDirectory) else { return [] }
        return Self.files(for: effectiveQuant).filter { $0.hasSuffix(".safetensors") }.map { dir.appendingPathComponent($0) }
    }
}
