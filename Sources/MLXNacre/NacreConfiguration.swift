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
public struct NacreConfiguration: PackageConfiguration, ModelStorable, QuantConfigured, BudgetAware {
    /// `.fp16` (default) or `.fp32`; anything else is treated as `.fp16`.
    public var quant: Quant
    /// LQ tile side for the tiled path (whole pipeline per tile, linear-ramp blend); `nil` = 128 (PORTING-SPEC S3c:
    /// peak set by the tile; better fidelity than whole-image). Multiples of 64 required.
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
    /// fp32's measured working set is ~5.5 GB (PORTING-SPEC S4); below 8 GB of headroom the fp16 lane loads instead.
    public static let fp32MinBudgetBytes: UInt64 = 8_000_000_000

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
