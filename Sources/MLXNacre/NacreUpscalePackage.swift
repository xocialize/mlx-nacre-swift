// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX
import MLXNN
import MLXToolKit
import NacreMLX
import UniformTypeIdentifiers

public enum NacrePackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
    case weightsDirectoryUnresolved
}

/// An MLXEngine `imageUpscale` package over **Nacre v1** (Xocialize, Apache-2.0): a 118.6 M-parameter SwinUNet driving a
/// 4-step residual-shift diffusion in the CompVis VQ-f4 latent space (MIT) — the provenance-clean GENERATIVE stills
/// tier (trained only on Wikimedia Commons CC0 / PD / CC BY images; AB-R-0396). At parity with ResShift v3 on neutral
/// benchmarks: fidelity level or better, MUSIQ ahead, CLIPIQA 0–0.04 behind.
///
/// ⚠️ Generative: invents plausible detail, and — like every generative upscaler measured — can damage text that was
/// already legible (SA-Text). Hosts should route legible small text to a fidelity tier.
///
/// Native scale is **4×**. A request `scale` below 4 is honored by post-downsampling the 4× result.
/// Born sweep-clean (split footprint per lane, `QuantConfigured`, `BudgetAware` fp32 → fp16, `unload()` flushes the
/// pool), materialization-clean (`WeightSourcing`), cancel-clean (entry checkpoint + one per diffusion step of every
/// tile, `RunProgress` at the same seam).
@InferenceActor
public final class NacreUpscalePackage: ModelPackage {
    public typealias Configuration = NacreConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the denoiser weights are Apache-2.0 (ours, trained from scratch); the VQ-f4 autoencoder weights are
            // CompVis MIT (NOTICE) — both permissive. C8: the port is our own Apache-2.0 code.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(sourceRepo: NacreConfiguration.repo, revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14), RE-BASELINED IN-APP (AB-T-0019, 2026-10-05): Nacre Demo, Release,
                // ValidationHarness on the app's own MLXServeEngine (isolate, engine GPU pool cap 2 GB), M5 Max, idle box,
                // tiled 128/32, RealSR Nikon_010 500×400 → 2000×1600, one fresh process per number. resident = post-load
                // phys − app baseline (0.05 GB); activation = the KERNEL's lifetime phys peak (ledger_phys_footprint_peak)
                // − post-load floor — the harness's 150 ms sampler under-read that peak by 0.2–0.45 GB on every run.
                //   fp16 ×6: floor 0.42 GB (MLX active 0.33 GB = the weights), kernel peak 4.57–4.70 GB, run 9.7–10.9 s
                //   fp32 ×3: floor 0.77–0.82 GB (MLX active 0.65 GB),           kernel peak 5.54–5.58 GB, run 13.4–14.5 s
                // The v0.1.1 CLI split (0.80 + 3.85 / 1.05 + 4.5) had the right TOTAL but read "resident" after the run,
                // folding in a 0.30–0.36 GB post-first-run residue that is not MLX memory (pool active == weights, cache 0).
                // Declared at the measured MAX, never the median: v0.1.2 shipped fp16 4.25 GB from the first three runs
                // and the next three reached 4.28 GB (run-to-run spread ~0.13 GB). The tile sets the activation, so it does
                // not grow with the image.
                footprints: [
                    QuantFootprint(quant: .fp16, residentBytes: 400_000_000, peakActivationBytes: 4_300_000_000),
                    QuantFootprint(quant: .fp32, residentBytes: 780_000_000, peakActivationBytes: 4_800_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageUpscaleContract.descriptor(
                    name: "nacre-upscale",
                    summary: "Nacre 4x generative super-resolution (ResShift-class residual-shift diffusion, 4 steps; Apache-2.0, trained only on Commons CC0/PD/CC-BY images): restores heavily degraded real-world photos with plausible fine detail. Generative — keep already-legible small text on a fidelity tier."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var nacre: Nacre?
    private var loadedQuant: Quant?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public nonisolated var plannedQuant: Quant {
        if configuration.effectiveQuant == .fp32, let b = configuration.availableBudgetBytes,
           b < NacreConfiguration.fp32MinBudgetBytes {
            return .fp16
        }
        return configuration.effectiveQuant
    }

    public func load() async throws {
        guard nacre == nil else { return }
        guard let dir = configuration.resolvedWeightsDirectory(storeRoot: configuration.modelsRootDirectory) else {
            throw NacrePackageError.weightsDirectoryUnresolved
        }
        let quant = plannedQuant
        let files = NacreConfiguration.files(for: quant)
        let model = Nacre()
        // Contract 1.24: the engine has already materialized the lane's files into `dir`; this just loads (strict key
        // contract, CPU-stream load).
        try model.loadWeights(unet: dir.appendingPathComponent(files[0]), vq: dir.appendingPathComponent(files[1]))
        nacre = model
        loadedQuant = quant
    }

    /// C14 seam: the loaded graphs by role (`nil` before `load()`).
    var inferenceModeGraphs: [String: Module?] { ["unet": nacre?.unet, "vq": nacre?.vq] }

    public func unload() async {
        nacre = nil
        loadedQuant = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation.
        try Task.checkCancellation()
        guard let nacre else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale, let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let rgb = try Self.decodeRGB(req.image)
        let (inW, inH) = (rgb.dim(2), rgb.dim(1))
        let tile = configuration.tileSize ?? Nacre.defaultTile
        let overlap = configuration.tileOverlap ?? Nacre.defaultOverlap
        // The tiler preconditions this; a precondition in a package kills the HOST app, so refuse it here instead.
        try Self.validateTiling(tile: tile, overlap: overlap)
        // CAN-2: a cooperative checkpoint after every diffusion step of every tile (rethrown unchanged); RunProgress
        // at the same seam — step = (tile-1)·steps + step over tiles·steps, stage = tile.
        var out = try nacre.upscaleTiled(rgb, tile: tile, overlap: overlap, seed: configuration.seed,
                                         checkpoint: { try Task.checkCancellation() },
                                         onProgress: { t, tiles, s, steps in
                                             RunProgress.report(.denoise, step: (t - 1) * steps + s,
                                                                totalSteps: tiles * steps, stage: t, totalStages: tiles)
                                         })
        var applied = Nacre.scale
        if let s = req.scale, s > 0, s < Nacre.scale {
            out = Self.downsample(out, toWidth: inW * s, height: inH * s)
            applied = s
        }
        let image: Image
        if req.image.format == .rawBGRA8 {
            image = Self.encodeRawBGRA8(out)
        } else {
            guard let png = Self.encodePNG(out) else { throw NacrePackageError.imageEncodeFailed }
            image = Image(format: .png, data: png, width: out.dim(2), height: out.dim(1))
        }
        return ImageUpscaleResponse(image: image, appliedScale: applied)
    }

    // MARK: - image codec (canonical Image ↔ [1, H, W, 3] float32 RGB in [0, 1])

    public nonisolated static func decodeRGB(_ image: Image) throws -> MLXArray {
        let w: Int, h: Int
        var rgba: [UInt8]
        if image.format == .rawBGRA8 {
            guard let iw = image.width, let ih = image.height, iw > 0, ih > 0 else {
                throw NacrePackageError.imageDecodeFailed("rawBGRA8 requires width/height")
            }
            let stride = image.bytesPerRow ?? iw * 4
            guard stride >= iw * 4, image.data.count >= stride * ih else {
                throw NacrePackageError.imageDecodeFailed("rawBGRA8 data too small")
            }
            (w, h) = (iw, ih)
            rgba = [UInt8](repeating: 0, count: w * h * 4)
            image.data.withUnsafeBytes { src in
                let b = src.bindMemory(to: UInt8.self)
                for y in 0 ..< h {
                    for x in 0 ..< w {
                        let s = y * stride + x * 4, d = (y * w + x) * 4
                        rgba[d] = b[s + 2]; rgba[d + 1] = b[s + 1]; rgba[d + 2] = b[s]; rgba[d + 3] = 255
                    }
                }
            }
        } else {
            guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                throw NacrePackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
            }
            (w, h) = (cg.width, cg.height)
            rgba = [UInt8](repeating: 0, count: w * h * 4)
            guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
                throw NacrePackageError.imageDecodeFailed("CGContext")
            }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        let a = MLXArray(rgba, [1, h, w, 4]).asType(.float32) / 255
        return a[0..., 0..., 0..., 0 ..< 3]
    }

    nonisolated static func rgba8(_ x: MLXArray) -> ([UInt8], Int, Int) {
        let (h, w) = (x.dim(1), x.dim(2))
        let rgb = clip(x[0] * 255 + 0.5, min: 0, max: 255).asType(.uint8)
        let rgba = concatenated([rgb, full([h, w, 1], values: MLXArray(UInt8(255)))], axis: -1)
        return (rgba.asArray(UInt8.self), w, h)
    }

    public nonisolated static func encodePNG(_ x: MLXArray) -> Data? {
        var (bytes, w, h) = rgba8(x)
        guard let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let cg = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    public nonisolated static func encodeRawBGRA8(_ x: MLXArray) -> Image {
        let (rgba, w, h) = rgba8(x)
        var bgra = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0 ..< w * h {
            bgra[i * 4] = rgba[i * 4 + 2]; bgra[i * 4 + 1] = rgba[i * 4 + 1]; bgra[i * 4 + 2] = rgba[i * 4]
        }
        return Image.rawBGRA8(data: Data(bgra), width: w, height: h)
    }

    /// High-quality downsample of the 4× result to `w`×`h` (CoreGraphics, high interpolation).
    nonisolated static func downsample(_ x: MLXArray, toWidth w: Int, height h: Int) -> MLXArray {
        var (bytes, sw, sh) = rgba8(x)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let info = CGImageAlphaInfo.noneSkipLast.rawValue
        guard let src = CGContext(data: &bytes, width: sw, height: sh, bitsPerComponent: 8, bytesPerRow: sw * 4,
                                  space: space, bitmapInfo: info)?.makeImage() else { return x }
        var out = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &out, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: info) else { return x }
        ctx.interpolationQuality = .high
        ctx.draw(src, in: CGRect(x: 0, y: 0, width: w, height: h))
        return MLXArray(out, [1, h, w, 4]).asType(.float32)[0..., 0..., 0..., 0 ..< 3] / 255
    }
}

extension NacreUpscalePackage {
    /// Tile ≥ 16 LQ px and 0 ≤ overlap < tile / 2 (`Nacre.upscaleTiled`'s blend needs a non-overlapping core).
    public nonisolated static func validateTiling(tile: Int, overlap: Int) throws {
        guard tile >= 16, overlap >= 0, overlap * 2 < tile else {
            throw NacreError.badInput("tiling: need tile ≥ 16 and 0 ≤ overlap < tile/2 (got tile \(tile), overlap \(overlap))")
        }
    }

    public nonisolated static var registration: PackageRegistration { .of(NacreUpscalePackage.self) }
}
