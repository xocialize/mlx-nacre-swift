// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
//
// The residual-shift sampler (reference `nacre/diffusion/residual_shift.py`, transcribed from Yue et al., ResShift,
// NeurIPS 2023) and the end-to-end 4× upscale (reference `tools/benchmark.py` `Upscaler`).
import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - schedule

/// The residual-shift chain's scalars, computed in Double exactly as the reference computes them in numpy float64.
public struct ResidualShiftSchedule: Sendable, Equatable {
    public let numTimesteps: Int
    public let kappa: Double
    public let sqrtEtas: [Double]
    public let etas: [Double]
    public let posteriorMeanCoef1: [Double]
    public let posteriorMeanCoef2: [Double]
    public let posteriorLogVarianceClipped: [Double]

    public init(numTimesteps: Int = 4, kappa: Double = 2.0, power: Double = 0.3, minNoiseLevel: Double = 0.2,
                etasEnd: Double = 0.99) {
        self.numTimesteps = numTimesteps
        self.kappa = kappa
        let lo = Foundation.log(minNoiseLevel / kappa), hi = Foundation.log(etasEnd)
        let denom = Double(max(numTimesteps - 1, 1))
        let se = (0 ..< numTimesteps).map { t in Foundation.exp(lo + (hi - lo) * Foundation.pow(Double(t) / denom, power)) }
        sqrtEtas = se
        let e = se.map { $0 * $0 }
        etas = e
        let prev = [0.0] + e.dropLast()
        let alpha = zip(e, prev).map { $0 - $1 }
        let pv = (0 ..< numTimesteps).map { kappa * kappa * prev[$0] / e[$0] * alpha[$0] }
        let pvc = [pv[1]] + pv.dropFirst()
        posteriorLogVarianceClipped = pvc.map { Foundation.log($0) }
        posteriorMeanCoef1 = zip(prev, e).map { $0 / $1 }
        posteriorMeanCoef2 = zip(alpha, e).map { $0 / $1 }
    }
}

// MARK: - weights

public enum NacreWeights {
    /// Strict load of an MLX-layout file (oracle/convert_weights.py): keys must equal the module tree's flattened keys
    /// — 0 missing / 0 unused — and shapes must match, else nothing is applied. Materialised on the CPU stream.
    public static func load(_ module: Module, from url: URL, dtype: DType? = nil) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { throw NacreError.weightsNotFound(url.path) }
        let raw: [String: MLXArray]
        do { raw = try Device.withDefaultDevice(.cpu) { try MLX.loadArrays(url: url) } }
        catch { throw NacreError.loadFailed(String(describing: error)) }
        let sd = dtype.map { dt in raw.mapValues { $0.dtype == dt ? $0 : $0.asType(dt) } } ?? raw
        let expected = Set(module.parameters().flattened().map(\.0))
        let got = Set(sd.keys)
        let missing = expected.subtracting(got).sorted(), extra = got.subtracting(expected).sorted()
        guard missing.isEmpty && extra.isEmpty else { throw NacreError.parameterMismatch(missing: missing, extra: extra) }
        do {
            try Device.withDefaultDevice(.cpu) {
                try module.update(parameters: ModuleParameters.unflattened(sd), verify: .all)
                eval(module)
            }
        } catch { throw NacreError.loadFailed(String(describing: error)) }
    }
}

// MARK: - the model

/// Nacre v1: SwinUNet denoiser + VQ-f4 + residual-shift chain. Input/output NHWC RGB in [0, 1]; output 4× the input.
public final class Nacre {
    public let unet: SwinUNet
    public let vq: VQAutoencoder
    public let schedule: ResidualShiftSchedule
    public static let scale = 4
    /// LQ sides are padded (reflect) to a multiple of this: four U-Net levels (/8), then 8-wide shifted windows.
    public static let pad = 64

    public init(unet: SwinUNet = SwinUNet(), vq: VQAutoencoder = VQAutoencoder(),
                schedule: ResidualShiftSchedule = ResidualShiftSchedule()) {
        self.unet = unet
        self.vq = vq
        self.schedule = schedule
    }

    public func loadWeights(unet unetURL: URL, vq vqURL: URL, dtype: DType? = nil, vqDtype: DType? = nil) throws {
        try NacreWeights.load(unet, from: unetURL, dtype: dtype)
        try NacreWeights.load(vq, from: vqURL, dtype: vqDtype ?? dtype)
    }

    var unetDtype: DType { (unet.timeEmbed[0] as! Linear).weight.dtype }
    var vqDtype: DType { vq.quantConv.weight.dtype }

    /// `scale_input`: x_t / sqrt(κ²·η_t + 1).
    func scaleInput(_ x: MLXArray, _ t: Int) -> MLXArray {
        x / Float((schedule.etas[t] * schedule.kappa * schedule.kappa + 1).squareRoot())
    }

    /// The reverse chain from the degraded latent `y0`. `noise[0]` is the prior draw, `noise[1...]` the per-step
    /// draws (t = T-1 … 1). `checkpoint` runs after every step (cooperative cancellation); `onStep(done, total)`.
    public func sample(y0: MLXArray, cond: MLXArray, noise: [MLXArray],
                       checkpoint: (() throws -> Void)? = nil,
                       onStep: ((Int, Int) -> Void)? = nil,
                       taps: ((Int, MLXArray) -> Void)? = nil) rethrows -> MLXArray {
        let s = schedule, T = s.numTimesteps
        precondition(noise.count == T, "need \(T) noise draws (prior + \(T - 1) chain)")
        var x = y0 + Float(s.kappa * s.sqrtEtas[T - 1]) * noise[0]
        var draw = 1
        for step in stride(from: T - 1, through: 0, by: -1) {
            let t = MLXArray([Int32(step)])
            let x0 = unet(scaleInput(x, step), t: t, lq: cond)
            let mean = Float(s.posteriorMeanCoef1[step]) * x + Float(s.posteriorMeanCoef2[step]) * x0
            if step == 0 {
                x = mean
            } else {
                x = mean + Float(Foundation.exp(0.5 * s.posteriorLogVarianceClipped[step])) * noise[draw]
                draw += 1
            }
            eval(x)
            taps?(step, x)
            try checkpoint?()
            onStep?(T - step, T)
        }
        return x
    }

    /// Seeded noise for a latent of `shape` (prior + chain draws), from one MLXRandom key — reproducible per seed.
    public func noise(shape: [Int], seed: UInt64, dtype: DType = .float32) -> [MLXArray] {
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: schedule.numTimesteps)
        return (0 ..< schedule.numTimesteps).map { MLXRandom.normal(shape, dtype: dtype, key: keys[$0]) }
    }

    /// Whole-image 4× restore. `lq01`: [1, h, w, 3] in [0, 1]. Pads to a multiple of 64 (reflect), runs, crops.
    public func upscale(_ lq01: MLXArray, seed: UInt64 = 20260923, injectedNoise: [MLXArray]? = nil,
                        checkpoint: (() throws -> Void)? = nil,
                        onStep: ((Int, Int) -> Void)? = nil) rethrows -> MLXArray {
        let (h, w) = (lq01.dim(1), lq01.dim(2))
        let ph = (Self.pad - h % Self.pad) % Self.pad, pw = (Self.pad - w % Self.pad) % Self.pad
        let dt = unetDtype, vdt = vqDtype
        let lq = reflectPad(lq01.asType(.float32), bottom: ph, right: pw)
        let up = clip(bicubicUpsample(lq, scale: Self.scale), min: 0, max: 1)
        let y0 = vq.encode((up * 2 - 1).asType(vdt)).asType(dt)
        eval(y0)
        try checkpoint?()
        let cond = (lq * 2 - 1).asType(dt)
        let n = injectedNoise ?? noise(shape: y0.shape, seed: seed, dtype: dt)
        let z = try sample(y0: y0, cond: cond, noise: n.map { $0.asType(dt) }, checkpoint: checkpoint, onStep: onStep)
        let img = clip((vq.decode(z.asType(vdt)).asType(.float32) + 1) / 2, min: 0, max: 1)
        return img[0..., 0 ..< (Self.scale * h), 0 ..< (Self.scale * w), 0...]
    }
}

// MARK: - tiled inference

extension Nacre {
    /// Tiled 4× restore: the whole pipeline (bicubic → encode → chain → decode) runs per LQ tile of `tile`² with
    /// `overlap` LQ pixels shared between neighbours, and the 4× outputs are blended with linear ramps across the
    /// overlap (weights sum to 1 everywhere). The ResShift recipe (its inference chops the LQ the same way); peak memory
    /// is set by the tile, not the image. Each tile draws its own noise from `seed` mixed with the tile's origin, so a
    /// given (image, seed, tiling) is reproducible. Images no larger than one tile take the whole-image path.
    public func upscaleTiled(_ lq01: MLXArray, tile: Int = 128, overlap: Int = 32, seed: UInt64 = 20260923,
                             checkpoint: (() throws -> Void)? = nil,
                             onTile: ((Int, Int) -> Void)? = nil) rethrows -> MLXArray {
        let (h, w) = (lq01.dim(1), lq01.dim(2))
        if h <= tile && w <= tile { return try upscale(lq01, seed: seed, checkpoint: checkpoint) }
        precondition(overlap * 2 < tile, "overlap must be < tile/2")
        func starts(_ n: Int) -> [Int] {
            if n <= tile { return [0] }
            let stride = tile - overlap
            var s = Array(Swift.stride(from: 0, to: n - tile, by: stride))
            s.append(n - tile)
            return s
        }
        let ys = starts(h), xs = starts(w), s = Self.scale
        var acc = MLXArray.zeros([1, s * h, s * w, 3])
        var wsum = MLXArray.zeros([1, s * h, s * w, 1])
        let total = ys.count * xs.count
        var done = 0
        for y in ys {
            for x in xs {
                let th = min(tile, h - y), tw = min(tile, w - x)
                let lq = lq01[0..., y ..< (y + th), x ..< (x + tw), 0...]
                let tileSeed = seed &+ UInt64(y) &* 0x9E37_79B9 &+ UInt64(x) &* 0x85EB_CA6B
                let out = try upscale(lq, seed: tileSeed, checkpoint: checkpoint)
                // ramp weights: 1 in the interior, linear to ~0 across each overlap that touches a neighbour
                let wy = Self.ramp(th * s, lead: y > 0 ? overlap * s : 0, trail: y + th < h ? overlap * s : 0)
                let wx = Self.ramp(tw * s, lead: x > 0 ? overlap * s : 0, trail: x + tw < w ? overlap * s : 0)
                let wt = (wy.reshaped([1, th * s, 1, 1]) * wx.reshaped([1, 1, tw * s, 1]))
                let (oy, ox) = (y * s, x * s)
                acc[0..., oy ..< (oy + th * s), ox ..< (ox + tw * s), 0...] =
                    acc[0..., oy ..< (oy + th * s), ox ..< (ox + tw * s), 0...] + out * wt
                wsum[0..., oy ..< (oy + th * s), ox ..< (ox + tw * s), 0...] =
                    wsum[0..., oy ..< (oy + th * s), ox ..< (ox + tw * s), 0...] + wt
                eval(acc, wsum)
                Memory.clearCache()
                done += 1
                onTile?(done, total)
                try checkpoint?()
            }
        }
        return acc / wsum
    }

    /// 1-D blend ramp of length n: rises over `lead`, falls over `trail` (half-sample offsets keep it > 0).
    static func ramp(_ n: Int, lead: Int, trail: Int) -> MLXArray {
        var v = [Float](repeating: 1, count: n)
        for i in 0 ..< lead { v[i] = (Float(i) + 0.5) / Float(lead) }
        for i in 0 ..< trail { v[n - 1 - i] = min(v[n - 1 - i], (Float(i) + 0.5) / Float(trail)) }
        return MLXArray(v)
    }
}
