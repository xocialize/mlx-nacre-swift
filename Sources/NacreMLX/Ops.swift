// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
import Foundation
import MLX
import MLXNN

// MARK: - errors

public enum NacreError: Error, CustomStringConvertible {
    case weightsNotFound(String)
    case loadFailed(String)
    case parameterMismatch(missing: [String], extra: [String])
    case badInput(String)

    public var description: String {
        switch self {
        case .weightsNotFound(let p): return "weights not found: \(p)"
        case .loadFailed(let m): return "weight load failed: \(m)"
        case .parameterMismatch(let m, let e):
            return "checkpoint/module key mismatch — missing \(m.count) \(m.prefix(5)), extra \(e.count) \(e.prefix(5))"
        case .badInput(let m): return "bad input: \(m)"
        }
    }
}

// MARK: - timestep embedding

/// `timestep_embedding` (nn_blocks.py): sinusoidal, cos half first then sin. `[N] -> [N, dim]`.
/// The frequency table is computed as torch does — float32 `arange / half`, scaled by float32 `-ln(10000)`, `exp` —
/// so the table is the same numbers, not a Double-rounded neighbour.
public func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Float = 10000) -> MLXArray {
    let half = dim / 2
    let freqs = exp(-Float(log(Double(maxPeriod))) * MLXArray(0 ..< half).asType(.float32) / Float(half))
    let args = t.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    var e = concatenated([cos(args), sin(args)], axis: -1)
    if dim % 2 == 1 { e = concatenated([e, zeros([e.dim(0), 1])], axis: -1) }
    return e
}

// MARK: - layout helpers (NHWC)

/// Nearest-neighbour 2× (torch `F.interpolate(scale_factor=2, mode="nearest")`) on NHWC.
public func nearestUp2(_ x: MLXArray) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let y = broadcast(x.reshaped([b, h, 1, w, 1, c]), to: [b, h, 2, w, 2, c])
    return y.reshaped([b, 2 * h, 2 * w, c])
}

/// torch `F.pad(x, (l, r, t, b), mode="reflect")` on NHWC, by concatenating mirrored slices (no edge repeat).
public func reflectPad(_ x: MLXArray, bottom: Int, right: Int) -> MLXArray {
    var y = x
    if bottom > 0 {
        let h = y.dim(1)
        precondition(bottom < h, "reflect pad must be smaller than the dimension")
        let tail = y[0..., (h - 1 - bottom) ..< (h - 1), 0..., 0...]
        y = concatenated([y, tail[0..., .stride(by: -1), 0..., 0...]], axis: 1)
    }
    if right > 0 {
        let w = y.dim(2)
        precondition(right < w, "reflect pad must be smaller than the dimension")
        let tail = y[0..., 0..., (w - 1 - right) ..< (w - 1), 0...]
        y = concatenated([y, tail[0..., 0..., .stride(by: -1), 0...]], axis: 2)
    }
    return y
}

// MARK: - bicubic ×s as a sub-pixel convolution (lifted from mlx-nerve-swift, Apache-2.0, same author)

/// Torch's `F.interpolate(x, scale_factor=s, mode="bicubic", align_corners=False)` (Keys a = −0.75, half-pixel
/// centres, tap indices clamped to the image, no antialias) at an INTEGER scale `s` is a fixed sub-pixel convolution:
/// edge-pad by 2 → a VALID 5×5 conv taking colour `c` to its `s²` phase channels → pixel shuffle. One output-sized
/// tensor instead of `MLXNN.Upsample(.cubic)`'s 16 output-resolution gathers. Gated against torch at S1.
final class BicubicPhaseKernel: @unchecked Sendable {   // a box — Module reflection must never see the constant
    let scale: Int
    let weight: MLXArray   // [3·s², 5, 5, 3], OHWI, block-diagonal over colour

    init(scale s: Int) {
        let w = Self.weights1D(scale: s)
        var k = [Float](repeating: 0, count: 3 * s * s * 5 * 5 * 3)
        for c in 0 ..< 3 {
            for p in 0 ..< s {
                for q in 0 ..< s {
                    let o = c * s * s + p * s + q
                    for a in 0 ..< 5 {
                        for b in 0 ..< 5 { k[((o * 5 + a) * 5 + b) * 3 + c] = Float(w[p][a] * w[q][b]) }
                    }
                }
            }
        }
        scale = s
        weight = MLXArray(k, [3 * s * s, 5, 5, 3])
    }

    static func weights1D(scale s: Int) -> [[Double]] {
        let A = -0.75
        func cc1(_ x: Double) -> Double { ((A + 2) * x - (A + 3)) * x * x + 1 }
        func cc2(_ x: Double) -> Double { ((A * x - 5 * A) * x + 8 * A) * x - 4 * A }
        return (0 ..< s).map { p in
            let delta = (Double(p) + 0.5) / Double(s) - 0.5
            let t = delta < 0 ? delta + 1 : delta
            let four = [cc2(t + 1), cc1(t), cc1(1 - t), cc2(2 - t)]
            return delta < 0 ? four + [0] : [0] + four
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [Int: BicubicPhaseKernel] = [:]

    static func forScale(_ s: Int) -> BicubicPhaseKernel {
        lock.lock(); defer { lock.unlock() }
        if let k = cache[s] { return k }
        let k = BicubicPhaseKernel(scale: s)
        cache[s] = k
        return k
    }
}

/// Channel-last pixel shuffle in torch's channel order (`c·r·r + i·r + j`).
public func pixelShuffleNHWC(_ x: MLXArray, _ r: Int) -> MLXArray {
    let (b, h, w, crr) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let c = crr / (r * r)
    return x.reshaped([b, h, w, c, r, r]).transposed(0, 1, 4, 2, 5, 3).reshaped([b, h * r, w * r, c])
}

/// `F.interpolate(x, scale_factor=s, mode="bicubic", align_corners=False)` on NHWC RGB.
public func bicubicUpsample(_ x: MLXArray, scale: Int) -> MLXArray {
    let k = BicubicPhaseKernel.forScale(scale)
    let w = k.weight.dtype == x.dtype ? k.weight : k.weight.asType(x.dtype)
    return pixelShuffleNHWC(conv2d(padded(x, widths: [0, 2, 2, 0], mode: .edge), w), scale)
}
