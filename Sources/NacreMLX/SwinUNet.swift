// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
//
// Isomorphic port of the reference `nacre/models/{nn_blocks,swin,unet}.py` (Apache-2.0; its own lineage: OpenAI
// improved-diffusion, MIT; Swin Transformer, MIT; SwinIR, Apache-2.0). NHWC throughout. Every module's property keys
// reproduce the checkpoint's parameter paths, so the release loads with 0 missing / 0 unused keys (S0).
import Foundation
import MLX
import MLXNN

// MARK: - protocol for the timestep-conditioned sequential

/// A layer inside `TimestepEmbedSequential`: ResBlocks take the embedding, everything else ignores it.
protocol NacreLayer {
    func forward(_ x: MLXArray, emb: MLXArray) -> MLXArray
}

/// `GroupNorm32` (nn_blocks.py) — accumulate in fp32, cast back; torch eps 1e-5. Parameters at `weight`/`bias`.
/// A torch GroupNorm whose parameters live directly at `weight` / `bias` (no wrapper level in the key path).
final class TorchGroupNorm: Module, UnaryLayer, NacreLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let groups: Int
    let eps: Float
    let fp32: Bool

    init(_ channels: Int, groups: Int = 32, eps: Float = 1e-5, fp32: Bool = true) {
        _weight.wrappedValue = ones([channels])
        _bias.wrappedValue = zeros([channels])
        self.groups = groups
        self.eps = eps
        self.fp32 = fp32
    }

    /// NHWC GroupNorm with torch's grouping (channels `g·cpg ..< (g+1)·cpg` form group g).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inDtype = x.dtype
        let xx = fp32 && inDtype != .float32 ? x.asType(.float32) : x
        let (b, h, w, c) = (xx.dim(0), xx.dim(1), xx.dim(2), xx.dim(3))
        let g = xx.reshaped([b, h * w, groups, c / groups])
        let mean = Self.groupMean(g)
        let centered = g - mean
        let v = Self.groupMean(centered * centered)
        let n = (centered * rsqrt(v + eps)).reshaped([b, h, w, c])
        let y = n * weight.asType(n.dtype) + bias.asType(n.dtype)
        return y.dtype == inDtype ? y : y.asType(inDtype)
    }

    func forward(_ x: MLXArray, emb: MLXArray) -> MLXArray { callAsFunction(x) }

    /// Mean over axes (1, 3) of `[b, M, groups, cpg]`, summed in two levels (blocks of ≤ 1024 rows, then the blocks).
    /// MLX's CPU stream reduces sequentially (AB-L-0180), and the VQ encoder's GroupNorms reduce up to 262k values per
    /// group at full image resolution: one flat fp32 sum drifted the encode 6× past torch's own fp32 noise on the CPU
    /// parity lane. The two-level sum is the same mathematics on either device.
    static func groupMean(_ g: MLXArray) -> MLXArray {
        let (b, m, gr, cpg) = (g.dim(0), g.dim(1), g.dim(2), g.dim(3))
        let block = 1024
        guard m > block, m % block == 0 else { return g.mean(axes: [1, 3], keepDims: true) }
        let partial = g.reshaped([b, m / block, block, gr, cpg]).sum(axes: [2, 4])   // [b, m/block, gr]
        return (partial.sum(axis: 1) / Float(m * cpg)).reshaped([b, 1, gr, 1])
    }
}

/// Parameter-free placeholders that keep sequential indices aligned with the checkpoint (`SiLU` at index 1, etc.).
final class SiLUBox: Module, UnaryLayer, NacreLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { silu(x) }
    func forward(_ x: MLXArray, emb: MLXArray) -> MLXArray { silu(x) }
}

final class IdentityBox: Module, UnaryLayer, NacreLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }
    func forward(_ x: MLXArray, emb: MLXArray) -> MLXArray { x }
}

extension Conv2d: NacreLayer {
    func forward(_ x: MLXArray, emb: MLXArray) -> MLXArray { callAsFunction(x) }
}

extension Linear: NacreLayer {
    func forward(_ x: MLXArray, emb: MLXArray) -> MLXArray { callAsFunction(x) }
}

// MARK: - resampling

/// `Downsample` (nn_blocks.py) with `use_conv`: strided 3×3, padding 1. Key `op`.
final class UNetDownsample: Module, NacreLayer {
    @ModuleInfo(key: "op") var op: Conv2d
    init(_ ch: Int, out: Int) {
        _op.wrappedValue = Conv2d(inputChannels: ch, outputChannels: out, kernelSize: 3, stride: 2, padding: 1)
    }
    func forward(_ x: MLXArray, emb: MLXArray) -> MLXArray { op(x) }
}

/// `Upsample` (nn_blocks.py) with `use_conv`: nearest 2×, 3×3. Key `conv`.
final class UNetUpsample: Module, NacreLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ ch: Int, out: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: ch, outputChannels: out, kernelSize: 3, padding: 1)
    }
    func forward(_ x: MLXArray, emb: MLXArray) -> MLXArray { conv(nearestUp2(x)) }
}

// MARK: - ResBlock

/// `ResBlock` (nn_blocks.py), scale-shift norm, no up/down (the released config uses conv resamplers).
/// Keys: in_layers.{0,2} · emb_layers.1 · out_layers.{0,3} · skip_connection (when channels change).
final class ResBlock: Module, NacreLayer {
    @ModuleInfo(key: "in_layers") var inLayers: [Module]       // [GroupNorm32, SiLU, Conv3×3]
    @ModuleInfo(key: "emb_layers") var embLayers: [Module]     // [SiLU, Linear]
    @ModuleInfo(key: "out_layers") var outLayers: [Module]     // [GroupNorm32, SiLU, Dropout(=id), Conv3×3]
    @ModuleInfo(key: "skip_connection") var skip: Conv2d?
    let outChannels: Int

    init(_ ch: Int, embChannels: Int, out: Int? = nil, groups: Int = 32) {
        let o = out ?? ch
        outChannels = o
        _inLayers.wrappedValue = [TorchGroupNorm(ch, groups: groups), SiLUBox(),
                                  Conv2d(inputChannels: ch, outputChannels: o, kernelSize: 3, padding: 1)]
        _embLayers.wrappedValue = [SiLUBox(), Linear(embChannels, 2 * o)]
        _outLayers.wrappedValue = [TorchGroupNorm(o, groups: groups), SiLUBox(), IdentityBox(),
                                   Conv2d(inputChannels: o, outputChannels: o, kernelSize: 3, padding: 1)]
        _skip.wrappedValue = o == ch ? nil : Conv2d(inputChannels: ch, outputChannels: o, kernelSize: 1)
    }

    func forward(_ x: MLXArray, emb: MLXArray) -> MLXArray {
        var h = (inLayers[0] as! TorchGroupNorm)(x)
        h = silu(h)
        h = (inLayers[2] as! Conv2d)(h)
        var e = (embLayers[1] as! Linear)(silu(emb))
        if e.dtype != h.dtype { e = e.asType(h.dtype) }
        e = e.reshaped([e.dim(0), 1, 1, e.dim(1)])
        let scale = e[0..., 0..., 0..., 0 ..< outChannels]
        let shift = e[0..., 0..., 0..., outChannels...]
        h = (outLayers[0] as! TorchGroupNorm)(h) * (1 + scale) + shift
        h = (outLayers[3] as! Conv2d)(silu(h))
        return (skip.map { $0(x) } ?? x) + h
    }
}

// MARK: - Swin

/// The shifted-window additive mask (swin.py `build_attn_mask`, the trained TWO-axis form). Pure Swift, cached per
/// (h, w, ws, shift) — a constant, never a parameter. `[nW, N, N]` with −100 across a seam, 0 within a region.
final class SwinMaskCache: @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [String: MLXArray] = [:]

    static func mask(h: Int, w: Int, ws: Int, shift: Int) -> MLXArray {
        let key = "\(h)x\(w)/\(ws)/\(shift)"
        lock.lock(); defer { lock.unlock() }
        if let m = cache[key] { return m }
        // region id per pixel: three bands per axis — [0, h-ws), [h-ws, h-shift), [h-shift, h)
        func band(_ i: Int, _ n: Int) -> Int { i < n - ws ? 0 : (i < n - shift ? 1 : 2) }
        let nWh = h / ws, nWw = w / ws, n = ws * ws
        var out = [Float](repeating: 0, count: nWh * nWw * n * n)
        for wy in 0 ..< nWh {
            for wx in 0 ..< nWw {
                let wi = wy * nWw + wx
                var ids = [Int](repeating: 0, count: n)
                for py in 0 ..< ws {
                    for px in 0 ..< ws {
                        ids[py * ws + px] = band(wy * ws + py, h) * 3 + band(wx * ws + px, w)
                    }
                }
                for i in 0 ..< n {
                    for j in 0 ..< n where ids[i] != ids[j] { out[(wi * n + i) * n + j] = -100 }
                }
            }
        }
        let m = MLXArray(out, [nWh * nWw, n, n])
        cache[key] = m
        return m
    }
}

/// Relative-position index for a window (swin.py `WindowAttention.__init__`), a constant box.
final class RelativePositionIndex: @unchecked Sendable {
    let index: MLXArray   // [N*N] int32
    init(window ws: Int) {
        let n = ws * ws
        var idx = [Int32](repeating: 0, count: n * n)
        for i in 0 ..< n {
            for j in 0 ..< n {
                let dy = (i / ws) - (j / ws) + ws - 1
                let dx = (i % ws) - (j % ws) + ws - 1
                idx[i * n + j] = Int32(dy * (2 * ws - 1) + dx)
            }
        }
        index = MLXArray(idx)
    }
}

/// `WindowAttention` (swin.py). Keys: qkv · proj · relative_position_bias_table.
final class WindowAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "relative_position_bias_table") var biasTable: MLXArray
    let numHeads: Int
    let window: Int
    let scale: Float
    let rpi: RelativePositionIndex

    init(dim: Int, window ws: Int, heads: Int) {
        numHeads = heads
        window = ws
        scale = pow(Float(dim / heads), -0.5)
        _qkv.wrappedValue = Linear(dim, dim * 3)
        _proj.wrappedValue = Linear(dim, dim)
        _biasTable.wrappedValue = zeros([(2 * ws - 1) * (2 * ws - 1), heads])
        rpi = RelativePositionIndex(window: ws)
    }

    /// x: [B·nW, N, C]; mask: [nW, N, N] or nil.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (bw, n, c) = (x.dim(0), x.dim(1), x.dim(2))
        let hd = c / numHeads
        let t = qkv(x).reshaped([bw, n, 3, numHeads, hd]).transposed(2, 0, 3, 1, 4)
        let q = t[0] * scale, k = t[1], v = t[2]
        var attn = matmul(q, k.transposed(0, 1, 3, 2))                       // [B·nW, heads, N, N]
        let bias = biasTable[rpi.index].reshaped([n, n, numHeads]).transposed(2, 0, 1)
        attn = attn + bias.expandedDimensions(axis: 0).asType(attn.dtype)
        if let m = mask {
            let nw = m.dim(0)
            attn = attn.reshaped([bw / nw, nw, numHeads, n, n])
                + m.expandedDimensions(axis: 1).expandedDimensions(axis: 0).asType(attn.dtype)
            attn = attn.reshaped([bw, numHeads, n, n])
        }
        attn = softmax(attn, axis: -1, precise: true)
        let y = matmul(attn, v).transposed(0, 2, 1, 3).reshaped([bw, n, c])
        return proj(y)
    }
}

/// 1×1-conv MLP (swin.py `Mlp`). Keys: fc1 · fc2. Exact (erf) GELU, as torch `nn.GELU()`.
final class SwinMlp: Module {
    @ModuleInfo(key: "fc1") var fc1: Conv2d
    @ModuleInfo(key: "fc2") var fc2: Conv2d
    init(_ dim: Int, hidden: Int) {
        _fc1.wrappedValue = Conv2d(inputChannels: dim, outputChannels: hidden, kernelSize: 1)
        _fc2.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: dim, kernelSize: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// `SwinTransformerBlock` (swin.py) over an NHWC map. Keys: norm1 · attn · norm2 · mlp.
final class SwinBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: TorchGroupNorm
    @ModuleInfo(key: "attn") var attn: WindowAttention
    @ModuleInfo(key: "norm2") var norm2: TorchGroupNorm
    @ModuleInfo(key: "mlp") var mlp: SwinMlp
    let window: Int
    let shiftSize: Int

    init(dim: Int, heads: Int, window: Int, shift: Int, mlpRatio: Float, groups: Int) {
        self.window = window
        shiftSize = shift
        _norm1.wrappedValue = TorchGroupNorm(dim, groups: groups)
        _attn.wrappedValue = WindowAttention(dim: dim, window: window, heads: heads)
        _norm2.wrappedValue = TorchGroupNorm(dim, groups: groups)
        _mlp.wrappedValue = SwinMlp(dim, hidden: Int(Float(dim) * mlpRatio))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let ws = min(window, h, w)
        let shift = (ws == window && h > ws && w > ws) ? shiftSize : 0
        // The reference reflect-pads to a window multiple; the pipeline guarantees multiples (LQ padded to 64), so
        // that branch is unreachable here — fail loudly rather than silently diverge.
        precondition(h % ws == 0 && w % ws == 0, "Swin map \(h)×\(w) is not a multiple of window \(ws)")

        var y = norm1(x)
        if shift != 0 { y = roll(roll(y, shift: -shift, axis: 1), shift: -shift, axis: 2) }
        let mask = shift != 0 ? SwinMaskCache.mask(h: h, w: w, ws: ws, shift: shift) : nil
        // window partition: [B, H/ws, ws, W/ws, ws, C] → [B·nW, ws², C]
        let win = y.reshaped([b, h / ws, ws, w / ws, ws, c]).transposed(0, 1, 3, 2, 4, 5)
            .reshaped([-1, ws * ws, c])
        let out = attn(win, mask: mask)
        y = out.reshaped([b, h / ws, w / ws, ws, ws, c]).transposed(0, 1, 3, 2, 4, 5).reshaped([b, h, w, c])
        if shift != 0 { y = roll(roll(y, shift: shift, axis: 1), shift: shift, axis: 2) }
        let x1 = x + y
        return x1 + mlp(norm2(x1))
    }
}

/// 1×1 projection with a parameter-free norm slot: keys `proj` (norm = Identity in the release).
final class PatchProj: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    init(_ i: Int, _ o: Int) { _proj.wrappedValue = Conv2d(inputChannels: i, outputChannels: o, kernelSize: 1) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

/// `SwinStage` (swin.py), NON-residual (the released architecture). Keys: patch_embed · blocks · patch_unembed.
final class SwinStage: Module, NacreLayer {
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchProj
    @ModuleInfo(key: "blocks") var blocks: [SwinBlock]
    @ModuleInfo(key: "patch_unembed") var patchUnembed: PatchProj

    init(_ ch: Int, embedDim: Int, depth: Int, headChannels: Int, window: Int, mlpRatio: Float, groups: Int) {
        let heads = max(1, embedDim / headChannels)
        _patchEmbed.wrappedValue = PatchProj(ch, embedDim)
        _blocks.wrappedValue = (0 ..< depth).map {
            SwinBlock(dim: embedDim, heads: heads, window: window, shift: $0 % 2 == 0 ? 0 : window / 2,
                      mlpRatio: mlpRatio, groups: groups)
        }
        _patchUnembed.wrappedValue = PatchProj(embedDim, ch)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = patchEmbed(x)
        for b in blocks { h = b(h) }
        return patchUnembed(h)
    }

    func forward(_ x: MLXArray, emb: MLXArray) -> MLXArray { callAsFunction(x) }
}

// MARK: - SwinUNet

public struct SwinUNetConfig: Codable, Sendable, Equatable {
    public var imageSize = 64
    public var inChannels = 3
    public var modelChannels = 160
    public var outChannels = 3
    public var numResBlocks = [2, 2, 2, 2]
    public var attentionResolutions = [64, 32, 16, 8]
    public var channelMult = [1, 2, 2, 4]
    public var numHeadChannels = 32
    public var swinDepth = 2
    public var swinEmbedDim = 192
    public var windowSize = 8
    public var mlpRatio: Float = 4.0
    public var numGroups = 32
    public init() {}
}

/// `SwinUNet` (unet.py), `cond_lq` with `lq_size == image_size` (feature_extractor = identity).
/// Keys: time_embed.{0,2} · input_blocks.N.M · middle_block.{0,1,2} · output_blocks.N.M · out.{0,2}.
public final class SwinUNet: Module {
    @ModuleInfo(key: "time_embed") var timeEmbed: [Module]          // [Linear, SiLU, Linear]
    @ModuleInfo(key: "input_blocks") var inputBlocks: [[Module]]
    @ModuleInfo(key: "middle_block") var middleBlock: [Module]
    @ModuleInfo(key: "output_blocks") var outputBlocks: [[Module]]
    @ModuleInfo(key: "out") var out: [Module]                        // [GroupNorm32, SiLU, Conv3×3]
    public let config: SwinUNetConfig

    public init(_ cfg: SwinUNetConfig = .init()) {
        config = cfg
        let mc = cfg.modelChannels, ted = mc * 4, g = cfg.numGroups
        func swin(_ ch: Int) -> SwinStage {
            SwinStage(ch, embedDim: cfg.swinEmbedDim, depth: cfg.swinDepth, headChannels: cfg.numHeadChannels,
                      window: cfg.windowSize, mlpRatio: cfg.mlpRatio, groups: g)
        }
        _timeEmbed.wrappedValue = [Linear(mc, ted), SiLUBox(), Linear(ted, ted)]

        var ch = cfg.channelMult[0] * mc
        var input: [[Module]] = [[Conv2d(inputChannels: 2 * cfg.inChannels, outputChannels: ch, kernelSize: 3,
                                         padding: 1)]]
        var chans = [ch]
        var ds = cfg.imageSize
        for (level, mult) in cfg.channelMult.enumerated() {
            for i in 0 ..< cfg.numResBlocks[level] {
                var layers: [Module] = [ResBlock(ch, embChannels: ted, out: mult * mc, groups: g)]
                ch = mult * mc
                if cfg.attentionResolutions.contains(ds) && i == 0 { layers.append(swin(ch)) }
                input.append(layers)
                chans.append(ch)
            }
            if level != cfg.channelMult.count - 1 {
                input.append([UNetDownsample(ch, out: ch)])
                chans.append(ch)
                ds /= 2
            }
        }
        _inputBlocks.wrappedValue = input
        _middleBlock.wrappedValue = [ResBlock(ch, embChannels: ted, groups: g), swin(ch),
                                     ResBlock(ch, embChannels: ted, groups: g)]

        var output: [[Module]] = []
        for (level, mult) in cfg.channelMult.enumerated().reversed() {
            for i in 0 ... cfg.numResBlocks[level] {
                let ich = chans.removeLast()
                var layers: [Module] = [ResBlock(ch + ich, embChannels: ted, out: mc * mult, groups: g)]
                ch = mc * mult
                if cfg.attentionResolutions.contains(ds) && i == 0 { layers.append(swin(ch)) }
                if level > 0 && i == cfg.numResBlocks[level] {
                    layers.append(UNetUpsample(ch, out: ch))
                    ds *= 2
                }
                output.append(layers)
            }
        }
        _outputBlocks.wrappedValue = output
        _out.wrappedValue = [TorchGroupNorm(ch, groups: g), SiLUBox(),
                             Conv2d(inputChannels: cfg.channelMult[0] * mc, outputChannels: cfg.outChannels,
                                    kernelSize: 3, padding: 1)]
        super.init()
        train(false)   // C14: born in inference mode (nothing here is train/eval-dependent; dropout is 0)
    }

    static func run(_ seq: [Module], _ x: MLXArray, _ emb: MLXArray) -> MLXArray {
        var h = x
        for m in seq { h = (m as! NacreLayer).forward(h, emb: emb) }
        return h
    }

    /// The time embedding MLP for integer timesteps `[B]`.
    public func embed(_ t: MLXArray) -> MLXArray {
        let e = timestepEmbedding(t, dim: config.modelChannels)
        return (timeEmbed[2] as! Linear)(silu((timeEmbed[0] as! Linear)(e)))
    }

    /// x: [B, H, W, 3] scaled noisy latent; t: [B] timesteps; lq: [B, H, W, 3] pixel-space condition in [-1, 1].
    public func callAsFunction(_ x: MLXArray, t: MLXArray, lq: MLXArray) -> MLXArray {
        let emb = embed(t).asType(x.dtype)
        var h = concatenated([x, lq.asType(x.dtype)], axis: -1)
        var hs: [MLXArray] = []
        for blk in inputBlocks {
            h = Self.run(blk, h, emb)
            hs.append(h)
        }
        h = Self.run(middleBlock, h, emb)
        for blk in outputBlocks {
            h = concatenated([h, hs.removeLast()], axis: -1)
            h = Self.run(blk, h, emb)
        }
        h = silu((out[0] as! TorchGroupNorm)(h))
        return (out[2] as! Conv2d)(h)
    }
}

// MARK: - gate access (S1 taps; not part of the inference API)

extension SwinUNet {
    public var inputBlocksPublic: [[Module]] { inputBlocks }
    public static func runPublic(_ seq: [Module], _ x: MLXArray, _ emb: MLXArray) -> MLXArray { run(seq, x, emb) }
    /// One block of a Swin stage, from the stage's embedded input (S1: shifted vs unshifted isolated).
    public func swinBlockPublic(_ stage: Module, _ i: Int, _ x: MLXArray) -> MLXArray {
        (stage as! SwinStage).blocks[i](x)
    }
}
