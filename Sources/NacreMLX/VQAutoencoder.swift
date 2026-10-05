// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
//
// Isomorphic port of the reference `nacre/models/vq.py`, itself adapted from CompVis latent-diffusion (MIT,
// Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich). Only the two seams Nacre uses are ported:
// `encode` (pre-quantization latent) and `decode` with `force_not_quantize` (continuous latents — the codebook is
// never consulted, so its weights are not part of the module tree). NHWC.
import Foundation
import MLX
import MLXNN

/// Swish as upstream writes it (`x * sigmoid(x)`).
@inline(__always) func swish(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

/// VQ `normalize`: GroupNorm(32, eps 1e-6). Statistics always in fp32: on the half lane the decoder's activations
/// reach magnitudes whose squares overflow fp16 (the first fp16 decode was all-black, 7.1 dB vs the fp32 lane).
func vqNorm(_ ch: Int) -> TorchGroupNorm { TorchGroupNorm(ch, groups: 32, eps: 1e-6, fp32: true) }

/// `ResnetBlock` (no timestep path). Keys: norm1 · conv1 · norm2 · conv2 · nin_shortcut (when channels change).
final class VQResnetBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norm1") var norm1: TorchGroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: TorchGroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "nin_shortcut") var ninShortcut: Conv2d?

    init(_ i: Int, _ o: Int) {
        _norm1.wrappedValue = vqNorm(i)
        _conv1.wrappedValue = Conv2d(inputChannels: i, outputChannels: o, kernelSize: 3, padding: 1)
        _norm2.wrappedValue = vqNorm(o)
        _conv2.wrappedValue = Conv2d(inputChannels: o, outputChannels: o, kernelSize: 3, padding: 1)
        _ninShortcut.wrappedValue = i == o ? nil : Conv2d(inputChannels: i, outputChannels: o, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(swish(norm1(x)))
        h = conv2(swish(norm2(h)))
        return (ninShortcut.map { $0(x) } ?? x) + h
    }
}

/// `AttnBlock`: single-head self-attention over the whole spatial grid, 1×1 q/k/v/proj_out.
/// Above `chunkAbove` tokens the queries are processed in chunks of `queryChunk` — exact (softmax is per query row),
/// and the only way a whole-image latent fits: a 256² grid is 65k tokens, a 4.3 G-element weight matrix.
final class VQAttnBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norm") var norm: TorchGroupNorm
    @ModuleInfo(key: "q") var q: Conv2d
    @ModuleInfo(key: "k") var k: Conv2d
    @ModuleInfo(key: "v") var v: Conv2d
    @ModuleInfo(key: "proj_out") var projOut: Conv2d
    nonisolated(unsafe) static var chunkAbove = 4096     // 64² latent (the training grid) stays one matmul
    nonisolated(unsafe) static var queryChunk = 1024

    init(_ ch: Int) {
        _norm.wrappedValue = vqNorm(ch)
        _q.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 1)
        _k.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 1)
        _v.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 1)
        _projOut.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h0 = norm(x)
        let (b, hh, ww, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let n = hh * ww
        // Scores and softmax in fp32 on the half lane: q·k sums 512 products and overflows fp16.
        let qq = q(h0).reshaped([b, n, c]).asType(.float32)
        let kT = k(h0).reshaped([b, n, c]).transposed(0, 2, 1).asType(.float32)   // [b, c, n]
        let vv = v(h0).reshaped([b, n, c])
        let s = Float(pow(Double(c), -0.5))
        let o: MLXArray
        if n <= Self.chunkAbove {
            o = matmul(softmax(matmul(qq, kT) * s, axis: -1, precise: true).asType(vv.dtype), vv)
        } else {
            var parts: [MLXArray] = []
            var start = 0
            while start < n {
                let end = min(start + Self.queryChunk, n)
                let w = softmax(matmul(qq[0..., start ..< end, 0...], kT) * s, axis: -1, precise: true)
                let part = matmul(w.asType(vv.dtype), vv)
                eval(part)   // bound the live set: never more than one chunk's [chunk × n] weights at once
                parts.append(part)
                start = end
            }
            o = concatenated(parts, axis: 1)
        }
        return x + projOut(o.reshaped([b, hh, ww, c]))
    }
}

/// `_Mid`: block_1 → attn_1 → block_2.
final class VQMid: Module, UnaryLayer {
    @ModuleInfo(key: "block_1") var block1: VQResnetBlock
    @ModuleInfo(key: "attn_1") var attn1: VQAttnBlock
    @ModuleInfo(key: "block_2") var block2: VQResnetBlock
    init(_ ch: Int) {
        _block1.wrappedValue = VQResnetBlock(ch, ch)
        _attn1.wrappedValue = VQAttnBlock(ch)
        _block2.wrappedValue = VQResnetBlock(ch, ch)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { block2(attn1(block1(x))) }
}

/// Strided 3×3 with upstream's asymmetric (right, bottom) zero pad of 1. Key `conv`.
final class VQDownsample: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ ch: Int) { _conv.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 3, stride: 2) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(padded(x, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))]))
    }
}

/// Nearest 2× then 3×3. Key `conv`.
final class VQUpsample: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ ch: Int) { _conv.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 3, padding: 1) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(nearestUp2(x)) }
}

/// One encoder level: `block` (+ `downsample`). vq-f4 has no `attn` entries (attn_resolutions is empty).
final class VQDownLevel: Module {
    @ModuleInfo(key: "block") var block: [VQResnetBlock]
    @ModuleInfo(key: "downsample") var downsample: VQDownsample?
    init(blocks: [VQResnetBlock], down: VQDownsample?) {
        _block.wrappedValue = blocks
        _downsample.wrappedValue = down
    }
}

final class VQUpLevel: Module {
    @ModuleInfo(key: "block") var block: [VQResnetBlock]
    @ModuleInfo(key: "upsample") var upsample: VQUpsample?
    init(blocks: [VQResnetBlock], up: VQUpsample?) {
        _block.wrappedValue = blocks
        _upsample.wrappedValue = up
    }
}

/// vq-f4: ch 128, ch_mult (1, 2, 4), 2 res blocks, z 3.
final class VQEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down") var down: [VQDownLevel]
    @ModuleInfo(key: "mid") var mid: VQMid
    @ModuleInfo(key: "norm_out") var normOut: TorchGroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(ch: Int = 128, chMult: [Int] = [1, 2, 4], numRes: Int = 2, zChannels: Int = 3) {
        _convIn.wrappedValue = Conv2d(inputChannels: 3, outputChannels: ch, kernelSize: 3, padding: 1)
        let inMult = [1] + chMult
        var levels: [VQDownLevel] = []
        var blockIn = ch
        for (i, m) in chMult.enumerated() {
            blockIn = ch * inMult[i]
            let out = ch * m
            var blocks: [VQResnetBlock] = []
            for _ in 0 ..< numRes {
                blocks.append(VQResnetBlock(blockIn, out))
                blockIn = out
            }
            levels.append(VQDownLevel(blocks: blocks, down: i == chMult.count - 1 ? nil : VQDownsample(blockIn)))
        }
        _down.wrappedValue = levels
        _mid.wrappedValue = VQMid(blockIn)
        _normOut.wrappedValue = vqNorm(blockIn)
        _convOut.wrappedValue = Conv2d(inputChannels: blockIn, outputChannels: zChannels, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for level in down {
            for b in level.block { h = b(h) }
            if let d = level.downsample { h = d(h) }
        }
        h = mid(h)
        return convOut(swish(normOut(h)))
    }
}

final class VQDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid") var mid: VQMid
    @ModuleInfo(key: "up") var up: [VQUpLevel]
    @ModuleInfo(key: "norm_out") var normOut: TorchGroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(ch: Int = 128, chMult: [Int] = [1, 2, 4], numRes: Int = 2, zChannels: Int = 3) {
        var blockIn = ch * chMult.last!
        _convIn.wrappedValue = Conv2d(inputChannels: zChannels, outputChannels: blockIn, kernelSize: 3, padding: 1)
        _mid.wrappedValue = VQMid(blockIn)
        var levels: [VQUpLevel] = []
        for i in (0 ..< chMult.count).reversed() {
            let out = ch * chMult[i]
            var blocks: [VQResnetBlock] = []
            for _ in 0 ... numRes {
                blocks.append(VQResnetBlock(blockIn, out))
                blockIn = out
            }
            levels.insert(VQUpLevel(blocks: blocks, up: i == 0 ? nil : VQUpsample(blockIn)), at: 0)
        }
        _up.wrappedValue = levels
        _normOut.wrappedValue = vqNorm(blockIn)
        _convOut.wrappedValue = Conv2d(inputChannels: blockIn, outputChannels: 3, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = mid(convIn(z))
        for level in up.reversed() {
            for b in level.block { h = b(h) }
            if let u = level.upsample { h = u(h) }
        }
        return convOut(swish(normOut(h)))
    }
}

/// The frozen VQ-f4 first stage. Keys: encoder · decoder · quant_conv · post_quant_conv.
public final class VQAutoencoder: Module {
    @ModuleInfo(key: "encoder") var encoder: VQEncoder
    @ModuleInfo(key: "decoder") var decoder: VQDecoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    public static let factor = 4

    public override init() {
        _encoder.wrappedValue = VQEncoder()
        _decoder.wrappedValue = VQDecoder()
        _quantConv.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 3, kernelSize: 1)
        _postQuantConv.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 3, kernelSize: 1)
        super.init()
        train(false)
    }

    /// [B, H, W, 3] in [-1, 1] → pre-quantization latent [B, H/4, W/4, 3].
    public func encode(_ x: MLXArray) -> MLXArray { quantConv(encoder(x)) }

    /// [B, h, w, 3] → [B, 4h, 4w, 3], nominally [-1, 1]. Continuous latent, no codebook snap.
    public func decode(_ z: MLXArray) -> MLXArray { decoder(postQuantConv(z)) }
}
