// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
//
// nacre-smoke — the CLI gate lane (PORTING-SPEC.md). Metal-context gates live here, not in XCTest.
//   keys            S0  module key sets == converted weight files (0 missing / 0 unused), both networks
//   gate [case]     S1/S2 per-stage + e2e parity vs the PyTorch goldens, CPU stream, fp32 (cases s64, s128)
//   run <in> <out>  S2b one real 4× upscale on the GPU (PNG in → PNG out), timed
import Foundation
import MLX
import MLXNN
import NacreMLX

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent()
let goldens = root.appendingPathComponent("oracle/goldens")
let weightsDir = root.appendingPathComponent("oracle/weights")
let unetURL = weightsDir.appendingPathComponent("nacre_v1_mlx_fp32.safetensors")
let vqURL = weightsDir.appendingPathComponent("vq_f4_mlx_fp32.safetensors")

// MARK: - .npy (float32, little-endian, C order)

func loadNpy(_ url: URL) throws -> MLXArray {
    let d = try Data(contentsOf: url)
    let hlen = Int(d[8]) | (Int(d[9]) << 8)
    let header = String(decoding: d[10 ..< 10 + hlen], as: UTF8.self)
    guard header.contains("'descr': '<f4'"), !header.contains("'fortran_order': True") else {
        throw NacreError.badInput("unsupported npy header \(header)")
    }
    let shapeStr = header.components(separatedBy: "'shape': (")[1].components(separatedBy: ")")[0]
    let shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    let body = d[(10 + hlen)...]
    let floats = body.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    return MLXArray(floats, shape)
}

func golden(_ c: String, _ n: String) -> MLXArray {
    do { return try loadNpy(goldens.appendingPathComponent(c).appendingPathComponent("\(n).npy")) }
    catch { fatalError("golden \(c)/\(n): \(error)") }
}

struct Cmp { let maxAbs: Float; let rel: Float; let cos: Float }
func compare(_ a: MLXArray, _ b: MLXArray) -> Cmp {
    let x = a.asType(.float32), y = b.asType(.float32)
    precondition(x.shape == y.shape, "shape \(x.shape) vs \(y.shape)")
    let d = abs(x - y)
    let maxAbs = d.max().item(Float.self)
    let rel = maxAbs / max(abs(y).max().item(Float.self), 1e-12)
    let cos = (sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)) + 1e-30)).item(Float.self)
    return Cmp(maxAbs: maxAbs, rel: rel, cos: cos)
}

var failures = 0
func check(_ name: String, _ a: MLXArray, _ b: MLXArray, relTol: Float) {
    let c = compare(a, b)
    let ok = c.rel <= relTol
    if !ok { failures += 1 }
    print(String(format: "  %@ %-22@ max|Δ| %.3e  rel %.3e  cos %.9f  (tol %.0e)",
                 ok ? "PASS" : "FAIL", name, c.maxAbs, c.rel, c.cos, relTol))
}

// MARK: - modes

func keysGate() throws {
    for (name, module, url) in [("unet", SwinUNet() as Module, unetURL), ("vq", VQAutoencoder() as Module, vqURL)] {
        let expected = Set(module.parameters().flattened().map(\.0))
        let got = Set(try MLX.loadArrays(url: url).keys)
        let missing = expected.subtracting(got), extra = got.subtracting(expected)
        let ok = missing.isEmpty && extra.isEmpty
        if !ok { failures += 1 }
        print("  \(ok ? "PASS" : "FAIL") \(name): module \(expected.count) keys, file \(got.count) — missing \(missing.count) \(missing.sorted().prefix(4)), extra \(extra.count) \(extra.sorted().prefix(4))")
        let n = module.parameters().flattened().reduce(0) { $0 + $1.1.size }
        print("       \(name) parameters: \(n)")
    }
}

func parityGate(_ c: String, gpu: Bool = false) throws {
    print("== S1/S2 parity, case \(c) (\(gpu ? "GPU" : "CPU") stream, fp32\(gpu ? ", MLX_ENABLE_TF32=\(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "default")" : ""))")
    try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
        let nacre = Nacre()
        try nacre.loadWeights(unet: unetURL, vq: vqURL)
        let lq = golden(c, "lq01")
        // bicubic ×4 (torch) and the encode seam
        let up = clip(bicubicUpsample(lq, scale: 4), min: 0, max: 1)
        check("bicubic_up", up, golden(c, "bicubic_up"), relTol: 1e-5)
        let y0 = nacre.vq.encode(golden(c, "bicubic_up") * 2 - 1)
        check("vq.encode → y0", y0, golden(c, "y0"), relTol: 2e-4)   // torch fp32 vs its own fp64: 3.8e-5 rel; see PORTING-SPEC S1
        // time embedding + MLP
        let t3 = MLXArray([Int32(3)])
        check("timestep_embedding", timestepEmbedding(t3, dim: 160), golden(c, "temb"), relTol: 1e-6)
        let emb = nacre.unet.embed(t3)
        check("time_embed", emb, golden(c, "emb"), relTol: 1e-5)
        // the first input path, stage by stage, each from the ORACLE's input (isolates the op)
        let u = nacre.unet
        let s = nacre.schedule
        let yg = golden(c, "y0")
        let xT = yg + Float(s.kappa * s.sqrtEtas[3]) * golden(c, "noise0")
        check("prior x_T", xT, golden(c, "x_T"), relTol: 1e-6)
        let cond = golden(c, "lq01") * 2 - 1
        let xin = golden(c, "x_in")
        let h0 = SwinUNet.runPublic(u.inputBlocksPublic[0], concatenated([xin, cond], axis: -1), emb)
        check("input_blocks[0]", h0, golden(c, "in0"), relTol: 1e-5)
        let rb = SwinUNet.runPublic([u.inputBlocksPublic[1][0]], golden(c, "in0"), emb)
        check("input_blocks[1] res", rb, golden(c, "in1_res"), relTol: 1e-5)
        let stage = u.inputBlocksPublic[1][1]
        check("input_blocks[1] swin", SwinUNet.runPublic([stage], golden(c, "in1_res"), emb),
              golden(c, "in1_swin"), relTol: 1e-5)
        check("swin block0 (unshifted)", u.swinBlockPublic(stage, 0, golden(c, "swin_b0_in")),
              golden(c, "swin_b0_out"), relTol: 1e-5)
        check("swin block1 (shifted)", u.swinBlockPublic(stage, 1, golden(c, "swin_b0_out")),
              golden(c, "swin_b1_out"), relTol: 1e-5)
        check("unet(x_in, t=3)", u(xin, t: t3, lq: cond), golden(c, "unet_t3"), relTol: 1e-4)
        // S2 — the full chain with injected noise, per step, then decode
        let noise = (0 ..< 4).map { golden(c, "noise\($0)") }
        let z = nacre.sample(y0: yg, cond: cond, noise: noise, taps: { step, x in
            check("chain t=\(step)", x, golden(c, "chain_t\(step)"), relTol: 1e-3)
        })
        check("z0", z, golden(c, "z0"), relTol: 1e-3)
        let img = clip((nacre.vq.decode(golden(c, "z0")) + 1) / 2, min: 0, max: 1)
        check("vq.decode(z0) → out", img, golden(c, "out01"), relTol: 1e-4)
        // e2e through the public entry point (bicubic + encode + chain + decode in one go)
        let e2e = nacre.upscale(lq, injectedNoise: noise)
        check("upscale() e2e", e2e, golden(c, "out01"), relTol: 2e-3)
        let mse = mean(square(e2e - golden(c, "out01"))).item(Float.self)
        print(String(format: "       e2e PSNR vs torch: %.1f dB", 10 * log10(1 / max(mse, 1e-20))))
    }
}

let args = CommandLine.arguments.dropFirst()
switch args.first ?? "keys" {
case "keys":
    print("== S0 key contract"); try keysGate()
case "encprobe":
    // AB-L-0180 probe: is the encode residual the CPU stream's naive reductions? Same weights, GPU stream.
    let nacre = Nacre()
    try nacre.loadWeights(unet: unetURL, vq: vqURL)
    for c in ["s64", "s128"] {
        let y = nacre.vq.encode(golden(c, "bicubic_up") * 2 - 1)
        check("GPU vq.encode \(c)", y, golden(c, "y0"), relTol: 1e-4)
    }
case "run":
    // S2b: one real 4× upscale on the GPU. nacre-smoke run <in.png> <out.png> [--fp16] [--seed N]
    let a = Array(args.dropFirst())
    guard a.count >= 2 else { print("run <in> <out> [--fp16]"); exit(2) }
    let half = a.contains("--fp16") || a.contains("--fp16-unet")
    let halfVQ = a.contains("--fp16") || a.contains("--fp16-vq")
    let nacre = Nacre()
    let t0 = Date()
    try nacre.loadWeights(unet: unetURL, vq: vqURL, dtype: half ? .float16 : nil, vqDtype: halfVQ ? .float16 : .float32)
    let tLoad = Date().timeIntervalSince(t0)
    let lq = try readImage(URL(fileURLWithPath: a[0]))
    Memory.peakMemory = 0
    let t1 = Date()
    let out = nacre.upscale(lq, onStep: { d, n in print("  step \(d)/\(n)  \(String(format: "%.2f", Date().timeIntervalSince(t1))) s") })
    eval(out)
    let tRun = Date().timeIntervalSince(t1)
    try writePNG(out, URL(fileURLWithPath: a[1]))
    print(String(format: "  unet %@ / vq %@  %d×%d → %d×%d  load %.2f s  run %.2f s  MLX peak %.2f GB", half ? "fp16" : "fp32", halfVQ ? "fp16" : "fp32",
                 lq.dim(2), lq.dim(1), out.dim(2), out.dim(1), tLoad, tRun, Double(Memory.peakMemory) / 1e9))
case "mem":
    // S3 memory curve: nacre-smoke mem <in.png> [--fp16] — whole-pipeline peak at square LQ crops 64…256
    let a = Array(args.dropFirst())
    let half = a.contains("--fp16")
    let nacre = Nacre()
    try nacre.loadWeights(unet: unetURL, vq: vqURL, dtype: half ? .float16 : nil, vqDtype: half ? .float16 : .float32)
    let img = try readImage(URL(fileURLWithPath: a[0]))
    for side in [64, 128, 192, 256] where side <= min(img.dim(1), img.dim(2)) {
        let lq = img[0..., 0 ..< side, 0 ..< side, 0...]
        _ = nacre.upscale(lq[0..., 0 ..< 64, 0 ..< 64, 0...])   // warm the kernels at the smallest grid
        Memory.clearCache(); Memory.peakMemory = 0
        let t = Date(); let out = nacre.upscale(lq); eval(out)
        print(String(format: "  %@ LQ %3d² → %4d²  %.2f s  MLX peak %.2f GB", half ? "fp16" : "fp32", side, 4 * side,
                     Date().timeIntervalSince(t), Double(Memory.peakMemory) / 1e9))
    }
case "tiled":
    // S3 tiling: nacre-smoke tiled <in.png> <out.png> [--fp16] [--tile N] [--overlap N]
    let a = Array(args.dropFirst())
    let half = a.contains("--fp16")
    func opt(_ k: String, _ d: Int) -> Int { a.firstIndex(of: k).flatMap { Int(a[$0 + 1]) } ?? d }
    let nacre = Nacre()
    try nacre.loadWeights(unet: unetURL, vq: vqURL, dtype: half ? .float16 : nil, vqDtype: half ? .float16 : .float32)
    let lq = try readImage(URL(fileURLWithPath: a[0]))
    Memory.peakMemory = 0
    let t = Date()
    let out = nacre.upscaleTiled(lq, tile: opt("--tile", 128), overlap: opt("--overlap", 32))  // defaults = Nacre.default*
    eval(out)
    try writePNG(out, URL(fileURLWithPath: a[1]))
    print(String(format: "  tiled %@ tile %d/%d  %d×%d → %d×%d  %.2f s  MLX peak %.2f GB", half ? "fp16" : "fp32",
                 opt("--tile", 128), opt("--overlap", 32), lq.dim(2), lq.dim(1), out.dim(2), out.dim(1),
                 Date().timeIntervalSince(t), Double(Memory.peakMemory) / 1e9))
case "engine":
    try await runEngine(Array(args.dropFirst()))
case "cancel":
    try await runCancel(Array(args.dropFirst()))
case "gate", "gate-gpu":
    let cases = args.dropFirst().filter { !$0.hasPrefix("-") }
    for c in (cases.isEmpty ? ["s64", "s128"] : Array(cases)) { try parityGate(c, gpu: args.first == "gate-gpu") }
default:
    print("usage: nacre-smoke keys | gate [s64 s128]")
}
print(failures == 0 ? "ALL PASS" : "FAILURES: \(failures)")
exit(failures == 0 ? 0 : 1)
