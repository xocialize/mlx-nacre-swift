// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
import Foundation
import MLX
import MLXNacre
import MLXServeCore
import MLXToolKit
import NacreMLX

func physFootprint() -> (current: UInt64, peak: UInt64)? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return nil }
    return (info.phys_footprint, UInt64(info.ledger_phys_footprint_peak))
}

/// S4 [VAL]: the real MLXServeEngine — register → prepare → run on a PNG → write the result.
/// engine <in.png> <out.png> [--weights DIR | --store ROOT] [--fp32] [--scale N] [--raw]
func runEngine(_ a: [String]) async throws {
    func opt(_ k: String) -> String? { a.firstIndex(of: k).map { a[$0 + 1] } }
    let quant: Quant = a.contains("--fp32") ? .fp32 : .fp16
    let cfg = NacreConfiguration(quant: quant,
                                 weightsDirectory: opt("--weights").map { URL(fileURLWithPath: $0) })
    let engine = MLXServeEngine()
    await engine.useModelStore(ModelStore(root: opt("--store").map { URL(fileURLWithPath: $0) }))
    let t0 = Date()
    let id = try await engine.register(NacreUpscalePackage.registration, configuration: cfg)
    let needs = await engine.needsDownload(.imageUpscale)
    _ = try await engine.prepare(.imageUpscale, package: id)
    let tPrep = Date().timeIntervalSince(t0)
    var image = Image(format: .png, data: try Data(contentsOf: URL(fileURLWithPath: a[0])))
    if a.contains("--raw") {
        image = NacreUpscalePackage.encodeRawBGRA8(try NacreUpscalePackage.decodeRGB(image))
    }
    let t1 = Date()
    let resp = try await engine.run(ImageUpscaleRequest(image: image, scale: opt("--scale").map { Int($0)! }), package: id)
    let tRun = Date().timeIntervalSince(t1)
    guard let up = resp as? ImageUpscaleResponse else { throw NacreError.badInput("response") }
    var data = up.image.data
    if up.image.format == .rawBGRA8 {
        data = NacreUpscalePackage.encodePNG(try NacreUpscalePackage.decodeRGB(up.image))!
    }
    try data.write(to: URL(fileURLWithPath: a[1]))
    let peak = Double(Memory.peakMemory) / 1e9
    let phys = physFootprint()
    print(String(format: "OK engine %@ ×%d → %dx%d %@ | needsDownload %@ | prepare %.2f s run %.2f s | MLX peak %.2f GB | phys now %.2f GB, lifetime peak %.2f GB",
                 quant.rawValue, up.appliedScale, up.image.width ?? -1, up.image.height ?? -1, up.image.format.rawValue,
                 needs ? "yes" : "no", tPrep, tRun, peak, Double(phys?.current ?? 0) / 1e9, Double(phys?.peak ?? 0) / 1e9))
}

/// CAN live probe: start a run, cancel after `--after` s, report the error type and the latency to unwind.
func runCancel(_ a: [String]) async throws {
    func opt(_ k: String) -> String? { a.firstIndex(of: k).map { a[$0 + 1] } }
    let after = Double(opt("--after") ?? "2.0")!
    let cfg = NacreConfiguration(weightsDirectory: URL(fileURLWithPath: a[1]))
    let engine = MLXServeEngine()
    await engine.useModelStore(ModelStore(root: nil))
    let id = try await engine.register(NacreUpscalePackage.registration, configuration: cfg)
    _ = try await engine.prepare(.imageUpscale, package: id)
    let image = Image(format: .png, data: try Data(contentsOf: URL(fileURLWithPath: a[0])))
    let t0 = Date()
    _ = try await engine.run(ImageUpscaleRequest(image: image), package: id)
    let tFull = Date().timeIntervalSince(t0)
    let task = Task { try await engine.run(ImageUpscaleRequest(image: image), package: id) }
    try await Task.sleep(nanoseconds: UInt64(after * 1e9))
    let tc = Date()
    task.cancel()
    var outcome = "returned a result (NOT cancelled)"
    do { _ = try await task.value } catch is CancellationError { outcome = "CancellationError (unwrapped)" }
    catch { outcome = "other error: \(error)" }
    print(String(format: "cancel probe: full run %.2f s; cancelled at %.2f s → %@ after %.3f s", tFull, after, outcome,
                 Date().timeIntervalSince(tc)))
}
