// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
//  Core structure tests that never evaluate a kernel (safe in `swift test`): the S0 key-contract counts, the schedule
//  against the oracle's float64 values, the shifted-window mask, the bicubic phase weights. Numerics vs the PyTorch
//  oracle are the CLI gates (`nacre-smoke gate` / `gate-gpu`, PORTING-SPEC S1–S2).
import XCTest
import MLX
import MLXNN
@testable import NacreMLX

final class NacreCoreTests: XCTestCase {

    /// S0: the module trees carry exactly the converted checkpoints' key counts (540 denoiser, 204 autoencoder).
    func testKeyContractCounts() {
        XCTAssertEqual(SwinUNet().parameters().flattened().count, 540)
        XCTAssertEqual(VQAutoencoder().parameters().flattened().count, 204)
        let keys = Set(SwinUNet().parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("input_blocks.1.1.blocks.1.attn.relative_position_bias_table"))
        XCTAssertTrue(keys.contains("middle_block.1.patch_unembed.proj.weight"))
        XCTAssertFalse(keys.contains { $0.contains("relative_position_index") }, "the index is a rebuilt constant")
    }

    /// The schedule in Double equals the oracle's numpy float64 values (oracle/goldens/schedule.json).
    func testScheduleMatchesTheOracle() {
        let s = ResidualShiftSchedule()
        let sqrtEtas = [0.10000000000000002, 0.5200963722082476, 0.7613819906192041, 0.9899999999999999]
        let c1 = [0.0, 0.036968544403774305, 0.46661903261452575, 0.5914728452599346]
        let c2 = [1.0, 0.9630314555962257, 0.5333809673854742, 0.4085271547400654]
        let lv = [-3.2565450284151387, -3.2565450284151387, -0.5497072952602773, -0.054142708091363005]
        for i in 0 ..< 4 {
            XCTAssertEqual(s.sqrtEtas[i], sqrtEtas[i], accuracy: 1e-15)
            XCTAssertEqual(s.posteriorMeanCoef1[i], c1[i], accuracy: 1e-15)
            XCTAssertEqual(s.posteriorMeanCoef2[i], c2[i], accuracy: 1e-15)
            XCTAssertEqual(s.posteriorLogVarianceClipped[i], lv[i], accuracy: 1e-14)
        }
    }

    /// The two-axis shifted-window mask (the trained form): symmetric, zero on the diagonal, and seam-crossing pairs
    /// masked in a window that straddles BOTH seams (the bottom-right window) — the axis the reference's compat mask
    /// left open.
    func testShiftedWindowMaskCoversBothAxes() {
        let ws = 8, shift = 4, h = 16, w = 16
        let m = SwinMaskCache.mask(h: h, w: w, ws: ws, shift: shift).asArray(Float.self)
        let n = ws * ws, nW = (h / ws) * (w / ws)
        XCTAssertEqual(m.count, nW * n * n)
        for wi in 0 ..< nW {
            for i in 0 ..< n {
                XCTAssertEqual(m[(wi * n + i) * n + i], 0)
                for j in 0 ..< n { XCTAssertEqual(m[(wi * n + i) * n + j], m[(wi * n + j) * n + i]) }
            }
        }
        // bottom-right window: token (0,0) vs (0,7) crosses the vertical seam; (0,0) vs (7,0) the horizontal one
        let last = nW - 1
        XCTAssertEqual(m[(last * n + 0) * n + 7], -100, "horizontal-axis seam must be masked")
        XCTAssertEqual(m[(last * n + 0) * n + 7 * ws], -100, "vertical-axis seam must be masked")
        // top-left window holds a single region: nothing masked
        XCTAssertEqual(m[0 ..< n * n].filter { $0 != 0 }.count, 0)
    }

    /// Each bicubic phase's 1-D weights sum to 1 (torch's Keys kernel, a = −0.75) at scale 4.
    func testBicubicPhaseWeightsPartitionUnity() {
        for row in BicubicPhaseKernel.weights1D(scale: 4) {
            XCTAssertEqual(row.reduce(0, +), 1.0, accuracy: 1e-12)
            XCTAssertEqual(row.count, 5)
        }
    }

    /// C14 at the choke point: both graphs are born in inference mode.
    func testGraphsBornInInferenceMode() {
        XCTAssertFalse(SwinUNet().training)
        XCTAssertFalse(VQAutoencoder().training)
    }
}
