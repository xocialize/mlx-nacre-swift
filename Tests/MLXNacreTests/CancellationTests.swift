// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
//  The offline CAN gate: CAN-1/CAN-2 pre-cancelled run() propagation + classification, CAN-3 the cadence of record.
//  The live mid-run probe is `nacre-smoke cancel`.
import XCTest
import Foundation
import MLXToolKit
import MLXServeConformance
@testable import MLXNacre

final class CancellationTests: XCTestCase {
    func testCANGatePreCancelledRun() async {
        let package = NacreUpscalePackage(configuration: NacreConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package, request: ImageUpscaleRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: NacreUpscalePackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: NacreUpscalePackage.manifest,
            posture: .cadence([
                // A checkpoint after EVERY diffusion step of every tile (4 per tile) — the step is the unit, the tile
                // the stage; RunProgress at the same seam (phase denoise, step over tiles·4, stage = tile).
                .init(phase: .denoise, unit: .step, reportsRunProgress: true),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
