// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
//  The C14 INF gate: every module in the constructed graphs reports `training == false` (set at the construction choke
//  points `SwinUNet.init` / `VQAutoencoder.init`), and the gate can FAIL on a graph flipped back to training mode.
import XCTest
import Foundation
import MLX
import MLXServeConformance
import MLXServeConformanceNN
import NacreMLX
@testable import MLXNacre

final class InferenceModeTests: XCTestCase {
    func testINFGatePassesOnConstructedGraphsAndCanFail() {
        Device.withDefaultDevice(Device(.cpu)) {
            let unet = SwinUNet(), vq = VQAutoencoder()
            let pass = InferenceModeConformance.check(
                flags: InferenceModeConformance.flags(of: ["unet": unet, "vq": vq]), posture: .moduleGraph)
            XCTAssertTrue(pass.passed, pass.summary)
            unet.train(true)
            let fail = InferenceModeConformance.check(
                flags: InferenceModeConformance.flags(of: ["unet": unet, "vq": vq]), posture: .moduleGraph)
            XCTAssertFalse(fail.passed, "the gate must be able to fail")
        }
    }

    func testPackageExposesTheGraphsByRole() async {
        let pkg = NacreUpscalePackage(configuration: NacreConfiguration())
        let graphs = await pkg.inferenceModeGraphs
        XCTAssertEqual(Set(graphs.keys), ["unet", "vq"])
        XCTAssertNil(graphs["unet"] ?? nil)
    }
}
