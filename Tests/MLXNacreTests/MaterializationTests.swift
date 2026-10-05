// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
//  The offline MAT gate (contract 1.24): MAT-1..5 per lane.
import XCTest
import Foundation
import MLXToolKit
import MLXServeCore
import MLXServeConformance
@testable import MLXNacre

final class MaterializationTests: XCTestCase {
    private func satisfiedDirectory(for quant: Quant) throws -> URL {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("nacre-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        for f in NacreConfiguration.files(for: quant) {
            FileManager.default.createFile(atPath: tmp.appendingPathComponent(f).path, contents: Data([0]))
        }
        return tmp
    }

    func testFullMATGatePassesPerLane() throws {
        for quant in [Quant.fp16, .fp32] {
            let dir = try satisfiedDirectory(for: quant)
            defer { try? FileManager.default.removeItem(at: dir) }
            let report = MaterializationConformance.check(
                freshConfiguration: NacreConfiguration(quant: quant),
                satisfiedConfiguration: NacreConfiguration(quant: quant, weightsDirectory: dir))
            XCTAssertTrue(report.passed, "\(quant):\n\(report.summary)")
        }
    }

    /// The store's flat layout satisfies a lane only when ITS files are there.
    func testStoreFlatLayoutSatisfiesOnlyTheLanesOwnFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("nacre-store-\(UUID().uuidString)")
        let repoDir = root.appendingPathComponent(ModelStore.repoFolderName(for: NacreConfiguration.repo))
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let half = NacreConfiguration(quant: .fp16, modelsRootDirectory: root)
        let full = NacreConfiguration(quant: .fp32, modelsRootDirectory: root)
        XCTAssertEqual(half.missingWeightSources(storeRoot: root).count, 1)
        for f in NacreConfiguration.files(for: .fp16) {
            FileManager.default.createFile(atPath: repoDir.appendingPathComponent(f).path, contents: Data([0]))
        }
        XCTAssertTrue(half.missingWeightSources(storeRoot: root).isEmpty)
        XCTAssertEqual(full.missingWeightSources(storeRoot: root).count, 1, "the fp32 lane is still missing")
        XCTAssertEqual(half.prewarmPaths.count, 2)
    }

    func testEngineNeedsDownloadOnAFreshRegistration() async throws {
        let engine = MLXServeEngine()
        _ = try await engine.register(NacreUpscalePackage.registration, configuration: NacreConfiguration())
        let needs = await engine.needsDownload(.imageUpscale)
        XCTAssertTrue(needs)
    }
}
