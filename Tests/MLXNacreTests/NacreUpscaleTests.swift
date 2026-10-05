// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
import Testing
import Foundation
import MLXToolKit
@testable import MLXNacre

/// Offline conformance — no Metal evaluation. Live upscaling is proven by `nacre-smoke engine` (the real engine path)
/// and the gates in PORTING-SPEC.md.
struct NacreUpscaleTests {

    @Test func manifestIsImageUpscaleAndPermissiveOnBothLayers() {
        let m = NacreUpscalePackage.manifest
        #expect(m.capabilities == [.imageUpscale])
        #expect(m.license.weightLicense == .apache2)      // C7 — our weights (VQ-f4 is CompVis MIT, NOTICE)
        #expect(m.license.portCodeLicense == .apache2)    // C8 — our code
        #expect(LicensePolicy.permissiveOnly.evaluate(m.license) == .admitted)
    }

    @Test func provenancePointsAtANamespaceWeControl() {
        #expect(NacreUpscalePackage.manifest.provenance.sourceRepo.hasPrefix("xocialize/"))
        #expect(NacreConfiguration.repo == "xocialize/nacre-v1-mlx")
    }

    @Test func manifestRequirements() {
        let r = NacreUpscalePackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
    }

    @Test func splitFootprintsDeclaredPerLane() {
        let fps = NacreUpscalePackage.manifest.requirements.footprints
        let fp16 = fps.first { $0.quant == .fp16 }, fp32 = fps.first { $0.quant == .fp32 }
        #expect(fp16 != nil && fp32 != nil)
        #expect((fp16?.residentBytes ?? 0) > 0 && (fp16?.peakActivationBytes ?? 0) > 0)
        #expect((fp16?.peakActivationBytes ?? 0) < (fp32?.peakActivationBytes ?? 0))
    }

    @Test func quantConfiguredAndBudgetAware() {
        let cfg: any PackageConfiguration = NacreConfiguration()
        #expect((cfg as? QuantConfigured)?.quant == .fp16)
        #expect(cfg is BudgetAware && cfg is ModelStorable && cfg is WeightSourcing && cfg is WeightPrewarming)
        #expect(NacreConfiguration(quant: .bf16).effectiveQuant == .fp16)
        #expect(NacreUpscalePackage(configuration: NacreConfiguration(quant: .fp32, availableBudgetBytes: 4_000_000_000)).plannedQuant == .fp16)
        #expect(NacreUpscalePackage(configuration: NacreConfiguration(quant: .fp32, availableBudgetBytes: 12_000_000_000)).plannedQuant == .fp32)
        #expect(NacreUpscalePackage(configuration: NacreConfiguration(quant: .fp16, availableBudgetBytes: 1)).plannedQuant == .fp16)
    }

    @Test func surfaceIsTheCanonicalUpscaleDescriptor() {
        let s = NacreUpscalePackage.manifest.surfaces.first
        #expect(s?.capability == .imageUpscale)
        #expect(s?.name == "nacre-upscale")
        #expect(s?.parameters.first?.kind == .image)
        #expect(s?.parameters.contains { $0.name == "scale" && !$0.required } == true)
    }

    @Test func registrationConstructs() throws {
        let reg = NacreUpscalePackage.registration
        #expect(reg.manifest.capabilities == [.imageUpscale])
        #expect(try reg.makePackage(NacreConfiguration()) is NacreUpscalePackage)
    }

    @Test func weightSourcesFollowTheLane() {
        let half = NacreConfiguration(quant: .fp16).weightSources
        #expect(half.count == 1 && half[0].role == "nacre-v1-fp16" && half[0].repo == NacreConfiguration.repo)
        #expect(half[0].matching == ["nacre_v1_mlx_fp16.safetensors", "vq_f4_mlx_fp16.safetensors", "config.json"])
        let full = NacreConfiguration(quant: .fp32).weightSources
        #expect(full[0].role == "nacre-v1-fp32")
        #expect(full[0].matching?.contains("nacre_v1_mlx_fp32.safetensors") == true)
    }

    @Test func configurationCodableRoundTripsPortableKnobsOnly() throws {
        let c = NacreConfiguration(quant: .fp32, tileSize: 192, tileOverlap: 16, seed: 7,
                                   weightsDirectory: URL(fileURLWithPath: "/x"),
                                   modelsRootDirectory: URL(fileURLWithPath: "/y"), availableBudgetBytes: 1)
        let back = try JSONDecoder().decode(NacreConfiguration.self, from: JSONEncoder().encode(c))
        #expect(back.quant == .fp32 && back.tileSize == 192 && back.tileOverlap == 16 && back.seed == 7)
        #expect(back.weightsDirectory == nil && back.modelsRootDirectory == nil && back.availableBudgetBytes == nil)
    }
}
