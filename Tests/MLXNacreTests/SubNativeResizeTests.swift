// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
import Testing
import Foundation
@testable import MLXNacre

/// The sub-native `scale` reduction, at the byte level (`downsample` wraps it): no MLX evaluation.
struct SubNativeResizeTests {

    /// AB-T-0232 (the AB-T-0225 fix, here): the ×4 → ×2 reduction stays registered across the whole frame. 2×2 white
    /// blocks on black near both corners and at the centre must stay centred on their output pixel. The CoreGraphics
    /// `.high` draw this replaced drifted ±0.26 px toward the edges (AB-R-0502).
    @Test func reductionStaysRegisteredAcrossTheFrame() throws {
        let n = 512, m = n / 2, spots = [8, 256, 496]
        var px = [UInt8](repeating: 0, count: n * n * 4)
        for y in 0..<n {
            for x in 0..<n {
                let white = spots.contains { ($0..<($0 + 2)).contains(x) && ($0..<($0 + 2)).contains(y) }
                for c in 0..<4 where c == 3 || white { px[(y * n + x) * 4 + c] = 255 }
            }
        }
        let out = try #require(NacreUpscalePackage.reduceRGBA8(px, width: n, height: n, toWidth: m, height: m))
        #expect(out.count == m * m * 4)
        for p in spots {
            let c = p / 2
            var sum = 0.0, sx = 0.0, sy = 0.0
            for y in (c - 4)...(c + 4) {
                for x in (c - 4)...(c + 4) {
                    let v = Double(out[(y * m + x) * 4]); sum += v; sx += v * Double(x); sy += v * Double(y)
                }
            }
            #expect(abs(sx / sum - Double(c)) < 0.02 && abs(sy / sum - Double(c)) < 0.02,
                    "block at \(p): centroid (\(sx / sum), \(sy / sum)), want (\(c), \(c))")
        }
    }
}
