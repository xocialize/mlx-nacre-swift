// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

/// PNG/JPEG → [1, H, W, 3] float32 in [0, 1] (sRGB values as stored; no colour management).
func readImage(_ url: URL) throws -> MLXArray {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw NSError(domain: "nacre", code: 1) }
    let (w, h) = (img.width, img.height)
    var buf = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!,
                        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    let rgba = MLXArray(buf, [1, h, w, 4]).asType(.float32) / 255
    return rgba[0..., 0..., 0..., 0 ..< 3]
}

/// [1, H, W, 3] in [0, 1] → 8-bit sRGB PNG.
func writePNG(_ x: MLXArray, _ url: URL) throws {
    let (h, w) = (x.dim(1), x.dim(2))
    let rgb = clip(x[0] * 255 + 0.5, min: 0, max: 255).asType(.uint8)
    let rgba = concatenated([rgb, full([h, w, 1], values: MLXArray(UInt8(255)))], axis: -1)
    var bytes = rgba.asArray(UInt8.self)
    let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!,
                        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
    guard CGImageDestinationFinalize(dest) else { throw NSError(domain: "nacre", code: 2) }
}
