// Copyright 2026 Xocialize. Licensed under the Apache License, Version 2.0.
//  The SwinUNet's input conv (latent + LQ stacked, 6 → 160, 3×3, stride 1) is the shape mlx 0.32.2 serves with
//  `pad_in_channels_conv_2D_gpu`, which frees its zero fill value while the GPU still reads it (AB-T-0214). It is a
//  `RoutedConv2d`, so ExactConv pads the channels in the graph; on the GPU it must still match a plain conv on the CPU.
//  Weights-free (seeded); evaluates one conv on each device under XCTest.
import XCTest
import MLX
import MLXExactConv
import MLXNN
@testable import NacreMLX

final class PadHelperTests: XCTestCase {
    func testInputConvIsRoutedOffThePadHelper() {
        let unet = SwinUNet()
        guard let conv = unet.inputBlocksPublic[0][0] as? RoutedConv2d else {
            return XCTFail("the input conv is not routed")
        }
        MLXRandom.seed(9)
        conv.update(parameters: conv.parameters().mapValues { MLXRandom.normal($0.shape) * 0.05 })
        let x = MLXRandom.normal([1, 64, 48, 6])
        XCTAssertTrue(
            ExactConv.takesPadHelper2D(input: x, weight: conv.weight, padding: (1, 1)),
            "a plain Conv2d at this shape would take mlx's pad helper")

        let gpu = Device.withDefaultDevice(Device(.gpu)) { conv(x) }
        let cpu = Device.withDefaultDevice(Device(.cpu)) { conv2d(x, conv.weight, padding: 1) + conv.bias! }
        eval(gpu, cpu)
        XCTAssertEqual(gpu.shape, cpu.shape)
        XCTAssertLessThan(abs(gpu - cpu).max().item(Float.self), 1e-4)
    }
}
