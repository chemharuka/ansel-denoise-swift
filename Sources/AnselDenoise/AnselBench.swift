//
//  AnselBench.swift
//  AnselDenoise
//
//  Block-size and backend sweep. GPLv3 component — see LICENSE-SCOPE.md.
//
//  Answers, on one machine, in one process:
//    - how long a tile takes on CPU vs GPU at 64/128/256/512²
//    - what each model variant costs (quarter/half/large × single/multi)
//
//  Deliberately runs the same weights on both backends so the ratio is meaningful;
//  a backend comparison across different weights is not a comparison.
//

import Foundation
import Metal

public struct BenchRow {
    public let backend: String
    public let tile: Int
    public let ms: Double
    public let ok: Bool
}

public enum AnselBench {
    /// Deterministic pseudo-random mosaic so runs are comparable.
    public static func syntheticMosaic(tile: Int, seed: UInt64 = 42) -> [Float] {
        var s = seed
        var out = [Float](repeating: 0, count: tile * tile)
        for i in 0..<out.count {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let u = Float(s >> 33) / Float(1 << 31)     // [0,1)
            out[i] = u * 0.45
        }
        return out
    }

    /// Time one denoise call (repeats for a median-ish estimate).
    public static func timeCPU(model: AnselDenoiseModel, tile: Int,
                               repeats: Int = 2) throws -> Double {
        let fwd = AnselDenoiseForward(model: model)
        let mosaic = syntheticMosaic(tile: tile)
        let color = AnselDenoiseForward.colorIndexMap(pattern: "rggb", width: tile, height: tile)
        let a: [Double] = [0.0004, 0.0003, 0.0004]
        let b: [Double] = [3e-6, 1e-6, 1e-6]
        let planes = AnselDenoiseForward.makeInputPlanes(
            mosaic: mosaic, colorIndex: color, width: tile, height: tile, a: a, b: b)

        _ = try fwd.denoise(planes)                       // warm
        var best = Double.infinity
        for _ in 0..<repeats {
            let t0 = Date()
            _ = try fwd.denoise(planes)
            best = min(best, Date().timeIntervalSince(t0))
        }
        return best * 1000
    }

    /// Time one denoise call on the GPU.
    public static func timeGPU(model: AnselDenoiseModel, engine: MetalConv,
                               tile: Int, repeats: Int = 3) throws -> Double {
        let gpu = AnselDenoiseMetal(model: model, metal: engine)
        let mosaic = syntheticMosaic(tile: tile)
        let a: [Double] = [0.0004, 0.0003, 0.0004]
        let b: [Double] = [3e-6, 1e-6, 1e-6]

        _ = try gpu.denoise(mosaic: mosaic, width: tile, height: tile,
                            cfa: "rggb", a: a, b: b)      // warm + shader compile
        var best = Double.infinity
        for _ in 0..<repeats {
            let t0 = Date()
            _ = try gpu.denoise(mosaic: mosaic, width: tile, height: tile,
                                cfa: "rggb", a: a, b: b)
            best = min(best, Date().timeIntervalSince(t0))
        }
        return best * 1000
    }
}
