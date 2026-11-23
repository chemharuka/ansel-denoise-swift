//
//  NoiseProbe.swift
//  AnselDenoise
//
//  Measure the noise curve of a DNG without denoising it.
//
//  `image` measures the curve as a side effect of running; this exposes the same measurement on its
//  own, which is what a review needs to answer "did the model get the right sigma for this frame"
//  without paying for a minute of inference. It shares the decode and normalization code with the
//  DNG pipeline, so the numbers it prints are the numbers the denoiser conditions on.
//

import Foundation

public enum NoiseProbe {

    public struct Result: Sendable {
        public let width: Int
        public let height: Int
        /// CFA as tag 33422 states it.
        public let fileCFA: String
        /// CFA the mosaic's (0,0) is, i.e. the one the estimator was given.
        public let mosaicCFA: String
        public let model: NoiseEstimateModel
    }

    /// - Parameter dumpMosaic: write the normalized mosaic here (float32, row-major) plus a `.txt`
    ///   sidecar of `width height black… white cfa`. The comparison a review needs is in this
    ///   domain: two files decoded and normalized the same way, not two renders.
    public static func measure(input: URL,
                               cfaOverride: String? = nil,
                               checkerboard: Bool = false,
                               dumpMosaic: URL? = nil) throws -> Result {
        let data: Data
        let doc: SourceDNG
        let decoded: (samples: [UInt16], width: Int, height: Int)
        do {
            data = try Data(contentsOf: input, options: .alwaysMapped)
            doc = try SourceDNG(data: data)
            decoded = try doc.readMosaic(data: data)
        } catch let error as SourceDNG.ReadError {
            if case .noRawSubIFD = error {
                throw DenoiseDNGError.notCFA(input.lastPathComponent)
            }
            throw DenoiseDNGError.decode(error.description)
        } catch {
            throw DenoiseDNGError.decode("\(error)")
        }

        let width = decoded.width, height = decoded.height
        guard width > 0, height > 0, decoded.samples.count == width * height else {
            throw DenoiseDNGError.badMosaic("\(width)×\(height), \(decoded.samples.count) samples")
        }

        let normalization = try DNGMetadata.normalization(of: doc)
        let cfa = try DNGMetadata.cfa(of: doc, width: width, height: height)
        var mosaic = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                mosaic[row + x] = normalization.normalize(decoded.samples[row + x], row: y, col: x)
            }
        }

        let mosaicCFA = cfaOverride ?? cfa.pattern
        if let dump = dumpMosaic {
            try mosaic.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: dump)
            let sidecar = "\(width) \(height) "
                + normalization.blackByPhase.map { String(format: "%.6f", $0) }.joined(separator: " ")
                + " " + String(format: "%.6f", normalization.white) + " " + cfa.filePattern + "\n"
            try sidecar.write(to: dump.appendingPathExtension("txt"), atomically: true, encoding: .utf8)
        }
        let model = try AnselNoiseEstimator.measure(mosaic, width: width, height: height,
                                                    cfa: mosaicCFA, checkerboard: checkerboard)
        return Result(width: width, height: height, fileCFA: cfa.filePattern,
                      mosaicCFA: mosaicCFA, model: model)
    }
}
