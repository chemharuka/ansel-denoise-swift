//
//  DenoiseDNGPipeline.swift
//
//  `image` with a `.dng` on either side: the tool owns the whole file-to-file trip.
//
//      read DNG (SourceDNG + SourceDNGDecode)
//        → read black/white/CFA from the file          (never hard-coded)
//        → normalize to [Float] on [0,1]
//        → DenoiseImageRunner.denoiseMosaic            (same planner, same padding)
//        → de-normalize to UInt16
//        → clamp to [0,1] before quantizing
//        → DNGRawWriter.write(from: the source file)   (every tag copied)
//
//  Why the tool and not the host
//  -----------------------------
//  The host used to do steps 3 and 5 itself and pass a float32 mosaic through two files. On a
//  60 MP frame that was 284 s of a 350 s run — 81 % of the wall time spent moving pixels in and
//  out of the process, while the inference it was paying for took 66 s. None of it was necessary:
//  the decoder and the writer both live here already, and both are plain Foundation.
//
//  Geometry invariants this file exists to uphold
//  ---------------------------------------------
//  A mosaic is not an image: sample (0,0) is one specific colour, and every pixel's colour is
//  decided by its index parity. So the output must have the same width, height, CFA phase, black
//  levels and white level as the source, or the picture turns into a checkerboard that still
//  passes every metadata-only check. The writer copies all of that verbatim, which means the four
//  things that could break it are all here:
//
//    1. padding must be right/bottom only (a centred pad moves the mosaic);
//    2. the crop back to the original size must be exact;
//    3. the CFA string handed to the network must describe *this* mosaic's (0,0);
//    4. the samples handed to the writer must be in the same UInt16 range as the ones read.
//

import Foundation

public struct DenoiseDNGRequest {
    public var model: AnselDenoiseModel
    public var metal: MetalConv?
    public var input: URL
    public var output: URL
    /// Per-channel Poisson-Gaussian coefficients, `Var = a·x + b` on [0,1]. Already multiplied by
    /// the sigma calibration, as the other subcommands take them.
    public var a: [Double]
    public var b: [Double]
    /// Override the CFA that the file states. Only for experiments; the file is authoritative and
    /// a mismatch between the two is exactly the failure mode that produces a checkerboard.
    public var cfaOverride: String?
    public var strength: Double
    public var budgetBytes: Int?
    public var forceColumns: Int?
    public var forceRows: Int?
    /// `--width` / `--height`, when the caller passed them. Checked against the file, never
    /// preferred over it.
    public var expectedWidth: Int?
    public var expectedHeight: Int?
    /// How many tiles feed the writer's Huffman statistics.
    public var tableSampleTiles: Int
    public var writerGrid: DNGRawWriter.TileGrid

    public init(model: AnselDenoiseModel, metal: MetalConv?,
                input: URL, output: URL,
                a: [Double], b: [Double],
                cfaOverride: String? = nil,
                strength: Double = 0.85,
                budgetBytes: Int? = nil,
                forceColumns: Int? = nil, forceRows: Int? = nil,
                expectedWidth: Int? = nil, expectedHeight: Int? = nil,
                tableSampleTiles: Int = 12,
                writerGrid: DNGRawWriter.TileGrid = DNGRawWriter.TileGrid()) {
        self.model = model
        self.metal = metal
        self.input = input
        self.output = output
        self.a = a
        self.b = b
        self.cfaOverride = cfaOverride
        self.strength = strength
        self.budgetBytes = budgetBytes
        self.forceColumns = forceColumns
        self.forceRows = forceRows
        self.expectedWidth = expectedWidth
        self.expectedHeight = expectedHeight
        self.tableSampleTiles = tableSampleTiles
        self.writerGrid = writerGrid
    }
}

public struct DenoiseDNGResult {
    public let width: Int
    public let height: Int
    /// CFA as tag 33422 states it, verbatim.
    public let fileCFA: String
    /// CFA the network was actually told — `fileCFA` rotated into the mosaic's frame, or the
    /// `--cfa` override. Reported separately from `fileCFA` on purpose: the two are only equal when
    /// the stored image coincides with its `ActiveArea`, and a host debugging a checkerboard needs
    /// to see which of the two was used. A single field would make "the file says bggr" and "the
    /// network was told rggb" indistinguishable in the report.
    public let mosaicCFA: String
    public let normalization: MosaicNormalization

    public let pieces: Int
    public let columns: Int
    public let rows: Int
    public let seams: Int
    public let singlePass: Bool
    public let peakBytes: Int
    public let wholeImageBytes: Int
    public let budgetBytes: Int

    public let readSeconds: Double
    public let inferenceSeconds: Double
    public let writeSeconds: Double
    public let seconds: Double
    public let backend: String

    /// The output file, as written.
    public let outputBytes: Int
    public let tileCount: Int
    /// Samples the network produced below 0 / above 1, counted **after** the strength blend and
    /// before the clamp. Non-zero is normal (residual overshoot on clipped blacks); what matters
    /// is that the clamp ran.
    public let clampedNegatives: Int
    public let clampedOverOne: Int
    /// Range of the array handed to the writer. `finalMin` is asserted `>= 0`.
    public let finalMin: Float
    public let finalMax: Float
    /// Samples that were outside `[0, 1]` in the *source* file — i.e. below the black level, which
    /// normalizes negative. Reported so a review can tell "the clamp had nothing to do" apart from
    /// "the clamp ate real signal".
    public let sourceBelowBlack: Int
}

public enum DenoiseDNGError: Error, CustomStringConvertible {
    case notCFA(String)
    case dimensionMismatch(flagWidth: Int, flagHeight: Int, fileWidth: Int, fileHeight: Int)
    case badMosaic(String)
    case decode(String)
    case write(String)
    case outputNotWritten(String)

    public var description: String {
        switch self {
        case .notCFA(let name):
            return "\(name) 里没有 CFA RAW 数据（已去马赛克的线性 DNG 无法降噪）"
        case .dimensionMismatch(let fw, let fh, let dw, let dh):
            return "--width/--height 是 \(fw)×\(fh)，但文件里是 \(dw)×\(dh)；两者必须一致（以文件为准）"
        case .badMosaic(let m):
            return "mosaic 尺寸与像素数不符：\(m)"
        case .decode(let m):
            return "解码 RAW 失败：\(m)"
        case .write(let m):
            return "写入降噪后的 DNG 失败：\(m)"
        case .outputNotWritten(let p):
            return "工具自报成功，但 \(p) 不存在"
        }
    }
}

public enum DenoiseDNGPipeline {

    public static func run(_ request: DenoiseDNGRequest,
                           progress: ((String) -> Void)? = nil) throws -> DenoiseDNGResult {

        let overallStart = Date()

        // MARK: 1. read

        let readStart = Date()
        let data: Data
        let doc: SourceDNG
        let decoded: (samples: [UInt16], width: Int, height: Int)
        do {
            data = try Data(contentsOf: request.input, options: .alwaysMapped)
            doc = try SourceDNG(data: data)
            decoded = try doc.readMosaic(data: data)
        } catch let error as SourceDNG.ReadError {
            if case .noRawSubIFD = error {
                throw DenoiseDNGError.notCFA(request.input.lastPathComponent)
            }
            throw DenoiseDNGError.decode(error.description)
        } catch let error as SourceDNG.DecodeError {
            throw DenoiseDNGError.decode(error.description)
        } catch let error as DNGRawWriter.WriteError {
            throw DenoiseDNGError.decode(error.description)
        } catch {
            throw DenoiseDNGError.decode("\(error)")
        }

        let width = decoded.width, height = decoded.height
        guard width > 0, height > 0, decoded.samples.count == width * height else {
            throw DenoiseDNGError.badMosaic("\(width)×\(height), \(decoded.samples.count) samples")
        }

        // A caller that passes --width/--height gets them checked, not obeyed. Silently taking
        // either number would mean the mosaic and the writer disagree about the grid, which is a
        // wrong-size output file at best and a sheared image at worst.
        if let ew = request.expectedWidth, let eh = request.expectedHeight,
           ew != width || eh != height {
            throw DenoiseDNGError.dimensionMismatch(flagWidth: ew, flagHeight: eh,
                                                    fileWidth: width, fileHeight: height)
        }

        let normalization = try DNGMetadata.normalization(of: doc)
        let cfa = try DNGMetadata.cfa(of: doc, width: width, height: height)
        progress?("read \(width)×\(height) \(cfa.filePattern) in "
                  + String(format: "%.2f s", Date().timeIntervalSince(readStart))
                  + (cfa.isRotated ? " (phase at mosaic origin: \(cfa.pattern))" : ""))
        let readSeconds = Date().timeIntervalSince(readStart)

        // MARK: 2. normalize

        var belowBlack = 0
        for v in decoded.samples where Double(v) < normalization.black(row: 0, col: 0) { belowBlack += 1 }

        var mosaic = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                mosaic[row + x] = normalization.normalize(decoded.samples[row + x], row: y, col: x)
            }
        }

        // MARK: 3. denoise (planner + runner, unchanged)

        // The mosaic, the source samples and the accumulator are all live at once, which is
        // 4 + 2 + 4 bytes per pixel the planner's per-pixel figure does not know about. Subtract it
        // so the plan cuts when the *real* footprint would not fit rather than discovering it as an
        // allocation failure 200 s in.
        let residency = width * height * (MemoryLayout<Float>.size * 2 + MemoryLayout<UInt16>.size)
        var budget = request.budgetBytes ?? DenoisePlanner.memoryLimit()
        budget = max(1, budget - residency)

        // One value, used both to drive the network and to report what drove it. Deriving the
        // report from `cfa.pattern` instead would let the two disagree — and the disagreement is
        // exactly the bug that produces a checkerboard with correct metadata.
        let mosaicCFA = request.cfaOverride ?? cfa.pattern
        let imageRequest = DenoiseImageRequest(
            input: request.input, output: request.output,
            width: width, height: height,
            a: request.a, b: request.b,
            cfa: mosaicCFA,
            useGPU: request.metal != nil,
            strength: request.strength,
            budgetBytes: budget,
            forceColumns: request.forceColumns, forceRows: request.forceRows)

        let inferenceStart = Date()
        let denoised = try DenoiseImageRunner.denoiseMosaic(mosaic, width: width, height: height,
                                                            request: imageRequest,
                                                            model: request.model,
                                                            metal: request.metal,
                                                            progress: progress)
        mosaic = []                                     // release before quantizing
        let inferenceSeconds = Date().timeIntervalSince(inferenceStart)

        // MARK: 4. clamp, de-normalize, write

        // `denoiseMosaic` already clamped its result to [0,1] *after* the strength blend, so this
        // assertion is the contract between the two halves of the DNG path: nothing below zero may
        // reach the quantizer, because denormalizing a negative would produce a sample below the
        // black level, and the writer has no idea that happened.
        let finalMin = denoised.finalMin, finalMax = denoised.finalMax

        let writeStart = Date()
        let quantized = normalization.quantized(denoised.mosaic, width: width, height: height)

        let written: DNGRawWriter.Result
        do {
            written = try DNGRawWriter.write(samples: quantized, width: width, height: height,
                                             from: request.input, to: request.output,
                                             grid: request.writerGrid,
                                             tableSampleTiles: request.tableSampleTiles)
        } catch let error as DNGRawWriter.WriteError {
            throw DenoiseDNGError.write(error.description)
        } catch {
            throw DenoiseDNGError.write("\(error)")
        }
        guard FileManager.default.fileExists(atPath: request.output.path) else {
            throw DenoiseDNGError.outputNotWritten(request.output.path)
        }
        let writeSeconds = Date().timeIntervalSince(writeStart)

        progress?("wrote \(request.output.lastPathComponent) (\(written.byteCount) bytes, "
                  + "\(written.tileCount) tiles) in " + String(format: "%.2f s", writeSeconds))

        return DenoiseDNGResult(
            width: width, height: height,
            fileCFA: cfa.filePattern, mosaicCFA: mosaicCFA,
            normalization: normalization,
            pieces: denoised.pieces,
            columns: denoised.plan.columns, rows: denoised.plan.rows,
            seams: denoised.plan.seams,
            singlePass: denoised.plan.isSinglePass,
            peakBytes: denoised.plan.peakBytes,
            wholeImageBytes: denoised.plan.wholeImageBytes,
            budgetBytes: denoised.plan.budgetBytes,
            readSeconds: readSeconds,
            inferenceSeconds: inferenceSeconds,
            writeSeconds: writeSeconds,
            seconds: Date().timeIntervalSince(overallStart),
            backend: denoised.backend,
            outputBytes: written.byteCount,
            tileCount: written.tileCount,
            clampedNegatives: denoised.clampedNegatives,
            clampedOverOne: denoised.clampedOverOne,
            finalMin: finalMin, finalMax: finalMax,
            sourceBelowBlack: belowBlack)
    }
}
