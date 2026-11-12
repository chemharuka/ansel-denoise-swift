//
//  AnselDenoiseClient.swift
//
//  Client for the `ansel-denoise-swift` executable (GPLv3). Part of PhotonsExtensions: it
//  spawns that program as a child process — nothing of it is linked, so this file stays MIT
//  and the extension module keeps its "no core dependency" rule (see README.md here).
//
//  Why a child process: the upstream network definition comes from `aurelienpierreeng/ansel-denoise`
//  (GPL-3.0). Compiling it into a separate executable and talking over a pipe keeps the two a mere
//  aggregate rather than a single work — the same arrangement already used for `hdrmerge-swift`
//  (also GPLv3). See Vendor/ansel-denoise-swift/README.md.
//
//  What this produces: a **denoised RGGB mosaic**, not an RGB image. The caller pairs it with
//  `DNGRawWriter` to get a legal lossless-compressed DNG whose black point and metadata are the
//  source file's own.
//

import Foundation

// MARK: - Public model

/// Which trained network to run.
///
/// All variants share one weight set per camera (the CFA layout and noise level are inputs, not
/// baked-in parameters), so choosing a variant is purely a quality/speed trade-off:
///
/// | variant | params | 256² tile | σ reduction |
/// |---|---|---|---|
/// | `quarter` | 0.48 M | ~50 ms | 88 % |
/// | `half` | 1.90 M | ~216 ms | 92 % |
/// | `large` | 7.59 M | ~882 ms | 93 % |
///
/// `quarter` is the default: it is 17× faster than `large` for 5 points less denoising, and it
/// degrades far more gracefully when the camera's noise profile is off (`large` reads as *worse*
/// than `quarter` under a mis-scaled sigma, which is why sigma calibration must be right before
/// comparing variants at all).
public enum AnselDenoiseVariant: String, Sendable, CaseIterable {
    case quarter, half, large

    /// File name as published by the upstream manifest.
    public var modelFileName: String { "denoise-\(rawValue)-single-v1.anselnn" }

    public var approximateParameters: Int {
        switch self {
        case .quarter: return 475_265
        case .half:    return 1_898_241
        case .large:   return 7_587_329
        }
    }
}

public struct AnselDenoiseOptions: Sendable, Equatable {
    public var variant: AnselDenoiseVariant = .quarter
    /// Tile edge in mosaic pixels. Must be a multiple of 16 (the network halves resolution
    /// `depth = 4` times, so anything smaller or misaligned collapses the bottleneck to 1×1).
    public var tileSize: Int = 256
    /// Context margin between tiles, in pixels. Ansel's measured receptive field decays to ~1e-6
    /// of peak at r = 32 and to zero by 96; 48 is their shipped value. A smaller overlap leaves
    /// faint seams that a single-tile test will never reveal.
    public var overlap: Int = 48
    /// CFA at the mosaic's top-left, row-major, e.g. "rggb".
    public var cfa: String = "rggb"
    /// Per-channel Poisson-Gaussian coefficients such that `Var = a·x + b` on values normalized
    /// to [0,1] (black subtracted, white-normalized). These are what a camera noise profile gives.
    public var noiseA: [Double] = []
    public var noiseB: [Double] = []
    /// Extra per-channel sigma calibration. Defaults to Ansel's shipped values
    /// (R 2.82, G 3.94, B 2.96 = an exact ×2 from the lifting-Haar profile fit, times the
    /// per-channel demosaic-attenuation medians). **Leaving this out under-denoises badly** —
    /// the network then believes the noise is 3–4× smaller than it is.
    public var sigmaScale: [Double] = [2.82, 3.94, 2.96]
    /// Run on the GPU (Metal). Roughly 30× faster than the CPU path at every tile size.
    public var useGPU: Bool = true
    /// How much of the denoised result to use, 0…1. The blend is
    /// `out = original + strength·(denoised − original)`, applied in the same normalized domain as
    /// the input, so it cannot disturb black/white levels.
    ///
    /// 0.85 is what Ansel ships: enough to visibly clean up, little enough that texture survives.
    /// Note this is **not** the same knob as `sigmaScale`. Sigma tells the network how much noise
    /// to expect (it re-runs inference and decides what to remove); strength only decides how much
    /// of the answer to keep. Lowering strength puts original noise back along with the detail.
    public var strength: Double = 0.85

    public init() {}
}

public struct AnselDenoiseReport: Sendable, Equatable {
    public let output: URL
    public let tiles: Int
    public let tileSize: Int
    public let milliseconds: Double
    public let backend: String
    public let cfa: String
}

public struct AnselDenoiseFailure: Error, CustomStringConvertible, Sendable {
    public let code: String
    public let message: String
    public let exitCode: Int32?
    public var description: String { "\(code): \(message)" }
}

// MARK: - Client

/// Locates and drives the `ansel-denoise-swift` executable.
///
/// Resolution order (mirrors `HDRMergeClient`):
///   1. explicit `executableURL`
///   2. `$ANSEL_DENOISE_SWIFT` (dev override)
///   3. `Bundle.main/Contents/Resources/bin/ansel-denoise-swift` (shipped inside the app)
///   4. `/opt/homebrew/bin/ansel-denoise-swift`, `/usr/local/bin/ansel-denoise-swift`
public struct AnselDenoiseClient: Sendable {
    public let executableURL: URL

    public init(executableURL: URL) {
        self.executableURL = executableURL
    }

    public init(resolvingExplicit explicit: URL? = nil,
                environment: [String: String] = ProcessInfo.processInfo.environment,
                bundle: Bundle = .main) throws {
        if let explicit {
            self.executableURL = explicit
            return
        }
        if let env = environment["ANSEL_DENOISE_SWIFT"],
           FileManager.default.isExecutableFile(atPath: env) {
            self.executableURL = URL(fileURLWithPath: env)
            return
        }
        var candidates: [URL] = []
        if let res = bundle.resourceURL {
            candidates.append(res.appendingPathComponent("bin/ansel-denoise-swift"))
        }
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/ansel-denoise-swift"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/ansel-denoise-swift"))
        for c in candidates where FileManager.default.isExecutableFile(atPath: c.path) {
            self.executableURL = c
            return
        }
        throw AnselDenoiseFailure(
            code: "tool_missing",
            message: "ansel-denoise-swift not found (looked in app resources and /usr/local/bin)",
            exitCode: nil)
    }

    /// `capabilities` — call once per launch to confirm the protocol version.
    public func protocolVersion() throws -> Int {
        let response = try runJSON(arguments: ["capabilities"])
        guard let tool = response["tool"] as? [String: Any],
              let protocolVersion = tool["protocol"] as? Int else {
            throw AnselDenoiseFailure(code: "bad_response",
                                      message: "capabilities did not report tool.protocol",
                                      exitCode: 0)
        }
        return protocolVersion
    }

    /// Report the GPU the tool would use, or nil on a CPU-only machine.
    public func gpuName() throws -> String? {
        let response = try runJSON(arguments: ["capabilities"])
        return response["gpu"] as? String
    }

    /// Denoise `count` tiles already packed back-to-back in `input`.
    ///
    /// The caller owns tiling: this tool works on a flat float32 buffer of
    /// `count × tileSize²` normalized mosaic values and returns the same shape. Doing the tiling
    /// (and the feathered reassembly) outside keeps the GPL component ignorant of image layout
    /// and lets the host reuse its own overlap logic.
    public func denoiseBatch(model: URL,
                             inputTiles: URL,
                             count: Int,
                             output: URL,
                             options: AnselDenoiseOptions = AnselDenoiseOptions()) throws -> AnselDenoiseReport {
        precondition(count > 0, "tile count must be positive")
        precondition(options.noiseA.count == 3 && options.noiseB.count == 3,
                     "noiseA/noiseB must each have 3 channel values")

        var arguments = [
            "batch", model.path, inputTiles.path,
            "--tile", String(options.tileSize),
            "--count", String(count),
            "--a", options.noiseA.map { String(format: "%.9g", $0) }.joined(separator: ","),
            "--b", options.noiseB.map { String(format: "%.9g", $0) }.joined(separator: ","),
            "--sigma-scale", options.sigmaScale.map { String(format: "%.9g", $0) }.joined(separator: ","),
            "--cfa", options.cfa.lowercased(),
            "--json",
            "--out", output.path,
        ]
        if options.useGPU { arguments.insert("--gpu", at: 3) }

        let response = try runJSON(arguments: arguments)
        guard let tiles = response["tiles"] as? Int,
              let ms = response["ms"] as? Double,
              let backend = response["backend"] as? String,
              let cfa = response["cfa"] as? String else {
            throw AnselDenoiseFailure(code: "bad_response",
                                      message: "batch did not report tiles/ms/backend/cfa",
                                      exitCode: 0)
        }
        guard FileManager.default.fileExists(atPath: output.path) else {
            throw AnselDenoiseFailure(code: "no_output",
                                      message: "tool reported success but \(output.path) is missing",
                                      exitCode: 0)
        }
        return AnselDenoiseReport(output: output, tiles: tiles,
                                  tileSize: response["tileSize"] as? Int ?? options.tileSize,
                                  milliseconds: ms, backend: backend, cfa: cfa)
    }

    // MARK: - process plumbing

    private func runJSON(arguments: [String]) throws -> [String: Any] {
        var env = ProcessInfo.processInfo.environment
        env["LC_ALL"] = "C"

        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments
        process.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        do {
            try process.run()
        } catch {
            throw AnselDenoiseFailure(code: "launch_failed",
                                      message: "cannot launch \(executableURL.path): \(error)",
                                      exitCode: nil)
        }

        // Drain stderr concurrently: a full pipe would deadlock the child.
        var stderrData = Data()
        let errHandle = errPipe.fileHandleForReading
        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            stderrData = errHandle.readDataToEndOfFile()
            group.leave()
        }

        let stdoutData = outPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        group.wait()

        guard process.terminationStatus == 0 else {
            let msg = String(decoding: stderrData, as: UTF8.self)
            throw AnselDenoiseFailure(
                code: "tool_failed",
                message: msg.isEmpty ? "exit \(process.terminationStatus)" : msg,
                exitCode: process.terminationStatus)
        }

        // The tool writes exactly one JSON object per invocation; ignore trailing noise.
        let text = String(decoding: stdoutData, as: UTF8.self)
        guard let line = text.split(separator: "\n").last,
              let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
        else {
            throw AnselDenoiseFailure(code: "bad_json",
                                      message: "unparseable response: \(text.prefix(200))",
                                      exitCode: 0)
        }
        return obj
    }
}
