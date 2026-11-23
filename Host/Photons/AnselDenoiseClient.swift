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

// MARK: - Child process lifetime (451)

/// 451 — 「降噪到一半杀掉 app」 leaves the `ansel-denoise-swift` / `hdrmerge-swift` child running: a
/// `SIGKILL`ed host cannot signal it, and the child then finishes writing its hidden temp file.
/// This registry does the one thing a normal quit can do — terminate what we started.
///
/// It is **not** a process-group change and it does not try to survive `SIGKILL`; that is the
/// startup sweep's job (`InterruptedWorkSweep`). The registry exists so
/// `applicationWillTerminate` can ask the live `Process` objects to stop before the app goes away.
public final class ChildProcessRegistry: @unchecked Sendable {
    public static let shared = ChildProcessRegistry()

    private let lock = NSLock()
    private var handles: [ObjectIdentifier: ChildProcessHandle] = [:]

    private init() {}

    /// Start tracking a launched child. The returned handle is released with `release(_:)` when the
    /// call that owns the process is done; until then it is retained here so a quit hook can find it.
    func track(_ process: Process) -> ChildProcessHandle {
        let handle = ChildProcessHandle(process)
        lock.lock()
        handles[ObjectIdentifier(handle)] = handle
        lock.unlock()
        return handle
    }

    func release(_ handle: ChildProcessHandle) {
        lock.lock()
        handles.removeValue(forKey: ObjectIdentifier(handle))
        lock.unlock()
    }

    /// Best-effort `terminate()` for every child this process launched and has not finished waiting
    /// for. Called from `applicationWillTerminate`; safe to call more than once.
    public func terminateAll() {
        lock.lock()
        let live = Array(handles.values)
        lock.unlock()
        for handle in live { handle.terminateIfRunning() }
    }
}

final class ChildProcessHandle: @unchecked Sendable {
    let process: Process

    init(_ process: Process) {
        self.process = process
    }

    func terminateIfRunning() {
        if process.isRunning { process.terminate() }
    }

    deinit {
        terminateIfRunning()
    }
}

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

/// Which conditioning the tool is asked to use.
public enum AnselDenoiseLook: String, Sendable, Equatable, CaseIterable {
    /// Let the tool measure **this** frame's Poisson-Gaussian curve and derive the blend weight
    /// and the brightness tapers itself (`--look nlmf`): the same conditioning Photons' nonlocal
    /// denoiser runs, so the two engines hand back comparable results.
    ///
    /// Why this is the default: the shipped `noiseA`/`noiseB` are one camera's numbers. On an SL3
    /// at ISO 100 they over-state sigma by ~6x once `sigmaScale` is applied, which flattens the
    /// shadows; the measured curve costs ~3 s and removes that guesswork entirely.
    case nlmf
    /// Send the caller's `noiseA`/`noiseB`/`sigmaScale`/`strength` verbatim — the pre-`--look`
    /// contract, kept so an experiment (or an older tool binary) can still pin every number.
    case classic
}

public struct AnselDenoiseOptions: Sendable, Equatable {
    public var variant: AnselDenoiseVariant = .quarter
    /// Conditioning passed to the tool. `.nlmf` (default) sends `--look nlmf` and none of the
    /// numbers below; `.classic` sends them. The float32-mosaic entry point always sends the
    /// numbers: that path exists for experiments that pin every input.
    public var look: AnselDenoiseLook = .nlmf
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
    /// 0.85 was what Ansel ships — enough to visibly clean up, little enough that texture survives.
    /// Note this is **not** the same knob as `sigmaScale`. Sigma tells the network how much noise
    /// to expect (it re-runs inference and decides what to remove); strength only decides how much
    /// of the answer to keep. Lowering strength puts original noise back along with the detail.
    ///
    /// 396 — **the default is 0.50**, per the user's own call (「模型降噪的默认强度修改为0.50，即
    /// `--strength 0.50`」). At 0.85 the tool over-corrects: the residual is *cleaner* than the
    /// original texture, which reads as plastic on skin and foliage. 0.50 leaves more of the
    /// original grain in place. Measured on `L1030433.dng` by §386's dark-patch
    /// adjacent-difference variance, both numbers are printed by
    /// `AnselDenoiseStrength396Tests`; the value is a **default only** — the range check and the
    /// CLI parsing are untouched, so a caller that passes `strength:` explicitly still gets it.
    ///
    /// 口径修正（覆盖 §396 ①）：这条只在 `look == .classic` 时被发出去。默认的 `.nlmf` 由工具按
    /// 本帧测得的 σ 自己定力度（0.70–0.85），所以 app 的默认路径不再发 `--strength 0.50`。
    public var strength: Double = 0.50

    public init() {}
}

public struct AnselDenoiseReport: Sendable, Equatable {
    public let output: URL
    /// How many pieces the image was handed to the network in. 1 means a single pass, no seams.
    public let pieces: Int
    public let columns: Int
    public let rows: Int
    public let seams: Int
    public let singlePass: Bool
    /// Bytes the largest single forward pass was expected to need.
    public let peakBytes: Int
    public let milliseconds: Double
    public let backend: String
    public let cfa: String
}

/// What the tool says about one DNG-in / DNG-out run.
///
/// Only fields the tool actually reports are here. In particular `readSeconds`/`writeSeconds` exist
/// so a slow run can be attributed without guessing: on 60 MP they are 2.2 s and 0.9 s against
/// 38.3 s of inference, which is the whole point of moving file IO into the tool.
public struct AnselDenoiseDNGReport: Sendable, Equatable {
    public let output: URL
    public let width: Int
    public let height: Int
    /// CFA as the source file states it (tag 33422).
    public let cfa: String
    /// CFA the network was told. Differs from `cfa` only when the stored image is larger than its
    /// `ActiveArea`, or when the caller overrode it.
    public let mosaicCfa: String
    public let pieces: Int
    public let columns: Int
    public let rows: Int
    /// Internal cuts that needed feathering. Zero means the frame went through in one pass.
    public let seams: Int
    public let seconds: Double
    public let readSeconds: Double
    public let inferenceSeconds: Double
    public let writeSeconds: Double
    public let backend: String
    public let outputBytes: Int
    /// Samples the network produced outside [0,1] and the tool clamped. Non-zero is normal.
    public let clampedNegatives: Int
    public let clampedOverOne: Int

    init?(response: [String: Any], output: URL) {
        func int(_ key: String) -> Int? { (response[key] as? NSNumber)?.intValue }
        func double(_ key: String) -> Double { (response[key] as? NSNumber)?.doubleValue ?? 0 }
        guard let width = int("width"), let height = int("height"),
              let cfa = response["cfa"] as? String, width > 0, height > 0 else { return nil }
        self.output = output
        self.width = width
        self.height = height
        self.cfa = cfa
        self.mosaicCfa = response["mosaicCfa"] as? String ?? cfa
        self.pieces = int("pieces") ?? 1
        self.columns = int("columns") ?? 1
        self.rows = int("rows") ?? 1
        self.seams = int("seams") ?? 0
        self.seconds = double("seconds")
        self.readSeconds = double("readSeconds")
        self.inferenceSeconds = double("inferenceSeconds")
        self.writeSeconds = double("writeSeconds")
        self.backend = response["backend"] as? String ?? "unknown"
        self.outputBytes = int("outputBytes") ?? 0
        self.clampedNegatives = int("clampedNegatives") ?? 0
        self.clampedOverOne = int("clampedOverOne") ?? 0
    }
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

    /// `true` when the tool can take a `.dng` in and write one back — i.e. when the host can stop
    /// doing the mosaic round trip itself. Checked once per launch.
    public func supportsDNG() throws -> Bool {
        let response = try runJSON(arguments: ["capabilities"])
        guard let features = response["features"] as? [String] else { return false }
        return features.contains("image-dng")
    }

    /// Denoise a DNG: **two paths, one JSON object.** The tool reads the RAW, denoises the mosaic,
    /// and writes a DNG whose geometry and metadata are the source's own.
    ///
    /// This is the entry point a host should use. Everything that used to justify the host owning
    /// the pixels — normalization against the file's black/white levels, tiling, feathered
    /// reassembly, the writer — is inside the tool, and both sides of it come from one file, so
    /// they cannot disagree.
    ///
    /// On a 60 MP frame this is 42 s in total, of which 2.2 s is reading and 0.9 s writing. The
    /// old shape (host reads → writes float32 → tool → host reads back → writes DNG) took 350 s,
    /// with 284 s of that in the host moving samples around.
    ///
    /// - Parameters:
    ///   - input: the source DNG. Its black level, white level and CFA phase are read from it.
    ///   - output: where to write the denoised DNG. Its parent directory must exist; the tool
    ///     writes no partial file on failure.
    public func denoiseImage(model: URL,
                             input: URL,
                             output: URL,
                             options: AnselDenoiseOptions = AnselDenoiseOptions()) throws -> AnselDenoiseDNGReport {
        precondition(input.pathExtension.lowercased() == "dng", "input must be a .dng")
        precondition(output.pathExtension.lowercased() == "dng", "output must be a .dng")
        // With `--look nlmf` the host sends no coefficients at all; requiring them would make the
        // default configuration trap on its own precondition.
        precondition(options.look == .nlmf || (options.noiseA.count == 3 && options.noiseB.count == 3),
                     "noiseA/noiseB must each have 3 channel values when look == .classic")

        var arguments = ["image", model.path, input.path]
        if options.useGPU { arguments.append("--gpu") }
        switch options.look {
        case .nlmf:
            // The tool measures the curve from the file, derives the weight from it and applies
            // the same brightness tapers the nonlocal path uses. Nothing about this frame's noise
            // is guessed on the host side — `--a/--b` are not sent at all.
            arguments += ["--look", "nlmf"]
        case .classic:
            arguments += ["--a", Self.csv(options.noiseA),
                          "--b", Self.csv(options.noiseB),
                          "--sigma-scale", Self.csv(options.sigmaScale),
                          "--strength", String(format: "%.9g", options.strength)]
        }
        arguments += ["--json", "--out", output.path]

        let response = try runJSON(arguments: arguments)
        guard let report = AnselDenoiseDNGReport(response: response, output: output) else {
            throw AnselDenoiseFailure(code: "bad_response",
                                      message: "image did not report width/height/cfa",
                                      exitCode: 0)
        }
        // The tool exits 0 only after writing, and a missing file is a bug rather than a race — but
        // a host that renders from this path should not be the first to notice it.
        guard FileManager.default.fileExists(atPath: output.path) else {
            throw AnselDenoiseFailure(code: "no_output",
                                      message: "tool reported success but \(output.path) is missing",
                                      exitCode: 0)
        }
        return report
    }

    /// Denoise a whole mosaic in one call.
    ///
    /// This entry point always sends explicit `--a/--b` (whatever `look` says): it exists for
    /// experiments and benchmarks that pin every input, and its caller already holds the numbers.
    ///
    /// Everything about fitting the image into memory happens on the other side of the process
    /// boundary: the tool decides whether the frame goes through in one pass or has to be cut, how
    /// many pieces that needs, where the cuts go, and how to feather them. This side stays a
    /// transport — write float32, run, read float32.
    ///
    /// That placement is deliberate. Splitting exists because of *that implementation's* memory
    /// profile (0.64 GB per megapixel), so hard-coding a piece count here would mean two programs
    /// agreeing on a number that changes whenever those kernels change.
    ///
    /// The tool removes the artefact that motivated this design: the network denoises less
    /// aggressively near an input edge, so tile borders used to leave visible structure. A single
    /// pass over 60 MP is also faster (55.8 s vs 77.9 s as 1426 tiles) because tiling spends about
    /// a third of its work on overlap that is then blended away.
    ///
    /// - Parameters:
    ///   - input: raw float32, `width × height`, normalized to [0,1], row-major.
    ///   - output: same shape written here.
    public func denoiseImage(model: URL,
                             input: URL,
                             width: Int,
                             height: Int,
                             output: URL,
                             options: AnselDenoiseOptions = AnselDenoiseOptions()) throws -> AnselDenoiseReport {
        precondition(width > 0 && height > 0, "image dimensions must be positive")
        precondition(options.noiseA.count == 3 && options.noiseB.count == 3,
                     "noiseA/noiseB must each have 3 channel values")

        var arguments = [
            "image", model.path, input.path,
            "--width", String(width),
            "--height", String(height),
            "--a", options.noiseA.map { String(format: "%.9g", $0) }.joined(separator: ","),
            "--b", options.noiseB.map { String(format: "%.9g", $0) }.joined(separator: ","),
            "--sigma-scale", options.sigmaScale.map { String(format: "%.9g", $0) }.joined(separator: ","),
            "--cfa", options.cfa.lowercased(),
            "--strength", String(format: "%.9g", options.strength),
            "--json",
            "--out", output.path,
        ]
        if options.useGPU { arguments.insert("--gpu", at: 3) }

        let response = try runJSON(arguments: arguments)
        guard let ms = response["ms"] as? Double,
              let backend = response["backend"] as? String,
              let cfa = response["cfa"] as? String else {
            throw AnselDenoiseFailure(code: "bad_response",
                                      message: "image run did not report ms/backend/cfa",
                                      exitCode: 0)
        }
        guard FileManager.default.fileExists(atPath: output.path) else {
            throw AnselDenoiseFailure(code: "no_output",
                                      message: "tool reported success but \(output.path) is missing",
                                      exitCode: 0)
        }
        return AnselDenoiseReport(output: output,
                                  pieces: response["pieces"] as? Int ?? 1,
                                  columns: response["columns"] as? Int ?? 1,
                                  rows: response["rows"] as? Int ?? 1,
                                  seams: response["seams"] as? Int ?? 0,
                                  singlePass: response["singlePass"] as? Bool ?? true,
                                  peakBytes: response["peakBytes"] as? Int ?? 0,
                                  milliseconds: ms, backend: backend, cfa: cfa)
    }

    /// Comma-separated, at the precision the CLI parses back without rounding drift.
    private static func csv(_ values: [Double]) -> String {
        values.map { String(format: "%.9g", $0) }.joined(separator: ",")
    }

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
        // 451 — registered only after a successful launch, released once this call stops waiting.
        let tracked = ChildProcessRegistry.shared.track(process)
        defer { ChildProcessRegistry.shared.release(tracked) }

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
