//
//  main.swift — ansel-denoise-swift command line
//
//  GPLv3 component. Photons invokes this as a child process; the MIT side never
//  links this code. See LICENSE-SCOPE.md.
//
//  Subcommands
//    info    <model.anselnn>
//    image   <model> <in.dng|in.f32> [--width W --height H] [--a r,g,b] [--b r,g,b]
//            [--cfa rggb] [--gpu] [--strength S] [--budget-gb G]
//            [--split CxR] --out <out.dng|out.f32> [--json]
//        Whole-image entry point: plans for memory, splits the few times it can,
//        feathers the cuts, blends `strength`. Hosts should use this rather than `tile`.
//        The format is chosen by extension: `.dng` on the input means the tool reads the RAW
//        itself (black/white/CFA from the file) and writes a DNG back, so no mosaic ever has to
//        cross the process boundary. Anything else is the older float32-mosaic path.
//    bench   <model.anselnn> [--gpu] [--tiles 64,128,256,512] [--compare-cpu]
//        Sweep block sizes on one backend, or compare both backends.
//

import Foundation
import Metal
import AnselDenoise

func die(_ msg: String) -> Never {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
    exit(2)
}

func parseDoubles(_ s: String) -> [Double] {
    s.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
}

let args = Array(CommandLine.arguments.dropFirst())
guard let cmd = args.first else {
    die("usage: ansel-denoise-swift <image|info|tile|batch|bench|capabilities> ...")
}

func flag(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

let hasFlag: (String) -> Bool = { name in args.contains(name) }

func readFloats(_ path: String, expected: Int) -> [Float] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    guard data.count == expected * MemoryLayout<Float>.size else {
        die("\(path): \(data.count) bytes, expected \(expected * 4)")
    }
    var buf = [Float](repeating: 0, count: expected)
    buf.withUnsafeMutableBytes { data.copyBytes(to: $0) }
    return buf
}

func writeFloats(_ values: [Float], to path: String) {
    let d = values.withUnsafeBufferPointer { Data(buffer: $0) }
    try! d.write(to: URL(fileURLWithPath: path))
}

/// Machine-readable response on stdout, newline-terminated, exactly one object.
func emitJSON(_ object: [String: Any]) -> Never {
    let data = (try? JSONSerialization.data(withJSONObject: object,
                                            options: [.sortedKeys])) ?? Data("{}".utf8)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
    exit(0)
}

/// Replicate-pad a mosaic up to a multiple of `mult`.
func padToMultiple(_ mosaic: [Float], w: Int, h: Int, mult: Int)
    -> (padded: [Float], W: Int, H: Int, ph: Int, pw: Int) {
    let padH = ((h + mult - 1) / mult) * mult - h
    let padW = ((w + mult - 1) / mult) * mult - w
    let ph = padH / 2, pw = padW / 2
    let H = h + padH, W = w + padW
    var padded = [Float](repeating: 0, count: W * H)
    for y in 0..<H {
        let sy = min(max(y - ph, 0), h - 1)
        for x in 0..<W {
            let sx = min(max(x - pw, 0), w - 1)
            padded[y * W + x] = mosaic[sy * w + sx]
        }
    }
    return (padded, W, H, ph, pw)
}

/// The CFA at the tile's top-left, so replicate padding keeps the phase.
func cfaString(_ pattern: String, _ shifts: Int) -> String {
    let p = Array(pattern.lowercased())
    guard p.count == 4, shifts % 2 == 1 else { return pattern }
    // An odd pixel shift swaps columns; an odd total swaps both rows and columns,
    // which for a 2×2 pattern is the same swap on the diagonal pair.
    return String([p[1], p[0], p[3], p[2]])
}

/// Run one subcommand, turning any thrown error into a one-line message and a non-zero exit.
///
/// Without this a thrown `DenoiseDNGError` reaches the top level as an uncaught Swift error: the
/// process prints a `Fatal error: Error raised at top level:` banner plus a backtrace and dies on
/// SIGTRAP (exit 133). A host driving this over a pipe cannot tell that apart from a crash, and a
/// SIGTRAP looks nothing like "this file is not a DNG". One line on stderr and exit 2 is the
/// protocol's contract.
func runMain(_ body: () throws -> Void) -> Never {
    do {
        try body()
        exit(0)
    } catch {
        let message: String
        if let described = error as? CustomStringConvertible {
            message = described.description
        } else {
            message = "\(error)"
        }
        die("error: \(message)")
    }
}

runMain {
switch cmd {
case "info":
    guard args.count >= 2 else { die("usage: info <model.anselnn>") }
    let m = try AnselDenoiseModel(contentsOf: URL(fileURLWithPath: args[1]))
    let c = m.config
    print("arch: \(c.arch)  base: \(c.base)  depth: \(c.depth)")
    print("in: \(c.inChannels)  out: \(c.outChannels)")
    print("revision: \(c.codeRevision ?? "-")")
    print("tensors: \(m.tensorInfos.count)  parameters: \(m.parameterCount)")

case "bench":
    guard args.count >= 2 else { die("usage: bench <model.anselnn> [--gpu] [--compare-cpu]") }
    let m = try AnselDenoiseModel(contentsOf: URL(fileURLWithPath: args[1]))
    let tileList = (flag("--tiles") ?? "64,128,256,512")
        .split(separator: ",").compactMap { Int($0) }
    let wantGPU = hasFlag("--gpu") || hasFlag("--compare-cpu")
    let wantCPU = hasFlag("--compare-cpu") || !hasFlag("--gpu")

    print("model: \(args[1])")
    print("params: \(m.parameterCount)  base: \(m.config.base)  depth: \(m.config.depth)")
    var engine: MetalConv?
    if wantGPU {
        do { engine = try MetalConv() }
        catch { die("Metal init failed: \(error)") }
        print("gpu: \(engine!.device.name)")
    }
    print("")
    print("tile        cpu_ms    gpu_ms   speedup")
    print(String(repeating: "-", count: 40))

    for t in tileList {
        var cpu = Double.nan, gpu = Double.nan
        if wantCPU {
            do { cpu = try AnselBench.timeCPU(model: m, tile: t) }
            catch { print("  cpu \(t): \(error)"); continue }
        }
        if let e = engine {
            do { gpu = try AnselBench.timeGPU(model: m, engine: e, tile: t) }
            catch { print("  gpu \(t): \(error)"); gpu = .nan }
        }
        let sp: String
        if cpu.isFinite && gpu.isFinite && gpu > 0 {
            sp = String(format: "%.1fx", cpu / gpu)
        } else { sp = "-" }
        let cStr = cpu.isFinite ? String(format: "%.1f", cpu) : "-"
        let gStr = gpu.isFinite ? String(format: "%.1f", gpu) : "-"
        print(String(format: "%-6d %10@ %9@ %9@", t, cStr as NSString, gStr as NSString, sp as NSString))
    }

case "image":
    // Whole-image entry point, in two flavours chosen by extension.
    //
    //   .dng → the tool reads the RAW and writes a DNG. Black level, white level and CFA phase come
    //          from the file; the mosaic never leaves this process.
    //   else → the older float32-mosaic-in / float32-mosaic-out path, still used by the host's
    //          legacy callers and by `bench`.
    guard args.count >= 3 else { die("usage: image <model> <in.dng|in.f32> --out <out.dng|out.f32> ...") }
    guard let outPath = flag("--out") else { die("missing --out") }

    let inPath = args[2]
    let isDNG = inPath.lowercased().hasSuffix(".dng")
    let outIsDNG = outPath.lowercased().hasSuffix(".dng")
    // Mixing the two is almost always a typo, and the resulting file would be a DNG named .f32 or
    // raw floats named .dng. Refuse instead of guessing.
    guard isDNG == outIsDNG else {
        die("input and output must be the same format: got \(inPath) → \(outPath)")
    }

    let sigmaScale = flag("--sigma-scale").map { parseDoubles($0) } ?? [2.82, 3.94, 2.96]
    guard sigmaScale.count == 3 else { die("--sigma-scale needs 3 values") }

    // The DNG path can be called with no noise model at all, in which case it uses the same
    // per-camera defaults the host used to send. `Var = a·x + b` on [0,1].
    let defaultA = [0.0004, 0.0003, 0.0004]
    let defaultB = [3e-6, 1e-6, 1e-6]
    let aRaw = flag("--a").map { parseDoubles($0) } ?? (isDNG ? defaultA : [])
    let bRaw = flag("--b").map { parseDoubles($0) } ?? (isDNG ? defaultB : [])
    guard aRaw.count == 3, bRaw.count == 3 else { die("--a and --b each need 3 values") }
    let a = aRaw.enumerated().map { $0.element * sigmaScale[$0.offset] * sigmaScale[$0.offset] }
    let b = bRaw.enumerated().map { $0.element * sigmaScale[$0.offset] * sigmaScale[$0.offset] }

    let model = try AnselDenoiseModel(contentsOf: URL(fileURLWithPath: args[1]))
    var engine: MetalConv?
    if hasFlag("--gpu") { engine = try MetalConv() }

    let budgetBytes = flag("--budget-gb").flatMap(Double.init).map { Int($0 * 1e9) }
    var forceColumns: Int? = nil, forceRows: Int? = nil
    // --split CxR forces a cut pattern; useful for measuring seams and for regression tests.
    if let s = flag("--split") {
        let parts = s.lowercased().split(separator: "x")
        if parts.count == 2, let c = Int(parts[0]), let r = Int(parts[1]) {
            forceColumns = c
            forceRows = r
        } else {
            die("--split wants CxR, e.g. 2x1")
        }
    }

    let progress: (String) -> Void = { msg in
        FileHandle.standardError.write(Data((msg + "\n").utf8))
    }

    if isDNG {
        let request = DenoiseDNGRequest(
            model: model, metal: engine,
            input: URL(fileURLWithPath: inPath),
            output: URL(fileURLWithPath: outPath),
            a: a, b: b,
            cfaOverride: flag("--cfa"),
            strength: flag("--strength").flatMap(Double.init) ?? 0.85,
            budgetBytes: budgetBytes,
            forceColumns: forceColumns, forceRows: forceRows,
            // Given, these are checked against the file — never preferred over it.
            expectedWidth: flag("--width").flatMap(Int.init),
            expectedHeight: flag("--height").flatMap(Int.init))

        let result = try DenoiseDNGPipeline.run(request, progress: progress)

        if hasFlag("--json") {
            emitJSON([
                "pieces": result.pieces,
                "columns": result.columns,
                "rows": result.rows,
                "seams": result.seams,
                "peakBytes": result.peakBytes,
                "budgetBytes": result.budgetBytes,
                "wholeImageBytes": result.wholeImageBytes,
                "singlePass": result.singlePass,
                "ms": result.seconds * 1000,
                "seconds": result.seconds,
                "readSeconds": result.readSeconds,
                "inferenceSeconds": result.inferenceSeconds,
                "writeSeconds": result.writeSeconds,
                "backend": result.backend,
                "cfa": result.fileCFA,
                "mosaicCfa": result.mosaicCFA,
                "inputPath": inPath,
                "outputPath": outPath,
                "output": outPath,
                "width": result.width,
                "height": result.height,
                "inputLayout": "dng-cfa",
                "outputLayout": "dng-cfa",
                "pattern": result.fileCFA,
                "blackLevels": result.normalization.blackByPhase,
                "whiteLevel": result.normalization.white,
                "blackRepeatRows": result.normalization.repeatRows,
                "blackRepeatCols": result.normalization.repeatCols,
                "outputBytes": result.outputBytes,
                "tiles": result.tileCount,
                "clampedNegatives": result.clampedNegatives,
                "clampedOverOne": result.clampedOverOne,
                "finalMin": Double(result.finalMin),
                "finalMax": Double(result.finalMax),
                "sourceBelowBlack": result.sourceBelowBlack,
            ])
        } else {
            let summary = [
                "\(inPath) → \(outPath): \(result.width)×\(result.height) \(result.fileCFA), "
                + "\(result.pieces) piece(s) \(result.columns)×\(result.rows), "
                + "\(result.seams) seam(s), peak \(String(format: "%.2f", Double(result.peakBytes) / 1e9)) GB "
                + "of \(String(format: "%.2f", Double(result.budgetBytes) / 1e9)) GB",
                "  read \(String(format: "%.2f", result.readSeconds)) s"
                + " + infer \(String(format: "%.2f", result.inferenceSeconds)) s"
                + " + write \(String(format: "%.2f", result.writeSeconds)) s"
                + " = \(String(format: "%.2f", result.seconds)) s (\(result.backend))",
            ].joined(separator: "\n") + "\n"
            FileHandle.standardError.write(Data(summary.utf8))
        }
        exit(0)
    }

    // MARK: float32 mosaic path (unchanged behaviour)

    guard let w = flag("--width").flatMap(Int.init),
          let h = flag("--height").flatMap(Int.init),
          flag("--a") != nil, flag("--b") != nil
    else { die("missing --width/--height/--a/--b/--out") }

    var request = DenoiseImageRequest(
        input: URL(fileURLWithPath: inPath),
        output: URL(fileURLWithPath: outPath),
        width: w, height: h,
        a: a, b: b,
        sigmaScale: sigmaScale,
        cfa: flag("--cfa") ?? "rggb",
        useGPU: hasFlag("--gpu"),
        strength: flag("--strength").flatMap(Double.init) ?? 0.85,
        budgetBytes: budgetBytes,
        forceColumns: forceColumns, forceRows: forceRows)

    let t0 = Date()
    let result = try DenoiseImageRunner.run(request, model: model, metal: engine, progress: progress)
    let ms = Date().timeIntervalSince(t0) * 1000

    if hasFlag("--json") {
        emitJSON([
            "pieces": result.pieces,
            "columns": result.plan.columns,
            "rows": result.plan.rows,
            "seams": result.plan.seams,
            "peakBytes": result.plan.peakBytes,
            "budgetBytes": result.plan.budgetBytes,
            "wholeImageBytes": result.plan.wholeImageBytes,
            "singlePass": result.plan.isSinglePass,
            "ms": ms,
            "seconds": ms / 1000,
            "backend": result.backend,
            "cfa": request.cfa,
            "inputPath": inPath,
            "outputPath": outPath,
            "output": outPath,
            "clampedNegatives": result.clampedNegatives,
            "finalMin": Double(result.finalMin),
            "finalMax": Double(result.finalMax),
        ])
    } else {
        FileHandle.standardError.write(Data(String(
            format: "image %dx%d: %d piece(s) %dx%d, %d seam(s), peak %.2f GB of %.2f GB, %.1f ms\n",
            w, h, result.pieces, result.plan.columns, result.plan.rows, result.plan.seams,
            Double(result.plan.peakBytes) / 1e9, Double(result.plan.budgetBytes) / 1e9, ms).utf8))
    }

case "tile", "batch":
    guard args.count >= 3 else { die("usage: \(cmd) <model> <in.f32> ...") }
    guard let aStr = flag("--a"), let bStr = flag("--b"), let outPath = flag("--out")
    else { die("missing --a/--b/--out") }
    // Ansel's shipped per-channel sigma calibration: an exact x2 from the
    // lifting-Haar profile fitting, times the per-channel demosaic-attenuation
    // medians (R 1.41, G 1.97, B 1.48). Without it the network sees sigma 3-4x
    // too small and under-denoises.
    let sigmaScale = flag("--sigma-scale").map { parseDoubles($0) } ?? [2.82, 3.94, 2.96]
    guard sigmaScale.count == 3 else { die("--sigma-scale needs 3 values") }
    let a = parseDoubles(aStr).enumerated().map { $0.element * sigmaScale[$0.offset] * sigmaScale[$0.offset] }
    let b = parseDoubles(bStr).enumerated().map { $0.element * sigmaScale[$0.offset] * sigmaScale[$0.offset] }
    guard a.count == 3, b.count == 3 else { die("--a and --b each need 3 values") }
    let cfa = flag("--cfa") ?? "rggb"
    let useGPU = hasFlag("--gpu")

    let model = try AnselDenoiseModel(contentsOf: URL(fileURLWithPath: args[1]))
    let mult = 1 << model.config.depth
    var engine: MetalConv?
    if useGPU { engine = try MetalConv() }

    if cmd == "tile" {
        guard let w = flag("--width").flatMap(Int.init),
              let h = flag("--height").flatMap(Int.init)
        else { die("missing --width/--height") }
        let mosaic = readFloats(args[2], expected: w * h)
        let (padded, W, H, ph, pw) = padToMultiple(mosaic, w: w, h: h, mult: mult)

        var out: [Float]
        if let e = engine {
            let gpu = AnselDenoiseMetal(model: model, metal: e)
            out = try gpu.denoise(mosaic: padded, width: W, height: H,
                                  cfa: cfa, a: a, b: b)
        } else {
            let fwd = AnselDenoiseForward(model: model)
            let color = AnselDenoiseForward.colorIndexMap(pattern: cfa, width: W, height: H)
            let planes = AnselDenoiseForward.makeInputPlanes(
                mosaic: padded, colorIndex: color, width: W, height: H, a: a, b: b)
            out = try fwd.denoise(planes).data
        }
        var cropped = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w { cropped[y * w + x] = out[(y + ph) * W + (x + pw)] }
        }
        writeFloats(cropped, to: outPath)
        FileHandle.standardError.write(Data("tile \(w)x\(h) -> \(outPath)\n".utf8))

    } else {
        guard let tileN = flag("--tile").flatMap(Int.init),
              let count = flag("--count").flatMap(Int.init)
        else { die("missing --tile/--count") }
        let input = readFloats(args[2], expected: tileN * tileN * count)
        let tileLen = tileN * tileN
        var out = [Float](repeating: 0, count: input.count)

        let t0 = Date()
        if let e = engine {
            let gpu = AnselDenoiseMetal(model: model, metal: e)
            for t in 0..<count {
                let mosaic = Array(input[(t * tileLen)..<((t + 1) * tileLen)])
                let r = try gpu.denoise(mosaic: mosaic, width: tileN, height: tileN,
                                        cfa: cfa, a: a, b: b)
                out.replaceSubrange((t * tileLen)..<((t + 1) * tileLen), with: r)
            }
        } else {
            let fwd = AnselDenoiseForward(model: model)
            let batch = TileBatch(tileSize: tileN, count: count, data: input)
            out = try fwd.denoiseBatch(batch, cfa: cfa, a: a, b: b)
        }
        let dt = Date().timeIntervalSince(t0)
        writeFloats(out, to: outPath)
        let ms = dt * 1000
        if hasFlag("--json") {
            // Same shape as hdrmerge-swift's responses: machine-readable on stdout,
            // progress on stderr, pixel data in files.
            emitJSON([
                "tiles": count,
                "tileSize": tileN,
                "ms": ms,
                "msPerTile": ms / Double(count),
                "output": outPath,
                "cfa": cfa,
                "backend": useGPU ? "gpu" : "cpu",
            ])
        } else {
            FileHandle.standardError.write(Data(String(
                format: "%@ %d tiles @ %d2 in %.1f ms (%.2f ms/tile)\n",
                cmd, count, tileN, ms, ms / Double(count)).utf8))
        }
    }

case "capabilities":
    // Protocol probe, mirroring hdrmerge-swift. The host calls this once per launch
    // so a mismatched bundled binary fails loudly instead of producing bad pixels.
    var info: [String: Any] = [
        "tool": ["name": "ansel-denoise-swift", "protocol": 2],
        "features": ["image", "image-dng", "tile", "batch", "bench", "info", "gpu"],
        "cfaPatterns": ["rggb", "bggr", "grbg", "gbrg"],
        "inputLayout": "float32-planar-mosaic",
        "outputLayout": "float32-planar-mosaic",
        // `image` also takes `.dng` on both sides; the extension picks the layout. A host that
        // only knows protocol 1 keeps working through the float32 fields above.
        "imageInputLayouts": ["float32-planar-mosaic", "dng-cfa"],
        "imageOutputLayouts": ["float32-planar-mosaic", "dng-cfa"],
    ]
    if let d = MTLCreateSystemDefaultDevice() {
        info["gpu"] = d.name
    } else {
        info["gpu"] = NSNull()
    }
    emitJSON(info)

case "models":
    // Report the config of every .anselnn passed in, so the host can show a choice
    // without parsing the binary header itself.
    var list: [[String: Any]] = []
    for p in args.dropFirst() {
        if let m = try? AnselDenoiseModel(contentsOf: URL(fileURLWithPath: p)) {
            list.append([
                "path": p,
                "base": m.config.base,
                "depth": m.config.depth,
                "params": m.parameterCount,
                "revision": m.config.codeRevision ?? "",
            ])
        }
    }
    emitJSON(["models": list])

default:
    die("unknown command '\(cmd)'")
}
}
