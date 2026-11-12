//
//  AnselDenoiseModel.swift
//  AnselDenoise
//
//  Reader + evaluator for Ansel's neural raw-denoise model (`.anselnn`).
//
//  This file is part of a GPLv3 component, vendored beside Photons and invoked
//  as a separate executable — see LICENSE-SCOPE.md. The network definition is
//  derived from `aurelienpierreeng/ansel-denoise` (GPL-3.0), `src/ansel_denoise/model.py`.
//
//  What the network is (from docs/architecture.md):
//    5 input planes at full resolution: [noisy mosaic, R one-hot, G one-hot,
//    B one-hot, sigma] → predicted noise plane; output = mosaid − noise (residual).
//    Depth 4, widths base·2^i, encoder [conv3x3→GELU]×2 + 2×2/2 conv,
//    decoder 2× nearest upsample → 1×1 conv → concat skip → [conv3x3→GELU]×2,
//    head conv3x3 → out_channels.
//
//  Deliberate omissions to keep parity with the C/OpenCL executor:
//    no normalization layers, nearest-neighbour upsampling, exact GELU.
//

import Foundation

/// Errors from model loading.
public enum AnselModelError: Error, CustomStringConvertible {
    case badMagic(String)
    case truncated(String)
    case headerTooShort
    case badJSON(String)
    case tensorMissing(String)
    case badConfig(String)

    public var description: String {
        switch self {
        case .badMagic(let m):      return "not an .anselnn file (magic \(m))"
        case .truncated(let w):     return "truncated file: \(w)"
        case .headerTooShort:       return "file too short to hold a header"
        case .badJSON(let w):       return "bad JSON header: \(w)"
        case .tensorMissing(let n): return "tensor missing from model: \(n)"
        case .badConfig(let w):     return "unsupported config: \(w)"
        }
    }
}

/// One tensor's location inside the file payload.
public struct AnselTensorInfo: Sendable {
    public let name: String
    public let shape: [Int]
    public let offset: Int
    public let size: Int
}

/// Architecture constants read from the model header.
public struct AnselModelConfig: Sendable {
    public let arch: String        // "unet"
    public let base: Int
    public let depth: Int
    public let inChannels: Int
    public let outChannels: Int
    public let codeRevision: String?

    /// Plane layout: signal planes followed by one-hot CFA planes and the sigma plane.
    public var signalChannels: Int { outChannels }
}

/// A weight tensor as a flat float array plus its 4-D shape (all convs are 2-D).
public struct AnselWeights: Sendable {
    public let name: String
    /// Conv layout is PyTorch's (outC, inC, kh, kw).
    public let shape: [Int]
    public let values: [Float]

    public var outC: Int { shape[0] }
    public var inC: Int { shape[1] }
    public var kh: Int { shape[2] }
    public var kw: Int { shape[3] }
}

/// Fully loaded model: config + every tensor, ready to evaluate.
public final class AnselDenoiseModel {
    public let config: AnselModelConfig
    public let tensors: [String: AnselWeights]
    public let tensorInfos: [AnselTensorInfo]

    /// Load and validate an `.anselnn` file.
    ///
    /// Format (from `doc/rawdenoiseai.md`):
    ///   bytes 0..8    magic "ANSELDN1"
    ///   bytes 8..12   uint32 LE length of the JSON header
    ///   bytes 12..    JSON header {cfg, tensors:[{name,shape,offset,size}]}
    ///   then          plain float32 payload; tensor `offset` is relative to payload start
    public convenience init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        try self.init(data: data)
    }

    public init(data: Data) throws {
        let magicLen = 8
        guard data.count >= magicLen + 4 else { throw AnselModelError.headerTooShort }

        let magic = String(decoding: data[0..<magicLen], as: UTF8.self)
        guard magic == "ANSELDN1" else { throw AnselModelError.badMagic(magic) }

        let headerLen = Int(data.withUnsafeBytes { raw -> UInt32 in
            raw.loadUnaligned(fromByteOffset: magicLen, as: UInt32.self)
        })
        guard headerLen > 0, data.count >= magicLen + 4 + headerLen else {
            throw AnselModelError.truncated("header claims \(headerLen) bytes, file has \(data.count)")
        }

        let headerStart = magicLen + 4
        let headerBytes = data[headerStart..<headerStart + headerLen]
        let payloadStart = headerStart + headerLen

        guard let root = try JSONSerialization.jsonObject(with: headerBytes) as? [String: Any]
        else { throw AnselModelError.badJSON("root is not an object") }

        guard let cfg = root["cfg"] as? [String: Any] else {
            throw AnselModelError.badJSON("no cfg")
        }
        let arch = (cfg["arch"] as? String) ?? "unet"
        guard arch == "unet" else {
            // "unet-ms" needs superpixel binning + a guide injection stage; not modelled here.
            throw AnselModelError.badConfig("arch '\(arch)' is not supported (only 'unet')")
        }
        let base = (cfg["base"] as? Int) ?? 32
        let depth = (cfg["depth"] as? Int) ?? 4
        let inCh = (cfg["in_channels"] as? Int) ?? 5
        let outCh = (cfg["out_channels"] as? Int) ?? 1
        self.config = AnselModelConfig(arch: arch, base: base, depth: depth,
                                       inChannels: inCh, outChannels: outCh,
                                       codeRevision: cfg["code_revision"] as? String)

        guard let rawTensors = root["tensors"] as? [[String: Any]] else {
            throw AnselModelError.badJSON("no tensors array")
        }

        var infos: [AnselTensorInfo] = []
        infos.reserveCapacity(rawTensors.count)
        for t in rawTensors {
            guard let name = t["name"] as? String,
                  let shape = t["shape"] as? [Int],
                  let offset = t["offset"] as? Int,
                  let size = t["size"] as? Int
            else { throw AnselModelError.badJSON("malformed tensor entry") }
            guard offset >= 0, payloadStart + offset + size <= data.count else {
                throw AnselModelError.truncated("tensor \(name) runs past end of payload")
            }
            infos.append(AnselTensorInfo(name: name, shape: shape, offset: offset, size: size))
        }
        self.tensorInfos = infos

        // Decode every tensor as little-endian float32.
        var loaded: [String: AnselWeights] = [:]
        loaded.reserveCapacity(infos.count)
        for info in infos {
            let count = info.shape.reduce(1, *)
            let expected = count * MemoryLayout<Float>.size
            guard info.size == expected else {
                throw AnselModelError.badJSON(
                    "tensor \(info.name) size \(info.size) != \(count) floats")
            }
            let start = payloadStart + info.offset
            let values: [Float] = data.withUnsafeBytes { raw in
                var out = [Float](repeating: 0, count: count)
                out.withUnsafeMutableBytes { dst in
                    dst.copyMemory(from: UnsafeRawBufferPointer(
                        rebasing: raw[start..<start + expected]))
                }
                return out
            }
            loaded[info.name] = AnselWeights(name: info.name, shape: info.shape, values: values)
        }
        self.tensors = loaded
    }

    /// Tensor by name, or throw — a missing tensor is a corrupt model, not a warning.
    public func weights(_ name: String) throws -> AnselWeights {
        guard let w = tensors[name] else { throw AnselModelError.tensorMissing(name) }
        return w
    }

    public var parameterCount: Int {
        tensors.values.reduce(0) { $0 + $1.values.count }
    }
}
