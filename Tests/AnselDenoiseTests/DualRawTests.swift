//
//  DualRawTests.swift
//
//  `--dual-raw` file layout: IFD0 = preview, `SubIFDs = [original, denoised]`, both CFA raws.
//
//  What is worth testing, and why
//  ------------------------------
//  The whole point of `--dual-raw` is that a standard reader still sees the *original*. ImageIO
//  resolves a multi-raw DNG by scanning `SubIFDs` and taking the **first** entry whose
//  `NewSubfileType` is 0 (`CIRAWFilter` was measured doing exactly that, both ways round), so the
//  file is only correct if (a) `SubIFDs[0]` carries the original samples, (b) `SubIFDs[1]` carries
//  the denoised ones, (c) IFD0 is the preview page rather than a third raw, and (d) IFD0 keeps the
//  EXIF/GPS pointers.
//
//  A tag-only check cannot see a mis-ordered pointer: both raws would be present and the picture
//  would simply be the denoised one. So every assertion below decodes pixels back out of the IFD it
//  claims to be testing and compares them with the array that was handed in.

import XCTest
@testable import AnselDenoise

final class DualRawTests: XCTestCase {

    static var fixtureURL: URL? {
        let here = URL(fileURLWithPath: #filePath)
        let root = here.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = root.appendingPathComponent("Tests/Fixtures/small-rggb.dng")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    func scratch(_ name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ansel-dual-raw-tests/\(name)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Decode the mosaic of one specific IFD (tiles, compression 7 — what the writer emits).
    ///
    /// `SourceDNG.readMosaic` always decodes `raw`, i.e. `SubIFDs[0]`; the alternate lives at
    /// another offset, so the test walks `324`/`325` itself rather than reusing it.
    static func decode(ifd: [UInt16: TIFFEntry], data: Data)
        throws -> (samples: [UInt16], width: Int, height: Int) {
        let width = Int(try XCTUnwrap(ifd[256]?.firstUInt32))
        let height = Int(try XCTUnwrap(ifd[257]?.firstUInt32))
        let tileWidth = Int(try XCTUnwrap(ifd[322]?.firstUInt32))
        let tileHeight = Int(try XCTUnwrap(ifd[323]?.firstUInt32))
        let offsets = try XCTUnwrap(ifd[324]).uint32Array
        let counts = try XCTUnwrap(ifd[325]).uint32Array
        XCTAssertEqual(offsets.count, counts.count)
        XCTAssertEqual(offsets.count,
                       ((width + tileWidth - 1) / tileWidth) * ((height + tileHeight - 1) / tileHeight))

        var samples = [UInt16](repeating: 0, count: width * height)
        var index = 0
        for ty in 0..<((height + tileHeight - 1) / tileHeight) {
            for tx in 0..<((width + tileWidth - 1) / tileWidth) {
                let start = Int(offsets[index]), count = Int(counts[index])
                XCTAssertLessThanOrEqual(start + count, data.count, "tile \(index) runs past EOF")
                let blob = [UInt8](data[start..<start + count])
                let pixels = try SOF3.decode(blob, expectWidth: tileWidth, expectHeight: tileHeight)
                for y in 0..<tileHeight {
                    let dy = ty * tileHeight + y
                    if dy >= height { break }
                    let copyWidth = min(tileWidth, width - tx * tileWidth)
                    for x in 0..<copyWidth {
                        samples[dy * width + tx * tileWidth + x] = pixels[y * tileWidth + x]
                    }
                }
                index += 1
            }
        }
        return (samples, width, height)
    }

    // MARK: - the two-layer file

    func testOriginalIsFirstSubIFDAndDenoisedIsSecond() throws {
        guard let fixture = Self.fixtureURL else { throw XCTSkip("fixture not present") }
        let dir = try scratch("two-layers")
        let out = dir.appendingPathComponent("dual.dng")

        let sourceData = try Data(contentsOf: fixture)
        let source = try SourceDNG(url: fixture)
        let (primary, width, height) = try source.readMosaic(data: sourceData)
        XCTAssertEqual(width, 1024)

        // An alternate that differs everywhere and cannot be confused with the original by
        // accident: half the level, plus a per-pixel pattern.
        let alternate = primary.enumerated().map { i, v -> UInt16 in
            UInt16(max(0, min(65535, Int(v) / 2 + (i % 97))))
        }
        let changed = zip(primary, alternate).filter { $0 != $1 }.count
        XCTAssertGreaterThan(changed, primary.count / 2, "the two layers must differ")

        let result = try DNGRawWriter.write(
            primary: .init(samples: primary, label: "original"),
            alternate: .init(samples: alternate, label: "denoised alternate: test"),
            width: width, height: height,
            from: fixture, to: out)

        XCTAssertTrue(result.hasAlternate)
        XCTAssertEqual(result.tileCount, result.alternateTileCount)
        XCTAssertGreaterThan(result.alternateByteCount, 0)
        XCTAssertGreaterThan(result.byteCount, 0)

        let data = try Data(contentsOf: out)
        let dst = try SourceDNG(data: data)

        // (a) IFD0 is the preview page: subfiletype 1, strips, not CFA, and it ends the chain.
        XCTAssertEqual(dst.ifd0[254]?.firstUInt32, 1, "IFD0 must be the reduced-resolution preview")
        XCTAssertNotEqual(dst.ifd0[262]?.firstUInt32, 32803, "IFD0 must not be a raw")
        XCTAssertNotNil(dst.ifd0[273], "the preview keeps its strips")
        XCTAssertEqual(TIFFReader.nextIFDOffset(data, after: 8), 0, "IFD0 ends the page chain")
        XCTAssertNotNil(dst.ifd0[34665], "EXIF hangs off IFD0")
        XCTAssertNotNil(dst.ifd0[34853], "GPS hangs off IFD0")

        // (b) `330 SubIFDs` has exactly two entries, both CFA raws of the full size.
        let subIFDs = try XCTUnwrap(dst.ifd0[330])
        XCTAssertEqual(subIFDs.count, 2, "SubIFDs must hold original + alternate")
        XCTAssertEqual(subIFDs.type, 4, "SubIFDs is typed LONG")
        let offsets = subIFDs.uint32Array
        XCTAssertEqual(offsets.count, 2)
        let originalIFD = try TIFFReader.readIFD(data, at: Int(offsets[0]))
        let alternateIFD = try TIFFReader.readIFD(data, at: Int(offsets[1]))
        for (name, ifd) in [("original", originalIFD), ("alternate", alternateIFD)] {
            XCTAssertEqual(ifd[262]?.firstUInt32, 32803, "\(name) must be a CFA raw")
            XCTAssertEqual(ifd[254]?.firstUInt32, 0, "\(name) must be subfiletype 0, or ImageIO skips it")
            XCTAssertEqual(ifd[256]?.firstUInt32, UInt32(width))
            XCTAssertEqual(ifd[257]?.firstUInt32, UInt32(height))
        }

        // (c) `SubIFDs[0]` decodes to the original, `SubIFDs[1]` to the alternate. This is the
        // assertion that fails if the two are swapped — a file that would look perfectly legal.
        let first = try Self.decode(ifd: originalIFD, data: data)
        XCTAssertEqual(first.width, width)
        XCTAssertEqual(first.samples, primary, "SubIFDs[0] must decode to the original samples")

        let second = try Self.decode(ifd: alternateIFD, data: data)
        XCTAssertEqual(second.samples, alternate, "SubIFDs[1] must decode to the alternate samples")
        XCTAssertNotEqual(first.samples, second.samples)
        XCTAssertTrue(Set(originalIFD[324]?.uint32Array ?? []).isDisjoint(
                        with: Set(alternateIFD[324]?.uint32Array ?? [])),
                      "the two layers must not share tile payloads")

        // (d) The alternate repeats only what describes its own samples (so a promoted alternate
        // renders at the right size and level) and none of the develop-look data (a second copy
        // makes ImageIO apply the profile twice, which changes the whole file's rendering).
        let label = String(decoding: (alternateIFD[270]?.value ?? []).dropLast(), as: UTF8.self)
        XCTAssertEqual(label, "denoised alternate: test")
        let notRepeated: Set<UInt16> = [330, 34665, 34853, 700, 50740,
                                        50721, 50722, 50728, 50938, 50939, 50981, 50982,
                                        51022, 51041]
        XCTAssertTrue(Set(alternateIFD.keys).isDisjoint(with: notRepeated),
                      "the alternate must not repeat develop-look data: "
                      + "\(Set(alternateIFD.keys).intersection(notRepeated))")
        // Without these, a promoted alternate renders at the sensor size with default levels.
        for tag: UInt16 in [33421, 33422, 50713, 50714, 50717, 50719, 50720, 50829] {
            XCTAssertNotNil(alternateIFD[tag], "the alternate must describe its own samples (tag \(tag))")
        }
        XCTAssertNotNil(originalIFD[50829], "the original keeps ActiveArea")
        XCTAssertNotNil(originalIFD[33422], "the original keeps CFAPattern")

        // (e) The tool's own reader agrees with ImageIO's rule: `raw` is the first SubIFD.
        let readBack = try dst.readMosaic(data: data)
        XCTAssertEqual(readBack.samples, primary, "SourceDNG must resolve SubIFDs[0] as the raw")
    }

    /// The old entry point must keep producing the old shape — IFD0 is the raw, IFD1 the preview,
    /// and **no** `330`. The host and `scripts/acceptance.sh` read that file, and a stray SubIFDs
    /// tag would make some readers second-guess which page is the raw.
    func testSingleLayerWriteKeepsTheOldShape() throws {
        guard let fixture = Self.fixtureURL else { throw XCTSkip("fixture not present") }
        let dir = try scratch("single")
        let out = dir.appendingPathComponent("single.dng")

        let sourceData = try Data(contentsOf: fixture)
        let source = try SourceDNG(url: fixture)
        let (samples, width, height) = try source.readMosaic(data: sourceData)
        let result = try DNGRawWriter.write(samples: samples, width: width, height: height,
                                            from: fixture, to: out)

        XCTAssertFalse(result.hasAlternate)
        XCTAssertEqual(result.alternateByteCount, 0)
        XCTAssertEqual(result.alternateTileCount, 0)

        let data = try Data(contentsOf: out)
        let dst = try SourceDNG(data: data)
        XCTAssertNil(dst.ifd0[330], "a single-raw file must not carry SubIFDs")
        XCTAssertEqual(dst.ifd0[262]?.firstUInt32, 32803, "IFD0 is still the raw")
        XCTAssertEqual(dst.ifd0[254]?.firstUInt32, 0)
        let next = try XCTUnwrap(TIFFReader.nextIFDOffset(data, after: 8))
        XCTAssertNotEqual(next, 0, "IFD0 still links to the preview")
        let ifd1 = try TIFFReader.readIFD(data, at: Int(next))
        XCTAssertNotEqual(ifd1[262]?.firstUInt32, 32803, "IFD1 is still the preview")
        XCTAssertNotNil(ifd1[273])
        XCTAssertEqual(try dst.readMosaic(data: data).samples, samples)
    }
}
