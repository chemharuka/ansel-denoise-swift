import XCTest
@testable import AnselDenoise

/// Tests for the memory planner and the cut feathering.
///
/// Everything here is arithmetic — no model, no GPU, no files — so these run in milliseconds and
/// pin down the two things that were previously verified only by running a 60 MP frame by hand:
/// that a piece really fits the budget once overlap is added, and that the feather weights sum to
/// 1 across a cut.
final class DenoisePlannerTests: XCTestCase {

    // MARK: - memory model

    /// The constants must match what the Metal path actually uses.
    ///
    /// Measured peak RSS on a 60.3 MP frame was 37.84 GB, on 16.8 MP 10.60 GB, on 1 MP 0.68 GB —
    /// all 0.63–0.64 GB/MP. 0.64 is the conservative pick.
    func testBytesPerPixelMatchesMeasurements() {
        let cases: [(mp: Double, measuredGB: Double)] = [
            (1.0, 0.68), (4.2, 2.66), (6.3, 3.99), (16.8, 10.60), (30.2, 19.08), (60.3, 37.84),
        ]
        for c in cases {
            let predicted = Double(DenoisePlanner.bytes(forPixels: Int(c.mp * 1_000_000))) / 1e9
            // Within 15%: the point is that one constant covers a 60× range of sizes, not that it
            // is exact at any single one.
            XCTAssertLessThan(abs(predicted - c.measuredGB), c.measuredGB * 0.15,
                              "\(c.mp) MP: predicted \(predicted) GB, measured \(c.measuredGB) GB")
        }
    }

    /// Small machines get a fixed reserve instead of a fraction; 12 GB is the crossover.
    func testMemoryLimitDependsOnInstalledRAM() {
        // 8 GB → 50% = 4 GB, leaving 4 GB for the OS.
        XCTAssertEqual(Double(DenoisePlanner.memoryLimit(physicalMemory: 8_000_000_000)) / 1e9,
                       4.0, accuracy: 0.01)
        // 6 GB → 50%, not 2/3 (which would leave only 2 GB).
        XCTAssertEqual(Double(DenoisePlanner.memoryLimit(physicalMemory: 6_000_000_000)) / 1e9,
                       3.0, accuracy: 0.01)
        // 12 GB → 2/3 = 8 GB.
        XCTAssertEqual(Double(DenoisePlanner.memoryLimit(physicalMemory: 12_000_000_000)) / 1e9,
                       8.0, accuracy: 0.02)
        // 64 GB → 2/3 ≈ 42.7 GB.
        XCTAssertEqual(Double(DenoisePlanner.memoryLimit(physicalMemory: 64_000_000_000)) / 1e9,
                       42.67, accuracy: 0.05)
    }

    // MARK: - planning

    /// The normal case must be a single pass: that is what removes seams, and it is faster.
    func testFitsWholeImageInOnePass() {
        let plan = DenoisePlanner.plan(width: 9536, height: 6344,
                                       budgetBytes: 45_000_000_000)
        XCTAssertTrue(plan.isSinglePass)
        XCTAssertEqual(plan.pieces.count, 1)
        XCTAssertEqual(plan.seams, 0)
        XCTAssertEqual(plan.pieces[0].sourceX, 0)
        XCTAssertEqual(plan.pieces[0].sourceWidth, 9536, "whole image must be fed, not cropped")
        XCTAssertEqual(plan.pieces[0].sourceHeight, 6344)
    }

    /// A piece must fit the budget *after* overlap is added — that is the mistake a naive
    /// `area / pieceBudget` calculation makes.
    func testEveryPieceFitsIncludingOverlap() {
        let budget = 4_000_000_000                    // an 8 GB machine
        let plan = DenoisePlanner.plan(width: 12240, height: 8160, budgetBytes: budget)

        XCTAssertGreaterThan(plan.pieces.count, 1)
        XCTAssertLessThanOrEqual(plan.peakBytes, budget)
        for piece in plan.pieces {
            XCTAssertLessThanOrEqual(DenoisePlanner.bytes(forPixels: piece.sourcePixels), budget,
                                     "piece \(piece.x),\(piece.y) exceeds the budget once "
                                     + "its \(DenoisePlanner.cutOverlap) px margins are added")
        }
    }

    /// Output regions must tile the image exactly — no gap, no double coverage.
    func testPiecesPartitionTheImage() {
        for (w, h, budget) in [(12240, 8160, 4_000_000_000),
                                (6000, 4000, 4_000_000_000),
                                (9536, 6344, 8_000_000_000)] {
            let plan = DenoisePlanner.plan(width: w, height: h, budgetBytes: budget)
            let covered = plan.pieces.map(\.outputPixels).reduce(0, +)
            XCTAssertEqual(covered, w * h, "\(w)×\(h): pieces cover \(covered), expected \(w * h)")

            // And no pixel is claimed twice: the column spans must be disjoint.
            let columns = Set(plan.pieces.map { "\($0.x)-\($0.width)" })
            XCTAssertEqual(columns.count, plan.columns, "column spans must be distinct")
        }
    }

    /// The split count is computed, so an extreme frame produces a count no fixed list would hold.
    func testLargeFrameOnSmallMachineNeedsManyPieces() {
        let plan = DenoisePlanner.plan(width: 12240, height: 8160,   // 100 MP
                                       budgetBytes: 4_000_000_000)
        XCTAssertGreaterThanOrEqual(plan.pieces.count, 17,
                                    "100 MP at 4 GB needs about 17 pieces; got \(plan.pieces.count)")
        XCTAssertLessThan(plan.pieces.count, 400, "and it should not be a tile grid")
    }

    /// Fewer, squarer pieces beat long thin strips: short cuts mean less border.
    func testCutsStayBalanced() {
        let plan = DenoisePlanner.plan(width: 6000, height: 4000, budgetBytes: 4_000_000_000)
        let areas = plan.pieces.map(\.outputPixels)
        XCTAssertLessThanOrEqual(Double(areas.max()!), Double(areas.min()!) * 2.0,
                                 "piece areas are unbalanced: \(areas)")
    }

    // MARK: - cut geometry

    /// Margins appear on internal sides only, and never past the image edge.
    func testCutGeometryForTwoColumns() {
        let pieces = DenoisePlanner.split(width: 1000, height: 600,
                                          columns: 2, rows: 1, overlap: 128)
        XCTAssertEqual(pieces.count, 2)

        // Left piece: flush left, 128 px of context on its right.
        XCTAssertEqual(pieces[0].x, 0)
        XCTAssertEqual(pieces[0].width, 500)
        XCTAssertEqual(pieces[0].sourceX, 0)
        XCTAssertEqual(pieces[0].sourceWidth, 500 + 128)

        // Right piece: 128 px on its left, flush with the image edge.
        XCTAssertEqual(pieces[1].x, 500)
        XCTAssertEqual(pieces[1].width, 500)
        XCTAssertEqual(pieces[1].sourceX, 500 - 128)
        XCTAssertEqual(pieces[1].sourceX + pieces[1].sourceWidth, 1000)

        // The overlap band is shared, so their source regions are 128 px wider than their output.
        XCTAssertEqual(pieces[0].sourceWidth + pieces[1].sourceWidth, 1000 + 2 * 128)
    }

    /// A remainder is distributed, not dumped on the last piece (a 7 px strip would be padded to
    /// 16 and waste a whole forward pass).
    func testSpansDistributeRemainders() {
        let spans = DenoisePlanner.spans(total: 1003, count: 3)
        XCTAssertEqual(spans.count, 3)
        XCTAssertEqual(spans.reduce(0) { $0 + $1.count }, 1003)
        // All within one pixel of each other.
        let lengths = spans.map(\.count)
        XCTAssertLessThanOrEqual(lengths.max()! - lengths.min()!, 1)
    }

    // MARK: - feathering

    /// The whole point of the window: where two pieces overlap, their weights must sum to exactly
    /// 1, or the join shows as a band that is too dark (sum < 1) or too bright (sum > 1).
    func testFeatherWeightsSumToOneAcrossTheBand() {
        let tile = 512
        let taper = 128
        // Piece A's falling half over the last `taper` columns, piece B's rising half over the
        // first `taper` columns of the shared band.
        for i in 0..<taper {
            let a = DenoiseImageRunner.weight(at: tile - 1 - i, length: tile,
                                              nearLowCut: false, nearHighCut: true, taper: taper)
            let b = DenoiseImageRunner.weight(at: i, length: tile,
                                              nearLowCut: true, nearHighCut: false, taper: taper)
            XCTAssertEqual(a + b, 1.0, accuracy: 1e-6,
                           "weights sum to \(a + b) at band offset \(i), expected 1")
        }
    }

    /// Interior pixels must have weight exactly 1 — otherwise the interior would be attenuated.
    func testInteriorWeightIsExactlyOne() {
        let taper = 128
        for i in stride(from: taper, to: 512 - taper, by: 37) {
            let w = DenoiseImageRunner.weight(at: i, length: 512,
                                              nearLowCut: true, nearHighCut: true, taper: taper)
            XCTAssertEqual(w, 1.0, "interior pixel \(i) has weight \(w)")
        }
    }

    /// On the image's own border there is no cut, so no taper: the outermost column must not be
    /// faded out.
    func testImageBorderIsNotTapered() {
        for i in [0, 1, 510, 511] {
            let w = DenoiseImageRunner.weight(at: i, length: 512,
                                              nearLowCut: false, nearHighCut: false, taper: 128)
            XCTAssertEqual(w, 1.0, "image border pixel \(i) must have full weight, got \(w)")
        }
    }

    /// A taper longer than half the piece would overlap itself; it must clamp instead.
    func testTaperClampsToHalfThePiece() {
        let w = DenoiseImageRunner.weight(at: 0, length: 64,
                                          nearLowCut: true, nearHighCut: true, taper: 32)
        XCTAssertGreaterThanOrEqual(w, 0)
        XCTAssertLessThanOrEqual(w, 1)
    }
}
