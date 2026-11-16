import Foundation
import XCTest
@testable import AnselDenoiseCLI

/// The host app is sandboxed and this CLI is not: a child process does not inherit the parent's
/// sandbox extensions, so a file the user authorised in an open panel is readable by the host and
/// EPERM here. The host therefore passes a manifest of base64 security-scoped bookmarks instead of
/// copying the bytes — a 200 MB RAW stays where it is. These pin the read side of that contract.
final class SandboxFileGrantsTests: XCTestCase {

    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("sandbox-grants-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: scratch)
    }

    /// Write a manifest and hand back its path.
    private func writeManifest(_ values: [String]) throws -> String {
        let path = scratch.appendingPathComponent("grants-\(UUID().uuidString).json")
        try JSONEncoder().encode(values).write(to: path)
        return path.path
    }

    /// A real .json manifest for a real directory, byte for byte the shape the host writes.
    private func realBookmarkManifest() throws -> (path: String, directory: URL) {
        let directory = scratch.appendingPathComponent("picked")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let bookmark = try directory.bookmarkData(options: [], includingResourceValuesForKeys: nil,
                                                  relativeTo: nil)
        return (try writeManifest([bookmark.base64EncodedString()]), directory)
    }

    /// The ordinary command-line run. `make app`, every developer build and every host that has not
    /// been taught to send grants take this path, so it has to be completely inert: no throw, no
    /// access taken out, nothing to release.
    func testNoManifestMeansNoGrants() throws {
        let grants = try SandboxFileGrants(manifestPath: nil)
        XCTAssertTrue(grants.active.isEmpty, "nil manifest must not take out any grant")
    }

    func testAManifestOfRealBookmarksIsResolved() throws {
        let (path, directory) = try realBookmarkManifest()
        let grants = try SandboxFileGrants(manifestPath: path)
        XCTAssertEqual(grants.active.count, 1, "a real bookmark must be resolved and started")
        XCTAssertEqual(grants.active.first?.resolvingSymlinksInPath().path,
                       directory.resolvingSymlinksInPath().path,
                       "the resolved grant must point at the authorised directory")
    }

    func testACorruptBase64Throws() throws {
        let path = try writeManifest(["this is not base64 !!!"])
        XCTAssertThrowsError(try SandboxFileGrants(manifestPath: path)) { error in
            guard let cocoa = error as? CocoaError else {
                return XCTFail("expected CocoaError, got \(type(of: error)): \(error)")
            }
            XCTAssertEqual(cocoa.code, .fileReadCorruptFile)
        }
    }

    /// A manifest the host named but never wrote is a host bug. Swallowing it would turn a
    /// diagnosable mistake into EPERM on every input file, so it must surface here instead.
    func testAMissingManifestFileThrows() throws {
        let missing = scratch.appendingPathComponent("never-written.json").path
        XCTAssertThrowsError(try SandboxFileGrants(manifestPath: missing)) { error in
            let cocoa = error as? CocoaError
            XCTAssertEqual(cocoa?.code, .fileReadNoSuchFile,
                           "expected a missing-file error, got \(type(of: error)): \(error)")
        }
    }
}
