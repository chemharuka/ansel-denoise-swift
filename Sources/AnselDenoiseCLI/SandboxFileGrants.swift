// GPL-3.0-or-later. Foundation file-grant transport for the independent CLI.
//
// The helper is a separate process, so it does not inherit the host app's sandbox
// extensions: a file the user authorised in an open panel is readable by the app and
// not by us. The host passes security-scoped bookmarks instead of copying the bytes —
// a 200 MB RAW stays where it is.
import Foundation

final class SandboxFileGrants {
    /// Visible so a test can assert that a grant was actually taken out. Internal, not public:
    /// nothing outside this executable has any business reading it.
    private(set) var active: [URL] = []

    /// `manifest` is a JSON array of base64 security-scoped bookmarks.
    init(manifestPath: String?) throws {
        guard let manifestPath else { return }
        let manifest = URL(fileURLWithPath: manifestPath)
        let values = try JSONDecoder().decode([String].self, from: Data(contentsOf: manifest))
        for value in values {
            guard let bookmark = Data(base64Encoded: value) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            var stale = false
            // Ordinary bookmarks passed by the parent carry its dynamic file grants. They are
            // deliberately not app-scoped bookmarks tied to the main executable's identity.
            let url = try URL(resolvingBookmarkData: bookmark, options: [.withoutUI, .withoutMounting],
                              relativeTo: nil, bookmarkDataIsStale: &stale)
            if url.startAccessingSecurityScopedResource() { active.append(url) }
        }
    }

    deinit { for url in active { url.stopAccessingSecurityScopedResource() } }
}
