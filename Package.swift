// swift-tools-version: 5.9
import PackageDescription

// GPLv3 component, vendored beside Photons. It is built as its own executable and
// invoked over a pipe, never linked into the MIT-licensed app — see LICENSE-SCOPE.md.
let package = Package(
    name: "AnselDenoise",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AnselDenoise", targets: ["AnselDenoise"]),
        .executable(name: "ansel-denoise-swift", targets: ["AnselDenoiseCLI"]),
    ],
    targets: [
        .target(name: "AnselDenoise"),
        .executableTarget(name: "AnselDenoiseCLI", dependencies: ["AnselDenoise"]),
        .testTarget(name: "AnselDenoiseTests", dependencies: ["AnselDenoise"]),
    ]
)
