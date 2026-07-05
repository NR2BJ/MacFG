// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "MacFG",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "MacFGApp", targets: ["MacFGApp"]),
        .executable(name: "InterpBench", targets: ["InterpBench"]),
        .executable(name: "TestPattern", targets: ["TestPattern"]),
    ],
    targets: [
        // MARK: - Main App
        .executableTarget(
            name: "MacFGApp",
            dependencies: ["CaptureKit", "Overlay", "FramePacing", "Monitoring", "Interpolation"],
            path: "Sources/MacFGApp"
        ),

        // MARK: - Modules
        .target(
            name: "CaptureKit",
            dependencies: ["FramePacing"],
            path: "Sources/CaptureKit"
        ),
        .target(
            name: "Overlay",
            dependencies: ["FramePacing", "Monitoring"],
            path: "Sources/Overlay"
        ),
        .target(
            name: "FramePacing",
            path: "Sources/FramePacing"
        ),
        .target(
            name: "Interpolation",
            dependencies: ["FramePacing", "Monitoring"],
            path: "Sources/Interpolation"
        ),
        .target(
            name: "Monitoring",
            dependencies: ["FramePacing"],
            path: "Sources/Monitoring"
        ),
        // MARK: - Benchmark
        .executableTarget(
            name: "InterpBench",
            dependencies: ["Interpolation"],
            path: "Sources/InterpBench"
        ),
        // MARK: - Test Pattern (자체 검증용 60fps 소스 창)
        .executableTarget(
            name: "TestPattern",
            path: "Sources/TestPattern"
        ),
    ]
)
