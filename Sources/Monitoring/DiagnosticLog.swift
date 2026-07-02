import Foundation

/// 파일 기반 진단 로그. 캡처 세션의 진단 데이터를 파일에 기록.
/// `/tmp/MacFG_diag.log`에 누적 기록하며, 앱 시작 시 이전 로그 초기화.
public final class DiagnosticLog: @unchecked Sendable {
    public static let shared = DiagnosticLog()

    private let fileURL: URL
    private let lock = NSLock()
    private let dateFormatter: DateFormatter

    private init() {
        self.fileURL = URL(fileURLWithPath: "/tmp/MacFG_diag.log")
        self.dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss.SSS"
        dateFormatter.timeZone = TimeZone.current

        // 앱 시작 시 로그 파일 초기화
        let header = "=== MacFG Diagnostic Log — \(Date()) ===\n"
        try? header.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    /// 진단 메시지를 파일에 기록
    public func log(_ message: String) {
        let timestamp = dateFormatter.string(from: Date())
        let line = "[\(timestamp)] \(message)\n"

        lock.lock()
        defer { lock.unlock() }

        if let handle = try? FileHandle(forWritingTo: fileURL) {
            handle.seekToEndOfFile()
            if let data = line.data(using: .utf8) {
                handle.write(data)
            }
            handle.closeFile()
        }
    }
}
