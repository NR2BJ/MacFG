import Foundation

/// 파일 기반 진단 로그. 캡처 세션의 진단 데이터를 파일에 기록.
/// `/tmp/MacFG_diag.log`에 누적 기록하며, 앱 시작 시 이전 로그 초기화.
///
/// 쓰기는 백그라운드 직렬 큐 + 영속 FileHandle — log()가 렌더 틱(메인스레드)에서 불리는데,
/// 호출마다 open→write→close 동기 I/O를 하면 디스크 상태에 따라 ~12ms 스파이크가 나서
/// vsync 콜백 2-3개를 삼킴 (tick=118Hz, gap=3(pre12.0) 실측 — 120 고정 실패 원인 중 하나).
public final class DiagnosticLog: @unchecked Sendable {
    public static let shared = DiagnosticLog()

    private let queue = DispatchQueue(label: "com.macfg.diaglog", qos: .utility)
    private let handle: FileHandle?
    private let dateFormatter: DateFormatter

    private init() {
        let fileURL = URL(fileURLWithPath: "/tmp/MacFG_diag.log")
        self.dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss.SSS"
        dateFormatter.timeZone = TimeZone.current

        // 앱 시작 시 로그 파일 초기화 후 핸들 유지 (호출마다 재오픈 금지)
        let header = "=== MacFG Diagnostic Log — \(Date()) ===\n"
        try? header.write(to: fileURL, atomically: true, encoding: .utf8)
        self.handle = try? FileHandle(forWritingTo: fileURL)
        _ = try? handle?.seekToEnd()
    }

    /// 진단 메시지를 파일에 기록 (비동기 — 호출 스레드 블로킹 없음, 직렬 큐가 순서 보존)
    public func log(_ message: String) {
        let timestamp = dateFormatter.string(from: Date())
        let line = "[\(timestamp)] \(message)\n"
        guard let data = line.data(using: .utf8) else { return }
        queue.async { [handle] in
            try? handle?.write(contentsOf: data)
        }
    }
}
