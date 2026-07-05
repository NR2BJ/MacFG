import Foundation
import os

/// 파일 기반 진단 로그. 캡처 세션의 진단 데이터를 `/tmp/MacFG_diag.log`에 기록.
///
/// **개발자 모드 게이트 (기본 OFF)**: 설정 "s.devlog"(또는 env MACFG_DIAG, --auto-capture 등
/// 무인 테스트)일 때만 파일을 만들고 기록한다. OFF면 파일을 지우고 아무 것도 안 쓴다 —
/// 일반 사용자에겐 디스크에 진단 흔적이 남지 않는다.
///
/// 쓰기는 백그라운드 직렬 큐 + 영속 FileHandle — log()가 렌더 틱(메인스레드)에서 불리는데,
/// 호출마다 open→write→close 동기 I/O를 하면 ~12ms 스파이크로 vsync 콜백을 삼킨다.
public final class DiagnosticLog: @unchecked Sendable {
    public static let shared = DiagnosticLog()

    private let queue = DispatchQueue(label: "com.macfg.diaglog", qos: .utility)
    private var handle: FileHandle?          // queue에서만 접근
    private let enabledFlag = OSAllocatedUnfairLock(initialState: false)
    private let dateFormatter: DateFormatter
    private let fileURL = URL(fileURLWithPath: "/tmp/MacFG_diag.log")

    private init() {
        self.dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss.SSS"
        dateFormatter.timeZone = TimeZone.current

        // 무인 테스트(auto-capture/env)나 설정 저장값이면 시작부터 켬
        let testMode = ProcessInfo.processInfo.environment["MACFG_DIAG"] != nil
            || CommandLine.arguments.contains("--auto-capture-title")
        let on = testMode || UserDefaults.standard.bool(forKey: "s.devlog")
        enabledFlag.withLock { $0 = on }
        if on { queue.async { [weak self] in self?.openHandle() } }
    }

    /// queue에서만 호출 — 파일 초기화 + 핸들 오픈
    private func openHandle() {
        let header = "=== MacFG Diagnostic Log — \(Date()) ===\n"
        try? header.write(to: fileURL, atomically: true, encoding: .utf8)
        handle = try? FileHandle(forWritingTo: fileURL)
        _ = try? handle?.seekToEnd()
    }

    /// 개발자 모드 토글 — on이면 파일 생성·기록, off면 핸들 닫고 파일 삭제(기록 중단)
    public func setEnabled(_ on: Bool) {
        UserDefaults.standard.set(on, forKey: "s.devlog")
        enabledFlag.withLock { $0 = on }
        queue.async { [weak self] in
            guard let self else { return }
            if on {
                if self.handle == nil { self.openHandle() }
            } else {
                try? self.handle?.close()
                self.handle = nil
                try? FileManager.default.removeItem(at: self.fileURL)
            }
        }
    }

    public var isEnabled: Bool { enabledFlag.withLock { $0 } }

    /// 진단 메시지를 파일에 기록 (게이트 OFF면 무동작; 비동기 — 호출 스레드 블로킹 없음)
    public func log(_ message: String) {
        guard enabledFlag.withLock({ $0 }) else { return }
        let timestamp = dateFormatter.string(from: Date())
        let line = "[\(timestamp)] \(message)\n"
        guard let data = line.data(using: .utf8) else { return }
        queue.async { [weak self] in
            try? self?.handle?.write(contentsOf: data)
        }
    }
}
