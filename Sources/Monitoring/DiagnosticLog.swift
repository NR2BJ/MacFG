import Foundation
import CoreFoundation
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
    private var lastPathCheck: CFAbsoluteTime = 0   // 경로 존재 확인 스로틀 (queue에서만 접근)
    private let enabledFlag = OSAllocatedUnfairLock(initialState: false)
    private let dateFormatter: DateFormatter
    private let fileURL = URL(fileURLWithPath: "/tmp/MacFG_diag.log")

    private init() {
        self.dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss.SSS"
        dateFormatter.timeZone = TimeZone.current

        // 무인 테스트(auto-capture/env)나 설정 저장값이면 시작부터 켬
        // 자동 캡처는 CLI 인자와 **defaults 둘 다** 테스트 신호로 친다.
        // defaults 경로(s.autocapturetitle)는 나중에 추가됐는데 이 판정에 빠져 있어서,
        // 무인 테스트를 걸어놓고 "왜 로그가 안 남지"로 한 사이클을 날렸다(2026-08-07).
        let autoCapture = CommandLine.arguments.contains("--auto-capture-title")
            || !(UserDefaults.standard.string(forKey: "s.autocapturetitle") ?? "").isEmpty
        let testMode = Knob.string("MACFG_DIAG") != nil || autoCapture
        let on = testMode || UserDefaults.standard.bool(forKey: "s.devlog")
        enabledFlag.withLock { $0 = on }
        if on { queue.async { [weak self] in self?.openHandle() } }
    }

    /// 한 로그 파일이 커질 수 있는 상한. 넘으면 잘라내고 그 사실을 남긴다.
    /// 실측 기준 ~1.2KB/s(시간당 4MB)라 이 값은 며칠치다 — 디스크를 채우는 사고만 막는 안전장치.
    private let maxBytes: UInt64 = 256 * 1024 * 1024

    /// queue에서만 호출 — 핸들 오픈. **파일을 덮어쓰지 않는다.**
    ///
    /// 예전엔 실행할 때마다 `atomically: true`로 써서 파일을 잘라냈다. 그런데 이 앱의 A/B는
    /// "설정 바꾸고 재시작"이 기본 절차라, 앱을 다시 켜는 순간 **직전 조건의 데이터가 사라졌다** —
    /// 실제로 CAS on/off 비교에서 off 쪽 로그를 통째로 날릴 뻔했다(2026-08-07).
    /// 세션 경계는 앱 실행이 아니라 **개발자 모드 토글**이다: setEnabled(false)가 파일을 지우므로,
    /// 모드를 켜 두는 동안의 모든 실행이 한 파일에 누적된다.
    private func openHandle() {
        let fm = FileManager.default
        if !fm.fileExists(atPath: fileURL.path) {
            fm.createFile(atPath: fileURL.path, contents: nil)
        }
        // 상한을 넘었으면 여기서만 잘라낸다 (무한 증가 방지)
        if let attrs = try? fm.attributesOfItem(atPath: fileURL.path),
           let size = attrs[.size] as? UInt64, size > maxBytes {
            try? Data().write(to: fileURL)
            let note = "=== (이전 로그가 \(size / 1024 / 1024)MB를 넘어 잘라냈습니다) ===\n"
            try? note.write(to: fileURL, atomically: false, encoding: .utf8)
        }
        handle = try? FileHandle(forWritingTo: fileURL)
        _ = try? handle?.seekToEnd()
        // 실행 경계 표식 — 여러 실행이 한 파일에 쌓이므로 구간을 가를 수 있어야 한다.
        let header = "\n===== MacFG 실행 시작 \(Date()) (pid \(ProcessInfo.processInfo.processIdentifier)) =====\n"
        if let d = header.data(using: .utf8) { try? handle?.write(contentsOf: d) }
    }

    /// 개발자 모드 토글 — on이면 파일 생성·기록, off면 핸들 닫고 **파일 삭제**(기록 중단).
    ///
    /// **여기가 유일한 세션 경계다.** 앱 재시작은 파일을 지우지 않고 이어서 쓴다(openHandle 주석 참조).
    /// 즉 "켠 뒤 끄기 전까지"의 모든 실행이 한 파일에 누적되고, 끄는 순간 흔적이 사라진다.
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
            guard let self else { return }
            // **경로가 사라졌으면 다시 연다.** FileHandle은 inode를 붙들기 때문에, 파일이
            // 지워지거나 옮겨져도 쓰기는 계속 성공한다 — 다만 아무도 볼 수 없는 유령 파일에.
            // 실제로 그렇게 됐다: 무인 A/B 스크립트가 실행마다 로그를 rm 하는데 앱은 이미
            // 열린 핸들에 계속 써서, 275KB가 쌓였는데 /tmp의 파일은 23KB에서 멈춰 있었고
            // 그 옛 파일을 최신으로 오독했다(2026-08-26). 진단 도구가 조용히 거짓말하면
            // 그 위에 쌓은 모든 판단이 무효가 된다.
            // stat은 로그 줄마다가 아니라 1초에 한 번만 — 핫패스 비용을 만들지 않는다.
            let now = CFAbsoluteTimeGetCurrent()
            if now - self.lastPathCheck > 1.0 {
                self.lastPathCheck = now
                if !FileManager.default.fileExists(atPath: self.fileURL.path) {
                    try? self.handle?.close()
                    self.handle = nil
                    self.openHandle()
                }
            }
            try? self.handle?.write(contentsOf: data)
        }
    }
}
