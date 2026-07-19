import AppKit
import Carbon.HIToolbox

/// 사용자 지정 가능한 단축키 (keyCode + Carbon 모디파이어 마스크 + 표시용 라벨).
/// UserDefaults에 Codable로 저장.
public struct HotKeyBinding: Codable, Equatable, Sendable {
    public var keyCode: UInt32
    public var modifiers: UInt32
    public var label: String
    /// 바인딩이 설정돼 있는지 — keyCode만으로 판정하면 A키(kVK_ANSI_A == 0)가 "미설정"
    /// 센티널과 충돌해 조용히 등록에서 누락된다 (리뷰 확정). 해제는 keyCode·modifiers를
    /// 모두 0으로 두고, 녹화는 모디파이어를 강제하므로 modifiers가 진짜 판별자다.
    public var isSet: Bool { modifiers != 0 }
    public init(keyCode: UInt32, modifiers: UInt32, label: String) {
        self.keyCode = keyCode
        self.modifiers = modifiers
        self.label = label
    }
}

/// 전역 단축키 등록기 (Carbon RegisterEventHotKey).
///
/// NSEvent 전역 모니터와 달리 앱이 백그라운드일 때도, 소스 앱이 전체화면일 때도 동작한다.
/// 접근성 권한만 있으면 되고(이미 보유), 시스템 단축키를 가로채지 않는다.
/// Carbon 이벤트는 메인 스레드 런루프로 배달되므로 액션은 MainActor에서 실행된다.
@MainActor
public final class HotKeyCenter {
    public struct Binding {
        public let id: UInt32
        public let keyCode: UInt32
        public let modifiers: UInt32
        public let action: () -> Void
        public init(id: UInt32, keyCode: UInt32, modifiers: UInt32, action: @escaping () -> Void) {
            self.id = id
            self.keyCode = keyCode
            self.modifiers = modifiers
            self.action = action
        }
    }

    public static let shared = HotKeyCenter()

    private var handlerRef: EventHandlerRef?
    private var hotKeyRefs: [EventHotKeyRef?] = []
    private var actions: [UInt32: () -> Void] = [:]

    private init() {}

    /// 바인딩 등록 (기존 등록은 대체). 앱 시작 시 1회 호출.
    public func register(_ bindings: [Binding]) {
        unregisterAll()
        installHandler()
        // 'MFG1' — 앱 고유 시그니처
        let signature: OSType = 0x4D464731
        for b in bindings {
            var ref: EventHotKeyRef?
            let hkID = EventHotKeyID(signature: signature, id: b.id)
            let status = RegisterEventHotKey(b.keyCode, b.modifiers, hkID,
                                             GetApplicationEventTarget(), 0, &ref)
            if status == noErr {
                hotKeyRefs.append(ref)
                actions[b.id] = b.action
            }
        }
    }

    private func unregisterAll() {
        for ref in hotKeyRefs where ref != nil {
            UnregisterEventHotKey(ref)
        }
        hotKeyRefs.removeAll()
        actions.removeAll()
    }

    private func fire(_ id: UInt32) {
        actions[id]?()
    }

    private func installHandler() {
        guard handlerRef == nil else { return }
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard),
                                 eventKind: UInt32(kEventHotKeyPressed))
        let selfPtr = Unmanaged.passUnretained(self).toOpaque()
        InstallEventHandler(GetApplicationEventTarget(), { _, event, userData in
            guard let event, let userData else { return noErr }
            var hkID = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject),
                              EventParamType(typeEventHotKeyID), nil,
                              MemoryLayout<EventHotKeyID>.size, nil, &hkID)
            let center = Unmanaged<HotKeyCenter>.fromOpaque(userData).takeUnretainedValue()
            MainActor.assumeIsolated { center.fire(hkID.id) }
            return noErr
        }, 1, &spec, selfPtr, &handlerRef)
    }
}
