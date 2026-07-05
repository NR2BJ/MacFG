import Foundation

/// 경량 in-code 로컬라이제이션 — en(기본)/ko/ja.
/// SwiftUI Text는 String 변수를 verbatim으로 렌더하므로 L()이 반환한 문자열이 그대로 표시된다.
/// 수동 .app 빌드(SPM 리소스 번들 미복사)에서도 dev 바이너리와 동일하게 동작.
/// current는 설정에서 런타임 변경 가능 — 변경 시 AppState 관찰로 뷰가 재구성되며 L()이 재평가된다.
enum AppLanguage {
    case en, ko, ja

    /// 시스템 선호 언어에서 감지
    static func detectSystem() -> AppLanguage {
        let pref = (Locale.preferredLanguages.first ?? "en").lowercased()
        if pref.hasPrefix("ko") { return .ko }
        if pref.hasPrefix("ja") { return .ja }
        return .en
    }

    /// 현재 UI 언어 — 저장된 설정("s.lang") → 없으면("system") 시스템 감지
    nonisolated(unsafe) static var current: AppLanguage = fromRaw(UserDefaults.standard.string(forKey: "s.lang"))

    static func fromRaw(_ raw: String?) -> AppLanguage {
        switch raw {
        case "en": return .en
        case "ko": return .ko
        case "ja": return .ja
        default:   return detectSystem()   // "system" 또는 미설정
        }
    }

    /// raw("system"/"en"/"ko"/"ja") 적용 + 영속
    static func apply(raw: String) {
        UserDefaults.standard.set(raw, forKey: "s.lang")
        current = fromRaw(raw)
    }
}

/// 영어/한국어/일본어 중 현재 언어 문자열 반환
func L(_ en: String, _ ko: String, _ ja: String) -> String {
    switch AppLanguage.current {
    case .en: return en
    case .ko: return ko
    case .ja: return ja
    }
}
