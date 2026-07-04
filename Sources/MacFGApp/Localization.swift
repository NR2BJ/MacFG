import Foundation

/// 경량 in-code 로컬라이제이션 — en(기본)/ko/ja.
/// SwiftUI Text는 String 변수를 verbatim으로 렌더하므로 L()이 반환한 문자열이 그대로 표시된다.
/// 수동 .app 빌드(SPM 리소스 번들 미복사)에서도 dev 바이너리와 동일하게 동작.
enum AppLanguage {
    case en, ko, ja

    /// 시스템 선호 언어에서 1회 결정
    static let current: AppLanguage = {
        let pref = (Locale.preferredLanguages.first ?? "en").lowercased()
        if pref.hasPrefix("ko") { return .ko }
        if pref.hasPrefix("ja") { return .ja }
        return .en
    }()
}

/// 영어/한국어/일본어 중 현재 언어 문자열 반환
func L(_ en: String, _ ko: String, _ ja: String) -> String {
    switch AppLanguage.current {
    case .en: return en
    case .ko: return ko
    case .ja: return ja
    }
}
