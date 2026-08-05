import Foundation

/// 진단·A/B용 노브 읽기. **환경변수가 우선, 없으면 UserDefaults `env.<이름>`.**
///
/// 왜 UserDefaults 폴백이 필요한가: 이 노브들은 전부 `MACFG_*` 환경변수로만 읽혔는데,
/// 환경변수를 주려면 셸에서 앱을 띄워야 한다. 그런데 macOS 26의 ControlCenter는 메뉴바
/// 항목 허용 목록을 앱이 아니라 **띄운 프로세스**(responsible process) 기준으로 기록해서,
/// 셸 실행이 반복되면 소유자 기록이 충돌해 아이콘이 영영 안 뜬다(2026-07-25 실측, 복구에
/// 하루). 즉 "A/B를 하려면 앱을 망가뜨려야 하는" 구조였고, 그래서 실제로는 A/B를 거의
/// 못 돌렸다. defaults 키로도 읽으면 Finder 실행 그대로 조건만 바꿀 수 있다.
///
///     defaults write com.macfg.MacFG env.MACFG_GOV_FORCE -string 3
///     defaults delete com.macfg.MacFG env.MACFG_GOV_FORCE
///
/// 환경변수를 먼저 보는 순서는 그대로 둔다 — 기존 스크립트·벤치가 그대로 동작해야 하고,
/// 저장된 값이 남아 있어도 그 실행만 덮어쓸 수 있어야 한다.
public enum Knob {
    public static func string(_ name: String) -> String? {
        if let v = ProcessInfo.processInfo.environment[name] { return v }
        // 부울 노브를 `defaults write ... -bool true`로 쓴 경우도 받아준다 —
        // string(forKey:)는 그때 nil을 돌려줘 "설정했는데 안 먹는" 함정이 된다.
        let key = "env.\(name)"
        if let s = UserDefaults.standard.string(forKey: key) { return s }
        guard UserDefaults.standard.object(forKey: key) != nil else { return nil }
        return UserDefaults.standard.bool(forKey: key) ? "1" : "0"
    }

    /// 존재 여부만 보는 노브용 (`env["X"] != nil` 자리).
    public static func isSet(_ name: String) -> Bool { string(name) != nil }

    public static func double(_ name: String) -> Double? { string(name).flatMap(Double.init) }
    public static func int(_ name: String) -> Int? { string(name).flatMap(Int.init) }
}
