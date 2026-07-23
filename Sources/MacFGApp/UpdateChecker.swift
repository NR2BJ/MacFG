import Foundation
import Observation
import Monitoring

/// 릴리즈 채널 — 정식만 볼지, 베타(prerelease)까지 볼지.
public enum ReleaseChannel: String, CaseIterable, Identifiable, Sendable {
    case stable
    case beta

    public var id: String { rawValue }
}

/// GitHub 릴리즈에서 새 버전을 확인한다.
///
/// 설치는 하지 않고 **알리고 릴리즈 페이지를 열어줄 뿐**이다. 실행 중인 앱을 스스로 교체하는
/// 자동 설치는 코드사인/격리속성/권한 재확인이 얽혀 실패 시 앱이 깨질 수 있어, 사용자가
/// 직접 받아 교체하는 쪽이 안전하다.
@MainActor
@Observable
public final class UpdateChecker {
    public struct Release: Sendable {
        public let version: String       // 정규화된 버전 (v 접두어 제거)
        public let tag: String
        public let name: String
        public let url: String
        public let isPrerelease: Bool
        public let publishedAt: String
    }

    /// 확인 결과 — 새 버전이 있으면 non-nil
    public private(set) var available: Release?
    public private(set) var isChecking = false
    public private(set) var lastCheckedAt: Date?
    public private(set) var lastError: String?

    /// 사용자가 고른 채널 (영속)
    public var channel: ReleaseChannel {
        didSet {
            UserDefaults.standard.set(channel.rawValue, forKey: "s.updatechannel")
            // 채널이 바뀌면 이전 판정이 무의미 — 즉시 재확인
            available = nil
            Task { await check() }
        }
    }

    /// 자동 확인 여부 (영속, 기본 on)
    public var autoCheck: Bool {
        didSet { UserDefaults.standard.set(autoCheck, forKey: "s.updateauto") }
    }

    private let repo: String
    private let currentVersion: String
    private var timer: Timer?

    public init(repo: String = "NR2BJ/MacFG") {
        self.repo = repo
        self.currentVersion = Self.normalize(
            Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0")
        let saved = UserDefaults.standard.string(forKey: "s.updatechannel") ?? ReleaseChannel.stable.rawValue
        self.channel = ReleaseChannel(rawValue: saved) ?? .stable
        self.autoCheck = UserDefaults.standard.object(forKey: "s.updateauto") as? Bool ?? true
    }

    /// 앱 시작 시 1회 + 이후 6시간마다 (autoCheck일 때만)
    public func startPeriodicCheck() {
        guard autoCheck else { return }
        Task { await check() }
        timer?.invalidate()
        let t = Timer(timeInterval: 6 * 3600, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.autoCheck else { return }
                await self.check()
            }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    public func stopPeriodicCheck() {
        timer?.invalidate()
        timer = nil
    }

    public func check() async {
        guard !isChecking else { return }
        isChecking = true
        lastError = nil
        defer { isChecking = false; lastCheckedAt = Date() }

        guard let url = URL(string: "https://api.github.com/repos/\(repo)/releases?per_page=30") else { return }
        var req = URLRequest(url: url)
        req.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        req.setValue("MacFG", forHTTPHeaderField: "User-Agent")
        req.timeoutInterval = 15

        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            if let http = resp as? HTTPURLResponse, http.statusCode != 200 {
                lastError = "HTTP \(http.statusCode)"
                DiagnosticLog.shared.log("[UPDATE] 확인 실패: HTTP \(http.statusCode)")
                return
            }
            guard let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                lastError = "형식 오류"; return
            }
            let releases: [Release] = items.compactMap { item in
                guard let tag = item["tag_name"] as? String,
                      let url = item["html_url"] as? String,
                      (item["draft"] as? Bool) != true else { return nil }
                return Release(
                    version: Self.normalize(tag),
                    tag: tag,
                    name: (item["name"] as? String) ?? tag,
                    url: url,
                    isPrerelease: (item["prerelease"] as? Bool) ?? false,
                    publishedAt: (item["published_at"] as? String) ?? "")
            }
            // 채널 필터: stable은 정식만, beta는 둘 다 (베타 사용자도 정식 최신을 받아야 함)
            let pool = channel == .stable ? releases.filter { !$0.isPrerelease } : releases
            let newest = pool.max { Self.compare($0.version, $1.version) == .orderedAscending }
            if let newest, Self.compare(newest.version, currentVersion) == .orderedDescending {
                available = newest
                DiagnosticLog.shared.log("[UPDATE] 새 버전 \(newest.tag) (현재 \(currentVersion), 채널 \(channel.rawValue))")
            } else {
                available = nil
            }
        } catch {
            lastError = error.localizedDescription
            DiagnosticLog.shared.log("[UPDATE] 확인 실패: \(error.localizedDescription)")
        }
    }

    public var currentVersionString: String { currentVersion }

    /// "v1.1.5" / "1.1.5-beta.2" → 비교용 정규화 (앞의 v만 제거, 나머지는 보존)
    static func normalize(_ s: String) -> String {
        var v = s.trimmingCharacters(in: .whitespaces)
        if v.hasPrefix("v") || v.hasPrefix("V") { v.removeFirst() }
        return v
    }

    /// 시맨틱 버전 비교. 숫자 파트를 먼저 비교하고, 같으면 프리릴리즈 < 정식 (SemVer 규칙).
    /// 예: 1.1.6 > 1.1.6-beta.1 > 1.1.5
    static func compare(_ a: String, _ b: String) -> ComparisonResult {
        func split(_ s: String) -> (nums: [Int], pre: String) {
            let parts = s.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
            let nums = parts[0].split(separator: ".").map { Int($0.filter(\.isNumber)) ?? 0 }
            return (nums, parts.count > 1 ? String(parts[1]) : "")
        }
        let (an, ap) = split(a), (bn, bp) = split(b)
        for i in 0..<max(an.count, bn.count) {
            let x = i < an.count ? an[i] : 0
            let y = i < bn.count ? bn[i] : 0
            if x != y { return x < y ? .orderedAscending : .orderedDescending }
        }
        // 숫자가 같으면: 프리릴리즈가 있는 쪽이 낮다
        if ap.isEmpty && bp.isEmpty { return .orderedSame }
        if ap.isEmpty { return .orderedDescending }   // a=정식 > b=프리릴리즈
        if bp.isEmpty { return .orderedAscending }
        return ap.compare(bp, options: .numeric)
    }
}
