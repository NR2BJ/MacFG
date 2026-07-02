import Metal
import os.lock

/// 트리플 버퍼링 락프리 스왑체인
///
/// 캡처(Writer) / 처리(Processor) / 출력(Reader) 3개 슬롯을 관리.
/// Writer는 항상 최신 프레임을 쓰고, Reader는 가장 최근 완성된 프레임을 읽는다.
public final class FrameBuffer: Sendable {
    private struct State: Sendable {
        var slots: [FrameSlot] = [.empty, .empty, .empty]
        var writeIndex: Int = 0
        var readIndex: Int = 2
        var lastWriteTimestamp: CFTimeInterval = 0
        var dropCount: Int = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    /// 프레임 드롭 감지 임계치 (예상 간격 대비 배수)
    public let dropThreshold: Double

    public init(dropThreshold: Double = 1.5) {
        self.dropThreshold = dropThreshold
    }

    /// 새 프레임 쓰기. 드롭 감지 시 true 반환.
    @discardableResult
    public func write(_ slot: FrameSlot) -> Bool {
        state.withLock { s in
            // Write → 다음 슬롯 (read가 아닌 슬롯)
            let nextWrite = nextAvailableSlot(current: s.writeIndex, avoid: s.readIndex)
            s.slots[nextWrite] = slot
            s.writeIndex = nextWrite
            s.lastWriteTimestamp = slot.timestamp
            s.dropCount += 1

            return false
        }
    }

    /// 가장 최근 쓰인 프레임 읽기
    public func read() -> FrameSlot {
        state.withLock { s in
            // 가장 최근 write된 슬롯을 read 슬롯으로 전환
            s.readIndex = s.writeIndex
            return s.slots[s.readIndex]
        }
    }

    /// 현재 프레임 드롭 수
    public var totalFrames: Int {
        state.withLock { $0.dropCount }
    }

    private func nextAvailableSlot(current: Int, avoid: Int) -> Int {
        for i in 1...2 {
            let candidate = (current + i) % 3
            if candidate != avoid {
                return candidate
            }
        }
        return (current + 1) % 3
    }
}
