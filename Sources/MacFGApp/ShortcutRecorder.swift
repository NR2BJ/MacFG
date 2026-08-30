import SwiftUI
import AppKit
import Carbon.HIToolbox

/// 녹화 앱 방식 단축키 입력 필드 — 클릭하면 다음 키 조합을 캡처해 바인딩.
/// Esc = 취소, Delete = 지우기. 모디파이어 없는 키는 거부(전역 훅 안전).
struct ShortcutRecorder: NSViewRepresentable {
    @Binding var binding: HotKeyBinding

    func makeNSView(context: Context) -> RecorderView {
        let v = RecorderView()
        v.binding = binding
        v.onChange = { binding = $0 }
        return v
    }

    func updateNSView(_ v: RecorderView, context: Context) {
        if !v.isRecording { v.binding = binding; v.needsDisplay = true }
    }
}

final class RecorderView: NSView {
    var binding = HotKeyBinding(keyCode: 0, modifiers: 0, label: "")
    var onChange: ((HotKeyBinding) -> Void)?
    private(set) var isRecording = false

    override var intrinsicContentSize: NSSize { NSSize(width: 96, height: 22) }
    override var acceptsFirstResponder: Bool { true }

    override func draw(_ dirtyRect: NSRect) {
        let path = NSBezierPath(roundedRect: bounds.insetBy(dx: 1, dy: 1), xRadius: 5, yRadius: 5)
        (isRecording ? NSColor.controlAccentColor.withAlphaComponent(0.22) : NSColor.controlBackgroundColor).setFill()
        path.fill()
        (isRecording ? NSColor.controlAccentColor : NSColor.separatorColor).setStroke()
        path.stroke()

        let text = isRecording ? L("Press keys…", "키 입력…", "キー入力…") : (binding.label.isEmpty ? L("Click to set", "클릭해 설정", "クリックで設定") : binding.label)
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 11, weight: isRecording ? .medium : .regular),
            .foregroundColor: isRecording ? NSColor.controlAccentColor
                : (binding.label.isEmpty ? NSColor.tertiaryLabelColor : NSColor.labelColor),
        ]
        let s = NSAttributedString(string: text, attributes: attrs)
        let sz = s.size()
        s.draw(at: NSPoint(x: (bounds.width - sz.width) / 2, y: (bounds.height - sz.height) / 2))
    }

    override func mouseDown(with event: NSEvent) {
        isRecording.toggle()
        if isRecording { window?.makeFirstResponder(self) }
        needsDisplay = true
    }

    override func keyDown(with event: NSEvent) {
        guard isRecording else { super.keyDown(with: event); return }
        // Esc = 취소
        if event.keyCode == UInt16(kVK_Escape) { stop(); return }
        // Delete = 지우기 (바인딩 제거)
        if event.keyCode == UInt16(kVK_Delete) {
            binding = HotKeyBinding(keyCode: 0, modifiers: 0, label: "")
            onChange?(binding); stop(); return
        }
        var mods: UInt32 = 0
        let f = event.modifierFlags
        if f.contains(.command) { mods |= UInt32(cmdKey) }
        if f.contains(.option)  { mods |= UInt32(optionKey) }
        if f.contains(.control) { mods |= UInt32(controlKey) }
        if f.contains(.shift)   { mods |= UInt32(shiftKey) }
        // **Shift 단독은 거부한다 (2026-08-31).**
        // Carbon의 RegisterEventHotKey는 ⇧A 같은 조합도 noErr로 받아준다(실측). 그러면 그 키가
        // **시스템 전역에서 삼켜져** 대문자 A가 안 쳐지고, 그 바인딩은 UserDefaults에 저장돼
        // 재실행해도 살아난다. 사용자는 'A가 안 쳐진다'와 이 앱을 연결짓기 어렵다.
        // 기본 바인딩 3개가 전부 ⌃⌥⌘이므로 잃는 것은 Shift 단독뿐이다.
        // (⌘Q·⌘Tab 같은 시스템 조합은 여기 도달조차 못 한다 — 메뉴/WindowServer가 먼저 가져간다.
        //  블랙리스트를 두지 않는 이유이고, 중복 조합 거부도 두지 않는다 —
        //  registerHotKeys의 등록 순서 규칙이 이미 "사용자 조합이 이긴다"로 설계돼 있다.)
        guard mods & UInt32(controlKey | optionKey | cmdKey) != 0 else { NSSound.beep(); return }

        // 문자/숫자 한 글자만 그대로 쓰고, 나머지는 keyName으로. 기존 조건은 && 우선순위 탓에
        // `key.isEmpty || (isLetter==false && count != 1)`로 묶여, 화살표(사설영역 문자 U+F700대)나
        // Space처럼 "한 글자지만 문자가 아닌" 키가 keyName을 못 타고 깨진 글자로 저장됐다 (리뷰 확정).
        let key = (event.charactersIgnoringModifiers ?? "").uppercased()
        let isPlainKey = key.count == 1 && (key.first!.isLetter || key.first!.isNumber)
        let label = Self.modifierSymbols(f) + (isPlainKey ? key : Self.keyName(event.keyCode))
        binding = HotKeyBinding(keyCode: UInt32(event.keyCode), modifiers: mods, label: label)
        onChange?(binding)
        stop()
    }

    override func resignFirstResponder() -> Bool { stop(); return true }
    private func stop() { isRecording = false; needsDisplay = true }

    static func modifierSymbols(_ f: NSEvent.ModifierFlags) -> String {
        var s = ""
        if f.contains(.control) { s += "⌃" }
        if f.contains(.option)  { s += "⌥" }
        if f.contains(.shift)   { s += "⇧" }
        if f.contains(.command) { s += "⌘" }
        return s
    }

    static func keyName(_ code: UInt16) -> String {
        switch Int(code) {
        case kVK_ANSI_Period: return "."
        case kVK_ANSI_Comma:  return ","
        case kVK_ANSI_Slash:  return "/"
        case kVK_Space:       return "Space"
        case kVK_Return:      return "↩"
        case kVK_Tab:         return "⇥"
        case kVK_LeftArrow:   return "←"
        case kVK_RightArrow:  return "→"
        case kVK_UpArrow:     return "↑"
        case kVK_DownArrow:   return "↓"
        default:              return "key\(code)"
        }
    }
}
