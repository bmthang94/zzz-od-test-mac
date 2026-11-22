"""MacPcGameWindow 激活/前台状态判断的单元测试(独立仓)。

用 fake 的 AppKit/AX 接口隔离真实 macOS 窗口系统，验证:
- is_win_active 在 GUI 前台时为 False(不再只看应用级 isActive 兜底)
- 应用前台但焦点窗口不是游戏窗口时为 False
- AX 不可用时退回应用级判断
- active() 先激活应用再 AXRaise 目标窗口;应用激活返回 True 但 frontmost
  未切换时(焦点窃取防护)会走 activate() 与 osascript 兜底
"""

import subprocess
from unittest.mock import MagicMock

import pytest

from one_dragon.base.controller import mac_game_window as mgw
from one_dragon.base.controller.mac_game_window import (
    MacGameWindowHandle,
    MacPcGameWindow,
)
from one_dragon.base.geometry.rectangle import Rect

GAME_PID = 555
GUI_PID = 999
GAME_WINDOW_ID = 100
GAME_BOUNDS = {'X': 100.0, 'Y': 50.0, 'Width': 1920.0, 'Height': 1040.0}
# AXPosition/AXSize 与 CGWindow 外框一致(原点左上)
AX_GAME_POS = {'x': 100.0, 'y': 50.0}
AX_GAME_SIZE = {'width': 1920.0, 'height': 1040.0}


class FakeAXValue:
    """AXValueGetValue 的返回值载体。"""

    def __init__(self, kind: str, value: object) -> None:
        self.kind = kind
        self.value = value


class FakePoint:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class FakeSize:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height


class FakeAS:
    """ApplicationServices 的最小 fake:按元素字典分发属性。"""

    kAXValueCGPointType = 'cgpoint'
    kAXValueCGSizeType = 'cgsize'

    def __init__(self) -> None:
        self.app_elements: dict[int, dict] = {}
        self.set_calls: list[tuple[dict, str, object]] = []
        self.raise_calls: list[tuple[dict, str]] = []
        self.raise_error = 0

    def AXUIElementCreateApplication(self, pid: int):
        return self.app_elements.get(pid)

    def AXUIElementCopyAttributeValue(self, element, name: str, _ref):
        if not isinstance(element, dict) or name not in element:
            return -1, None
        return 0, element[name]

    def AXValueGetValue(self, value: FakeAXValue, kind: str, _ref):
        if value is None or value.kind != kind:
            return False, None
        return True, value.value

    def AXUIElementSetAttributeValue(self, element: dict, name: str, value: object) -> int:
        self.set_calls.append((element, name, value))
        element[name] = value
        return 0

    def AXUIElementPerformAction(self, element: dict, action: str) -> int:
        self.raise_calls.append((element, action))
        return self.raise_error


class FakeRunningApp:
    """NSRunningApplication 的最小 fake。

    frontmost_box 为 {'app': ...} 字典;steals_frontmost=True 时激活成功会
    把自己写进 box,模拟真实激活切换 frontmost 的效果。
    """

    def __init__(self, pid: int, *, active: bool = True,
                 activate_result: bool = True,
                 fallback_result: bool = True,
                 has_activate: bool = True,
                 frontmost_box: dict | None = None,
                 steals_frontmost: bool = False) -> None:
        self.pid = pid
        self._active = active
        self._activate_result = activate_result
        self._fallback_result = fallback_result
        self._has_activate = has_activate
        self._frontmost_box = frontmost_box
        self._steals_frontmost = steals_frontmost
        self.hidden = False
        self.unhide_calls = 0
        self.activate_options_calls: list[object] = []
        self.activate_calls = 0

    def processIdentifier(self) -> int:
        return self.pid

    def isActive(self) -> bool:
        return self._active

    def isHidden(self) -> bool:
        return self.hidden

    def unhide(self) -> None:
        self.hidden = False
        self.unhide_calls += 1

    def _maybe_steal_frontmost(self) -> None:
        if self._steals_frontmost and self._frontmost_box is not None:
            self._frontmost_box['app'] = self

    def activateWithOptions_(self, options) -> bool:
        self.activate_options_calls.append(options)
        if self._activate_result:
            self._maybe_steal_frontmost()
        return self._activate_result

    def activate(self) -> bool:
        self.activate_calls += 1
        if self._fallback_result:
            self._maybe_steal_frontmost()
        return self._fallback_result


class FakeWorkspace:
    """NSWorkspace.sharedWorkspace() 的最小 fake,frontmost 存于可变 box。"""

    def __init__(self, frontmost_box: dict) -> None:
        self._frontmost_box = frontmost_box

    def frontmostApplication(self):
        return self._frontmost_box.get('app')


def _frontmost_box(app: FakeRunningApp | None = None) -> dict:
    return {'app': app if app is not None else FakeRunningApp(GUI_PID, active=True)}


def _patch_ns_running_app(monkeypatch, apps: dict[int, FakeRunningApp]) -> None:
    """让 NSRunningApplication.runningApplicationWithProcessIdentifier_ 按 pid 查找。"""
    fake_ns_app = MagicMock()
    fake_ns_app.runningApplicationWithProcessIdentifier_ = (
        lambda pid: apps.get(pid)
    )
    monkeypatch.setattr(mgw, 'NSRunningApplication', fake_ns_app)


def _patch_frontmost(monkeypatch, frontmost_box: dict) -> None:
    fake_workspace = MagicMock()
    fake_workspace.sharedWorkspace.return_value = FakeWorkspace(frontmost_box)
    monkeypatch.setattr(mgw, 'NSWorkspace', fake_workspace)


@pytest.fixture
def fake_as(monkeypatch) -> FakeAS:
    fake = FakeAS()
    monkeypatch.setattr(mgw, 'AS', fake)
    return fake


def _make_window_info() -> dict:
    """CGWindowList 中的游戏窗口信息,键为真实 Quartz 常量。"""
    return {
        mgw.Quartz.kCGWindowNumber: GAME_WINDOW_ID,
        mgw.Quartz.kCGWindowOwnerPID: GAME_PID,
        mgw.Quartz.kCGWindowLayer: 0,
        mgw.Quartz.kCGWindowName: 'ZenlessZoneZero',
        mgw.Quartz.kCGWindowOwnerName: 'ZenlessZoneZero',
        mgw.Quartz.kCGWindowBounds: dict(GAME_BOUNDS),
    }


def _ax_window_element() -> dict:
    """AXWindows 中与 CGWindow 外框一致的窗口元素。"""
    return {
        'AXPosition': FakeAXValue('cgpoint', FakePoint(**AX_GAME_POS)),
        'AXSize': FakeAXValue('cgsize', FakeSize(**AX_GAME_SIZE)),
        'AXChildren': [],
    }


def _make_game_window(monkeypatch, window_info: dict | None = None) -> MacPcGameWindow:
    """构造已定位到游戏窗口的 MacPcGameWindow,隔离窗口枚举。"""
    info = _make_window_info() if window_info is None else window_info
    monkeypatch.setattr(
        MacPcGameWindow, '_get_windows', staticmethod(lambda: [info])
    )
    win = MacPcGameWindow()
    win._win = MacGameWindowHandle(GAME_WINDOW_ID, GAME_PID, 'ZenlessZoneZero')
    win._window_id = GAME_WINDOW_ID
    return win


class TestIsWinActive:
    """is_win_active 的前台/焦点窗口判断。"""

    def test_false_when_gui_is_frontmost(self, monkeypatch, fake_as):
        """GUI 进程 frontmost 时即使应用对象 isActive 也为 False。"""
        game_app = FakeRunningApp(GAME_PID, active=False)
        _patch_ns_running_app(monkeypatch, {GAME_PID: game_app})
        _patch_frontmost(monkeypatch, _frontmost_box())
        win = _make_game_window(monkeypatch)
        assert win.is_win_active is False

    def test_true_when_game_frontmost_and_window_focused(
        self, monkeypatch, fake_as
    ):
        """应用 frontmost 且 AX 焦点窗口与游戏窗口一致时为 True。"""
        game_app = FakeRunningApp(GAME_PID, active=True)
        _patch_ns_running_app(monkeypatch, {GAME_PID: game_app})
        _patch_frontmost(monkeypatch, _frontmost_box(FakeRunningApp(GAME_PID)))
        fake_as.app_elements[GAME_PID] = {
            'AXFocusedWindow': _ax_window_element(),
            'AXWindows': [_ax_window_element()],
        }
        win = _make_game_window(monkeypatch)
        assert win.is_win_active is True

    def test_false_when_other_window_of_same_app_is_focused(
        self, monkeypatch, fake_as
    ):
        """PlayCover 前台但焦点在别的窗口时为 False,触发窗口级激活。"""
        game_app = FakeRunningApp(GAME_PID, active=True)
        _patch_ns_running_app(monkeypatch, {GAME_PID: game_app})
        _patch_frontmost(monkeypatch, _frontmost_box(FakeRunningApp(GAME_PID)))
        other_window = {
            'AXPosition': FakeAXValue('cgpoint', FakePoint(0.0, 0.0)),
            'AXSize': FakeAXValue('cgsize', FakeSize(800.0, 600.0)),
        }
        fake_as.app_elements[GAME_PID] = {
            'AXFocusedWindow': other_window,
            'AXWindows': [_ax_window_element()],
        }
        win = _make_game_window(monkeypatch)
        assert win.is_win_active is False

    def test_falls_back_to_app_level_when_ax_unavailable(
        self, monkeypatch, fake_as
    ):
        """AX 元素拿不到(无辅助功能权限)时退回应用级判断。"""
        game_app = FakeRunningApp(GAME_PID, active=True)
        _patch_ns_running_app(monkeypatch, {GAME_PID: game_app})
        _patch_frontmost(monkeypatch, _frontmost_box(FakeRunningApp(GAME_PID)))
        # 不放 app_elements:AXUIElementCreateApplication 返回 None
        win = _make_game_window(monkeypatch)
        assert win.is_win_active is True

    def test_false_when_window_missing(self, monkeypatch, fake_as):
        """找不到游戏窗口时为 False。"""
        win = MacPcGameWindow()
        monkeypatch.setattr(MacPcGameWindow, '_get_windows', staticmethod(lambda: []))
        assert win.is_win_active is False


class TestActive:
    """active() 的应用激活验证 + AXRaise 窗口提起 + osascript 兜底。"""

    @pytest.fixture(autouse=True)
    def fake_subprocess(self, monkeypatch):
        """隔离 osascript 调用,默认返回失败;用例可改 return_value/side_effect。"""
        runner = MagicMock()
        runner.run.return_value = subprocess.CompletedProcess(
            [], 1, '', 'osascript failed'
        )
        monkeypatch.setattr(mgw.subprocess, 'run', runner.run)
        return runner.run

    def _setup_inactive(self, monkeypatch, fake_as, *,
                        activate_result: bool = True,
                        fallback_result: bool = True,
                        raise_error: int = 0,
                        steals_frontmost: bool = True
                        ) -> tuple[MacPcGameWindow, FakeRunningApp, dict]:
        """构造 GUI frontmost、游戏窗口存在但未激活的场景。"""
        box = _frontmost_box()
        game_app = FakeRunningApp(
            GAME_PID,
            active=False,
            activate_result=activate_result,
            fallback_result=fallback_result,
            frontmost_box=box,
            steals_frontmost=steals_frontmost,
        )
        _patch_ns_running_app(monkeypatch, {GAME_PID: game_app})
        _patch_frontmost(monkeypatch, box)
        ax_window = _ax_window_element()
        fake_as.app_elements[GAME_PID] = {
            'AXFocusedWindow': ax_window,
            'AXWindows': [ax_window],
        }
        fake_as.raise_error = raise_error
        return _make_game_window(monkeypatch), game_app, box

    def test_activates_app_and_raises_window(self, monkeypatch, fake_as, fake_subprocess):
        """正常路径:激活应用确认 frontmost + AXMain/AXFocused/AXRaise 目标窗口。"""
        win, game_app, _ = self._setup_inactive(monkeypatch, fake_as)
        assert win.active() is True
        assert len(game_app.activate_options_calls) == 1
        ax_window = fake_as.app_elements[GAME_PID]['AXWindows'][0]
        set_names = {name for _, name, _ in fake_as.set_calls}
        assert {'AXMain', 'AXFocused'} <= set_names
        assert fake_as.raise_calls == [(ax_window, 'AXRaise')]
        fake_subprocess.assert_not_called()

    def test_falls_back_to_osascript_when_frontmost_stays(
        self, monkeypatch, fake_as, fake_subprocess
    ):
        """activateWithOptions_/activate 返回 True 但 frontmost 未切换
        (焦点窃取防护)时,走 osascript System Events 兜底。"""
        win, game_app, box = self._setup_inactive(
            monkeypatch, fake_as, steals_frontmost=False
        )

        def osascript_success(cmd, **_kwargs):
            box['app'] = game_app  # 模拟 System Events 生效
            return subprocess.CompletedProcess(cmd, 0, '', '')

        fake_subprocess.side_effect = osascript_success
        assert win.active() is True
        fake_subprocess.assert_called_once()
        cmd = fake_subprocess.call_args.args[0]
        assert cmd[0] == '/usr/bin/osascript'
        assert str(GAME_PID) in cmd[2]

    def test_succeeds_when_only_window_raise_works(
        self, monkeypatch, fake_as, fake_subprocess
    ):
        """应用激活与 osascript 都失败但 AXRaise 成功时仍返回 True。"""
        win, _, _ = self._setup_inactive(
            monkeypatch, fake_as, activate_result=False, fallback_result=False
        )
        assert win.active() is True
        fake_subprocess.assert_called_once()

    def test_uses_activate_fallback_when_options_fail(
        self, monkeypatch, fake_as, fake_subprocess
    ):
        """activateWithOptions_ 失败时退回 activate()。"""
        win, game_app, _ = self._setup_inactive(
            monkeypatch, fake_as, activate_result=False, fallback_result=True
        )
        assert win.active() is True
        assert game_app.activate_calls == 1
        fake_subprocess.assert_not_called()

    def test_unhides_hidden_app_before_activating(
        self, monkeypatch, fake_as, fake_subprocess
    ):
        """应用被隐藏时先 unhide 再激活。"""
        win, game_app, _ = self._setup_inactive(monkeypatch, fake_as)
        game_app.hidden = True
        assert win.active() is True
        assert game_app.unhide_calls == 1

    def test_returns_false_and_logs_once_when_all_fail(
        self, monkeypatch, fake_as, fake_subprocess
    ):
        """应用激活、osascript 与 AXRaise 都失败时返回 False,错误只记一次。"""
        win, _, _ = self._setup_inactive(
            monkeypatch, fake_as,
            activate_result=False, fallback_result=False, raise_error=-1,
        )
        mock_log = MagicMock()
        monkeypatch.setattr(mgw, 'log', mock_log)
        assert win.active() is False
        assert win.active() is False
        error_calls = [c for c in mock_log.mock_calls if c[0] == 'error']
        assert len(error_calls) == 1

    def test_skips_activation_when_already_active(
        self, monkeypatch, fake_as, fake_subprocess
    ):
        """is_win_active 为 True 时不再执行激活,避免重复抢焦点。"""
        game_app = FakeRunningApp(GAME_PID, active=True)
        _patch_ns_running_app(monkeypatch, {GAME_PID: game_app})
        _patch_frontmost(monkeypatch, _frontmost_box(FakeRunningApp(GAME_PID)))
        ax_window = _ax_window_element()
        fake_as.app_elements[GAME_PID] = {
            'AXFocusedWindow': ax_window,
            'AXWindows': [ax_window],
        }
        win = _make_game_window(monkeypatch)
        assert win.active() is True
        assert game_app.activate_options_calls == []
        assert fake_as.raise_calls == []
        fake_subprocess.assert_not_called()

    def test_returns_false_when_window_missing(
        self, monkeypatch, fake_as, fake_subprocess
    ):
        """找不到游戏窗口时返回 False。"""
        win = MacPcGameWindow()
        monkeypatch.setattr(MacPcGameWindow, '_get_windows', staticmethod(lambda: []))
        assert win.active() is False


class TestWinRectRegression:
    """win_rect 重构后仍能通过 AXGroup 子元素返回内容区。"""

    def test_win_rect_returns_content_bounds(self, monkeypatch, fake_as):
        outer = Rect(100, 50, 2020, 1090)
        content = {
            'AXPosition': FakeAXValue(
                'cgpoint', FakePoint(outer.x1, outer.y1 + 28)
            ),
            'AXSize': FakeAXValue(
                'cgsize',
                FakeSize(outer.width, outer.height - 28),
            ),
        }
        ax_window = _ax_window_element()
        ax_window['AXChildren'] = [
            {'AXRole': 'AXStaticText', 'AXPosition': content['AXPosition']},
            {'AXRole': 'AXGroup', **content},
        ]
        fake_as.app_elements[GAME_PID] = {'AXWindows': [ax_window]}
        win = _make_game_window(monkeypatch)
        rect = win.win_rect
        assert rect is not None
        assert (rect.x1, rect.y1, rect.x2, rect.y2) == (
            outer.x1, outer.y1 + 28, outer.x2, outer.y2
        )

    def test_win_rect_none_when_ax_window_not_found(
        self, monkeypatch, fake_as
    ):
        fake_as.app_elements[GAME_PID] = {'AXWindows': []}
        win = _make_game_window(monkeypatch)
        mock_log = MagicMock()
        monkeypatch.setattr(mgw, 'log', mock_log)
        assert win.win_rect is None


class TestWindowRanking:
    """init_win 对同应用多窗口的排序:标题优先于应用名, 应用名兜底时选大窗口。

    复现 PlayCover 实机场景: 同 PID 的 layer-0 窗口里 66x20 'Window' 浮层
    排在真正的游戏窗口之前, 只按应用名取第一个会选错。
    """

    OVERLAY_WINDOW_ID = 200
    OVERLAY_BOUNDS = {'X': 0.0, 'Y': 0.0, 'Width': 66.0, 'Height': 20.0}

    def _overlay_window_info(self) -> dict:
        info = _make_window_info()
        info[mgw.Quartz.kCGWindowNumber] = self.OVERLAY_WINDOW_ID
        info[mgw.Quartz.kCGWindowName] = 'Window'
        info[mgw.Quartz.kCGWindowBounds] = dict(self.OVERLAY_BOUNDS)
        return info

    @pytest.mark.parametrize('game_title', ['ZenlessZoneZero', 'Zenless Zone Zero', ''])
    def test_overlay_window_does_not_win(self, monkeypatch, game_title: str) -> None:
        """浮层先于游戏窗口返回时, 无论游戏窗口标题是否为空都选中游戏窗口。"""
        overlay = self._overlay_window_info()
        game = _make_window_info()
        game[mgw.Quartz.kCGWindowName] = game_title
        monkeypatch.setattr(
            MacPcGameWindow, '_get_windows', staticmethod(lambda: [overlay, game])
        )
        win = MacPcGameWindow()
        win.update_win_title('ZenlessZoneZero')
        win.refresh_win()
        assert win.get_hwnd() == GAME_WINDOW_ID

    def test_owner_fallback_prefers_larger_window(self, monkeypatch) -> None:
        """标题都匹配不到时, 同名应用下选面积更大的窗口。"""
        small = self._overlay_window_info()
        small[mgw.Quartz.kCGWindowOwnerName] = 'MyGame'
        game = _make_window_info()
        game[mgw.Quartz.kCGWindowName] = ''
        game[mgw.Quartz.kCGWindowOwnerName] = 'MyGame'
        monkeypatch.setattr(
            MacPcGameWindow, '_get_windows', staticmethod(lambda: [small, game])
        )
        win = MacPcGameWindow()
        win.update_win_title('MyGame')  # 自定义标题: owner 兜底时按面积选大窗口
        win.refresh_win()
        assert win.get_hwnd() == GAME_WINDOW_ID

    def test_no_match_when_title_and_owner_differ(self, monkeypatch) -> None:
        """窗口标题与应用名都不在期望集合时, 不选任何窗口。"""
        overlay = self._overlay_window_info()
        overlay[mgw.Quartz.kCGWindowOwnerName] = 'PlayCover'
        monkeypatch.setattr(
            MacPcGameWindow, '_get_windows', staticmethod(lambda: [overlay])
        )
        win = MacPcGameWindow()
        win.update_win_title('ZenlessZoneZero')
        win.refresh_win()
        assert win.get_hwnd() is None
