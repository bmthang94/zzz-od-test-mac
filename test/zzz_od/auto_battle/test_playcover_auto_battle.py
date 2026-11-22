"""PlayCover 触控布局自动战斗回归测试。

验证:
1. ZScreenContext 按 is_playcover_touch_ui 在 按键-普通攻击 与 按键-普通攻击-PlayCover 之间分发。
2. PlayCover 实机画面通过代理人状态区归一化复用桌面头像/资源识别 (remielle/phoenix/velina)。
3. Windows 画面不受归一化影响 (yeshunguang/sunna/zhao)。
4. check_battle_state 在截图缺失 (None) 时保留缓存状态且不触发 CV 识别。
5. AreaPatrol.wait_battle_screen 对 None 截图重试, 拿到画面后成功。
6. macOS 截图连续失败告警只打一次, 成功后恢复。
7. Remielle 两个战斗配置在触控桩控制器上发出期望的触点序列。
8. abnormal_status_interval 只通过当前实例间隔生效, 不修改全局任务定义。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cv2.typing import MatLike
from test.conftest import MockController, TestContext

from one_dragon.base.conditional_operation.operation_executor import OperationExecutor
from one_dragon.base.geometry.point import Point
from one_dragon.base.geometry.rectangle import Rect
from one_dragon.base.matcher.template_matcher import TemplateMatcher
from one_dragon.base.operation.operation_round_result import OperationRoundResultEnum
from one_dragon.base.screen.template_loader import TemplateLoader
from one_dragon.utils import cv2_utils
from zzz_od.auto_battle.auto_battle_context import AutoBattleContext
from zzz_od.auto_battle.auto_battle_operator import AutoBattleOperator
from zzz_od.auto_battle.auto_battle_state import BattleStateEnum
from zzz_od.context.zzz_screen_context import ZScreenContext
from zzz_od.controller.zzz_pc_controller import ZPcController
from zzz_od.operation.compendium.area_patrol import AreaPatrol

_SCREEN_DIR = Path(__file__).parent.parent.parent.parent / 'screens' / '战斗画面'
_PLAYCOVER_FIXTURE_1 = 'playcover-实机-1'
_PLAYCOVER_FIXTURE_2 = 'playcover-实机-2'
_WINDOWS_FIXTURE = '默认'

_PLAYCOVER_ATTACK_RECT = Rect(1470, 715, 1670, 915)
_DESKTOP_ATTACK_RECT = Rect(1403, 916, 1505, 1017)


@pytest.fixture()
def isolated_battle_context(test_context: TestContext, monkeypatch: pytest.MonkeyPatch) -> TestContext:
    """复用已初始化的 session ctx, 但把本模块会改动的成员换成新的实例,
    防止 touch_ui_enabled / 状态记录器 / last_check_in_battle / auto_op / 检测间隔
    泄漏到测试仓其他用例。"""
    monkeypatch.setattr(test_context, 'controller', MockController(test_context.game_config))
    monkeypatch.setattr(test_context, 'auto_battle_context', AutoBattleContext(test_context))
    return test_context


def _load_fixture(name: str) -> MatLike:
    """从中央存档读 战斗画面 截图。"""
    path = _SCREEN_DIR / f'{name}.webp'
    assert path.exists(), f'存档截图不存在: {path}'
    return cv2_utils.read_image(str(path))


def _screen_context(playcover: bool) -> ZScreenContext:
    sc = ZScreenContext(lambda: playcover)
    sc.reload(from_separated_files=False)
    return sc


def _reset_agent_ctx(ctx: TestContext, playcover: bool) -> None:
    """按平台切换触控标记, 并重读识别区域(触发 PlayCover 别名分发)。"""
    ctx.controller.touch_ui_enabled = playcover
    abc = ctx.auto_battle_context
    abc.agent_context._check_agent_interval = 0
    abc.init_screen_area()


def _touch_stub_controller() -> tuple[ZPcController, list[tuple]]:
    """构造 ZPcController 触控桩: touch_down/touch_up/click 记录事件, 不发出真实输入。"""
    ctl = ZPcController.__new__(ZPcController)
    ctl.touch_ui_enabled = True
    ctl._touch_lock = threading.RLock()
    ctl._touch_owner = None
    ctl._touch_generation = 0
    ctl._touch_held = False
    ctl._touch_move_actions = set()
    ctl._touch_move_tokens = {}
    ctl._touch_vec = Point(0, 0)
    ctl._touch_float_pos = None
    ctl._touch_sent_pos = None
    ctl.action_keys = _ActionKeys()

    events: list[tuple] = []

    def fake_touch_down(pos: Point) -> bool:
        events.append(('down', time.time(), pos.tuple()))
        ctl._touch_held = True
        return True

    def fake_touch_up() -> None:
        events.append(('up', time.time(), None))
        ctl._touch_held = False

    def fake_click(pos: Point, press_time: float = 0, pc_alt: bool = False,
                   gamepad_key: str | None = None) -> bool:
        events.append(('click', time.time(), pos.tuple()))
        if press_time > 0:
            time.sleep(press_time)
        return True

    ctl.touch_down = fake_touch_down
    ctl.touch_up = fake_touch_up
    ctl.click = fake_click
    return ctl, events


class _ActionKeys(dict):
    """action_keys 桩: 缺失键统一返回 'x', 让触控路由能找到对应动作。"""

    def __missing__(self, key):
        return 'x'

    def get(self, key, default=None):
        return self[key]


# ==================== 画面区域分发 ====================

class TestPlayCoverAreaAlias:
    """ZScreenContext 对 按键-普通攻击 的平台分发。"""

    @pytest.mark.parametrize('from_separated_files', [False, True])
    @pytest.mark.parametrize('playcover,expected_template,expected_threshold,expected_rect', [
        (True, 'btn_normal_attack_playcover', 0.6, _PLAYCOVER_ATTACK_RECT),
        (False, 'btn_normal_attack', 0.7, _DESKTOP_ATTACK_RECT),
    ])
    def test_attack_area_alias(self, from_separated_files: bool, playcover: bool,
                               expected_template: str, expected_threshold: float,
                               expected_rect: Rect) -> None:
        sc = ZScreenContext(lambda: playcover)
        sc.reload(from_separated_files=from_separated_files)
        area = sc.get_area('战斗画面', '按键-普通攻击')
        assert area.template_id == expected_template
        assert area.template_match_threshold == expected_threshold
        assert area.rect == expected_rect

    def test_hud_areas_loaded(self) -> None:
        """归一化需要的两个代理人状态区都应存在。"""
        sc = _screen_context(True)
        playcover_rect = sc.get_area('战斗画面', '代理人状态-PlayCover').rect
        standard_rect = sc.get_area('战斗画面', '代理人状态-标准布局').rect
        assert playcover_rect == Rect(2, 2, 1017, 218)
        assert standard_rect == Rect(10, 0, 950, 180)


# ==================== 角色识别 ====================

class TestAgentRecognition:

    def test_windows_fixture_roster(self, isolated_battle_context: TestContext) -> None:
        """Windows 画面不经过归一化, 仍识别出原有队伍。"""
        _reset_agent_ctx(isolated_battle_context, playcover=False)
        screen = _load_fixture(_WINDOWS_FIXTURE)
        isolated_battle_context.auto_battle_context.agent_context.check_agent_related(screen, 100.0)

        roster = [i.agent.agent_id if i.agent else None
                  for i in isolated_battle_context.auto_battle_context.agent_context.team_info.agent_list]
        assert roster == ['yeshunguang', 'sunna', 'zhao']

    @pytest.mark.parametrize('fixture_name', [_PLAYCOVER_FIXTURE_1, _PLAYCOVER_FIXTURE_2])
    def test_playcover_fixture_roster(self, isolated_battle_context: TestContext, fixture_name: str) -> None:
        """PlayCover 画面归一化后识别 remielle/phoenix/velina。"""
        _reset_agent_ctx(isolated_battle_context, playcover=True)
        screen = _load_fixture(fixture_name)
        isolated_battle_context.auto_battle_context.agent_context.check_agent_related(screen, 100.0)

        agent_list = isolated_battle_context.auto_battle_context.agent_context.team_info.agent_list
        roster = [i.agent.agent_id if i.agent else None for i in agent_list]
        assert roster == ['remielle', 'phoenix', 'velina']
        assert agent_list[0].matched_template_id == 'remielle_seashade'

    def test_playcover_fixture_2_voidflare(self, isolated_battle_context: TestContext) -> None:
        """第二张实机画面应识别出 3 格虚曜。"""
        _reset_agent_ctx(isolated_battle_context, playcover=True)
        screen = _load_fixture(_PLAYCOVER_FIXTURE_2)
        isolated_battle_context.auto_battle_context.agent_context.check_agent_related(screen, 100.0)

        recorder = isolated_battle_context.auto_battle_context.state_record_service.get_state_recorder('蕾米埃尔-虚曜')
        assert recorder is not None
        assert recorder.last_value == 3


# ==================== 缺帧处理 ====================

class TestMissingFrame:

    def _prepare_battle_context(self, test_context: TestContext, playcover: bool = True):
        """返回 (abc, spy_list): 真实模板识别 + 屏蔽无关子任务。"""
        _reset_agent_ctx(test_context, playcover=playcover)
        abc = test_context.auto_battle_context
        spies = [
            patch.object(abc, 'dodge_context', MagicMock()),
            patch.object(abc, 'target_context', MagicMock()),
            patch.object(abc, 'check_quick_assist'),
            patch.object(abc, 'check_switch_backup'),
            patch.object(abc, 'check_chain_attack'),
            patch.object(abc, 'check_btn_ban', wraps=abc.check_btn_ban),
            patch.object(abc, 'is_normal_attack_btn_available',
                         wraps=abc.is_normal_attack_btn_available),
        ]
        return abc, spies

    def test_missing_frame_keeps_cached_state(self, isolated_battle_context: TestContext) -> None:
        """valid → None → valid: None 不清状态、不重发、不调用 CV。"""
        abc, spies = self._prepare_battle_context(isolated_battle_context)
        for p in spies:
            p.start()
        try:
            screen = _load_fixture(_PLAYCOVER_FIXTURE_1)
            assert abc.check_battle_state(screen, 100.0, sync=True) is True
            assert abc.last_check_in_battle is True
            ready = abc.state_record_service.get_state_recorder(
                BattleStateEnum.STATUS_NORMAL_ATTACK_READY.value)
            assert ready.last_record_time == 100.0

            cv_calls = abc.is_normal_attack_btn_available.call_count

            # None: 返回缓存的 True, 不调用 CV, 不重复写入状态
            assert abc.check_battle_state(None, 150.0, sync=True) is True
            assert abc.last_check_in_battle is True
            assert abc.is_normal_attack_btn_available.call_count == cv_calls
            assert abc.check_btn_ban.call_count == 1
            assert ready.last_record_time == 100.0

            # 恢复有效帧: 状态时间戳推进
            assert abc.check_battle_state(screen, 200.0, sync=True) is True
            assert ready.last_record_time == 200.0
        finally:
            for p in spies:
                p.stop()

    def test_black_screen_not_battle(self, isolated_battle_context: TestContext) -> None:
        """全黑画面 (无战斗UI) 不算战斗画面。"""
        abc, spies = self._prepare_battle_context(isolated_battle_context)
        for p in spies:
            p.start()
        try:
            black = np.zeros((1080, 1920, 3), dtype=np.uint8)
            assert abc.check_battle_state(black, 100.0, sync=True) is False
            assert abc.last_check_in_battle is False
            ready = abc.state_record_service.get_state_recorder(
                BattleStateEnum.STATUS_NORMAL_ATTACK_READY.value)
            # 初始没有记录时清除是 no-op, 记录器可能不存在或 last_record_time<=0
            if ready is not None:
                assert ready.last_record_time <= 0
        finally:
            for p in spies:
                p.stop()


# ==================== AreaPatrol 等待战斗画面 ====================

class TestWaitBattleScreen:

    def _make_patrol(self, playcover: bool) -> AreaPatrol:
        op = AreaPatrol.__new__(AreaPatrol)
        op.ctx = SimpleNamespace(
            screen_loader=_screen_context(playcover),
            tm=TemplateMatcher(TemplateLoader()),
        )
        op.node_retry_times = 0
        op.round_start_time = time.time() - 10  # 已过等待时间, 不再真的 sleep
        return op

    def test_none_screenshot_retries(self) -> None:
        op = self._make_patrol(playcover=False)
        op.last_screenshot = None
        result = op.wait_battle_screen()
        assert result.result == OperationRoundResultEnum.RETRY
        assert result.status == '未获取到游戏画面'

    def test_fixture_screenshot_succeeds(self) -> None:
        """PlayCover 触控布局下, wait_battle_screen 通过别名区域命中实机截图。"""
        op = self._make_patrol(playcover=True)
        op.last_screenshot = _load_fixture(_PLAYCOVER_FIXTURE_1)
        result = op.wait_battle_screen()
        assert result.result == OperationRoundResultEnum.SUCCESS


# ==================== macOS 截图失败告警 ====================

@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS 截图告警只在 darwin 路径生效')
class TestMacScreenshotWarning:

    def test_failure_warning_once_until_success(self) -> None:
        """连续失败只告警一次, 成功一次后再次失败重新告警。"""
        from one_dragon.base.controller.pc_screenshot import (
            pc_screenshot_controller as sc_mod,
        )

        fake_win = SimpleNamespace(win_rect=Rect(0, 0, 1920, 1080))
        ctl = sc_mod.PcScreenshotController(fake_win, 1920, 1080)
        ctl.active_strategy_name = 'pil'

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        capture_returns = iter([None, None, frame, None])
        fake_strategy = SimpleNamespace(
            capture=lambda rect, independent: next(capture_returns))
        ctl.strategies = {'pil': fake_strategy}

        with patch.object(sc_mod.Quartz, 'CGPreflightScreenCaptureAccess', return_value=True), \
                patch.object(sc_mod, 'log') as mock_log:
            for _ in range(4):
                ctl.get_screenshot()

            warning_calls = mock_log.warning.call_args_list
            assert len(warning_calls) == 2
            for call in warning_calls:
                assert '未能获取游戏窗口截图' in call.args[0]


# ==================== 触点序列 ====================

class TestTouchDispatch:

    def _run_ops(self, test_context: TestContext, op_name: str) -> list[tuple]:
        """走真实战斗门禁识别 fixture1 → 取 normal_scene 首条匹配 → 桩控制器执行, 返回事件序列。"""
        _reset_agent_ctx(test_context, playcover=True)
        abc = test_context.auto_battle_context
        screen = _load_fixture(_PLAYCOVER_FIXTURE_1)
        # 状态默认有效期 [0,1] 秒且未触发状态 last_record_time=-1,
        # 识别时间与匹配时间都必须用真实时间戳 否则否定状态表达式结果会反
        now = time.time()
        # 用完整 check_battle_state 证明这帧 PlayCover 画面能通过战斗门禁并写入角色状态
        with patch.object(abc, 'dodge_context', MagicMock()), \
                patch.object(abc, 'target_context', MagicMock()), \
                patch.object(abc, 'check_quick_assist'), \
                patch.object(abc, 'check_switch_backup'):
            assert abc.check_battle_state(screen, now, sync=True) is True

        ctl, events = _touch_stub_controller()
        original_controller = test_context.controller
        test_context.controller = ctl
        op: AutoBattleOperator | None = None
        try:
            op = AutoBattleOperator(abc, 'auto_battle', op_name, read_from_merged=True)
            success, msg = op.init_before_running()
            assert success, msg
            abc.auto_op = op

            exec_info = op.normal_scene.match_execution(now)
            assert exec_info is not None, f'{op_name} normal_scene 没有匹配到操作'
            executor = OperationExecutor(exec_info.op_list, trigger_time=time.time())
            future = executor.run_async()
            assert future.result(timeout=60) is True
        finally:
            if op is not None:
                op.dispose()
            test_context.controller = original_controller
            abc.auto_op = None
        return events

    def test_remielle_tap_and_hold(self, isolated_battle_context: TestContext) -> None:
        """点按普攻 → 按住普攻 2s → 松开, 并写入 连招完成 状态。"""
        events = self._run_ops(isolated_battle_context, 'Remielle - Tap and Hold')
        kinds = [(e[0], e[2]) for e in events]

        na = ZPcController.TOUCH_NORMAL_ATTACK.tuple()
        assert ('click', na) in kinds
        # 按下 → 2s 后松开
        down_idx = next(i for i, e in enumerate(events) if e[0] == 'down')
        up_idx = next(i for i, e in enumerate(events) if e[0] == 'up')
        assert events[down_idx][2] == na
        assert events[up_idx][1] - events[down_idx][1] >= 1.9
        click_idx = kinds.index(('click', na))
        assert click_idx < down_idx

        recorder = isolated_battle_context.auto_battle_context.state_record_service \
            .get_state_recorder('自定义-蕾米埃尔-连招完成')
        assert recorder is not None
        assert recorder.last_record_time > 0

    def test_remielle_m2_opener(self, isolated_battle_context: TestContext) -> None:
        """M2 开场: 长按普攻 → 长按特殊攻击 → 两次切换下一个。"""
        events = self._run_ops(isolated_battle_context, 'Remielle M2 - Phoenix - Velina')
        kinds = [(e[0], e[2]) for e in events]

        na = ZPcController.TOUCH_NORMAL_ATTACK.tuple()
        sa = ZPcController.TOUCH_BUTTON_POS['special_attack'].tuple()
        sn = ZPcController.TOUCH_BUTTON_POS['switch_next'].tuple()

        na_down = next(i for i, k in enumerate(kinds) if k == ('down', na))
        na_up = next(i for i in range(na_down + 1, len(kinds)) if kinds[i][0] == 'up')
        sa_down = next(i for i, k in enumerate(kinds) if k == ('down', sa))
        sa_up = next(i for i in range(sa_down + 1, len(kinds)) if kinds[i][0] == 'up')
        switch_indices = [i for i, k in enumerate(kinds) if k == ('click', sn)]
        assert len(switch_indices) == 2
        assert na_down < na_up < sa_down < sa_up < switch_indices[0] < switch_indices[1]

    def test_touch_button_positions(self) -> None:
        """触控按钮坐标: 终结技修正为 PlayCover 位置, 其他保持原值。"""
        assert ZPcController.TOUCH_NORMAL_ATTACK.tuple() == (1570, 805)
        assert ZPcController.TOUCH_BUTTON_POS['ultimate'].tuple() == (1716, 518)
        assert ZPcController.TOUCH_BUTTON_POS['switch_next'].tuple() == (1716, 724)


# ==================== 异常状态检测间隔 ====================

class TestAbnormalStatusInterval:

    def test_interval_toggle(self, isolated_battle_context: TestContext) -> None:
        """正数开启检测, 0 关闭, 且不修改全局任务定义。"""
        abc = isolated_battle_context.auto_battle_context
        target_ctx = abc.target_context

        task = next(t for t in target_ctx.tasks if t.task_id == 'abnormal_statuses')
        original_task_interval = task.interval

        m2_op: AutoBattleOperator | None = None
        basic_op: AutoBattleOperator | None = None
        try:
            # M2 间隔 1 → 开启; target_lock_interval 置 0 避免改动无关的全局任务定义
            m2_op = AutoBattleOperator(abc, 'auto_battle',
                                       'Remielle M2 - Phoenix - Velina', read_from_merged=True)
            ok, _ = m2_op.init_before_running()
            assert ok
            m2_op.target_lock_interval = 0
            target_ctx.init_auto_op(m2_op)
            assert target_ctx._current_intervals['abnormal_statuses'] == 1

            # 基础配置间隔 0 → 关闭, 任务定义不变
            basic_op = AutoBattleOperator(abc, 'auto_battle',
                                          'Remielle - Tap and Hold', read_from_merged=True)
            ok, _ = basic_op.init_before_running()
            assert ok
            basic_op.target_lock_interval = 0
            target_ctx.init_auto_op(basic_op)
            assert target_ctx._current_intervals['abnormal_statuses'] == 0
            assert task.interval == original_task_interval
        finally:
            if basic_op is not None:
                basic_op.dispose()
            if m2_op is not None:
                m2_op.dispose()
