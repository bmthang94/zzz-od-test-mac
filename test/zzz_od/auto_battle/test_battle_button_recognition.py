"""战斗画面普通攻击按键模板与状态判定测试。

验证:
1. battle.yml 与 _od_merged.yml 中 按键-普通攻击 (桌面默认) 与
   按键-普通攻击-PlayCover 的配置一致性。
2. 离线模板匹配对实机战斗画面的高置信度与负样本(纯色)的区分度。
3. AutoBattleContext.is_normal_attack_btn_available 能够正确读取配置的 template_id 与 threshold。
"""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from one_dragon.base.geometry.rectangle import Rect
from one_dragon.base.matcher.template_matcher import TemplateMatcher
from one_dragon.base.screen.screen_area import ScreenArea
from one_dragon.base.screen.screen_loader import ScreenContext
from one_dragon.base.screen.template_loader import TemplateLoader
from one_dragon.utils import cv2_utils
from zzz_od.auto_battle.auto_battle_context import AutoBattleContext

_SCREEN_DIR = Path(__file__).parent.parent.parent.parent / 'screens' / '战斗画面'


def test_battle_button_screen_area_config() -> None:
    """验证 按键-普通攻击 恢复为桌面默认, PlayCover 走独立区域。"""
    sc = ScreenContext()
    sc.reload(from_separated_files=False)
    area_merged = sc.get_area('战斗画面', '按键-普通攻击')
    assert area_merged is not None
    assert area_merged.template_id == 'btn_normal_attack'
    assert area_merged.template_sub_dir == 'battle'
    assert area_merged.template_match_threshold == 0.7
    assert area_merged.rect == Rect(1403, 916, 1505, 1017)

    playcover_merged = sc.get_area('战斗画面', '按键-普通攻击-PlayCover')
    assert playcover_merged is not None
    assert playcover_merged.template_id == 'btn_normal_attack_playcover'
    assert playcover_merged.template_match_threshold == 0.6
    assert playcover_merged.rect == Rect(1470, 715, 1670, 915)

    sc.reload(from_separated_files=True)
    area_sep = sc.get_area('战斗画面', '按键-普通攻击')
    assert area_sep is not None
    assert area_sep.template_id == 'btn_normal_attack'
    assert area_sep.template_match_threshold == 0.7

    playcover_sep = sc.get_area('战斗画面', '按键-普通攻击-PlayCover')
    assert playcover_sep is not None
    assert playcover_sep.template_id == 'btn_normal_attack_playcover'
    assert playcover_sep.template_match_threshold == 0.6


def test_battle_button_offline_discrimination() -> None:
    """验证 btn_normal_attack_playcover 在实机战斗画面与负样本上的区分度。"""
    tl = TemplateLoader()
    tm = TemplateMatcher(tl)

    template = tl.get_template('battle', 'btn_normal_attack_playcover')
    assert template is not None
    assert template.raw is not None
    assert template.mask is not None

    rect = Rect(1470, 715, 1670, 915)

    # 负样本1: 纯黑 (loading 画面典型)
    black_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mrl_black = tm.crop_and_match_template(black_screen, rect, 'battle', 'btn_normal_attack_playcover', threshold=0.6)
    assert mrl_black.max is None

    # 负样本2: 纯白 (高光过曝/白屏)
    white_screen = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    mrl_white = tm.crop_and_match_template(white_screen, rect, 'battle', 'btn_normal_attack_playcover', threshold=0.6)
    assert mrl_white.max is None

    # 实机正样本 (PlayCover 存档画面) 验证匹配置信度 >= 0.6
    fixture_path = _SCREEN_DIR / 'playcover-实机-1.webp'
    assert fixture_path.exists(), f'存档截图不存在: {fixture_path}'
    img_1080 = cv2_utils.read_image(str(fixture_path))
    mrl_combat = tm.crop_and_match_template(img_1080, rect, 'battle', 'btn_normal_attack_playcover', threshold=0.6)
    assert mrl_combat.max is not None
    assert mrl_combat.max.confidence >= 0.6


def test_auto_battle_context_uses_configured_template() -> None:
    """验证 AutoBattleContext.is_normal_attack_btn_available 使用 area_btn_normal 配置。"""
    mock_ctx = MagicMock()
    mock_ctx.project_config.screen_standard_width = 1920
    mock_tm = MagicMock()
    mock_ctx.tm = mock_tm

    abc = AutoBattleContext(mock_ctx)
    abc.area_btn_normal = ScreenArea(
        area_name='按键-普通攻击',
        pc_rect=Rect(1470, 715, 1670, 915),
        template_sub_dir='battle',
        template_id='btn_normal_attack_playcover',
        template_match_threshold=0.6,
    )

    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mock_mrl = MagicMock()
    mock_mrl.max = MagicMock()
    mock_tm.crop_and_match_template.return_value = mock_mrl

    available = abc.is_normal_attack_btn_available(screen)
    assert available is True
    mock_tm.crop_and_match_template.assert_called_once_with(
        screen,
        Rect(1470, 715, 1670, 915),
        'battle',
        'btn_normal_attack_playcover',
        threshold=0.6,
    )


def test_auto_battle_context_desktop_threshold() -> None:
    """桌面 btn_normal_attack 保持历史运行阈值 0.9。"""
    mock_ctx = MagicMock()
    mock_ctx.project_config.screen_standard_width = 1920
    mock_tm = MagicMock()
    mock_ctx.tm = mock_tm

    abc = AutoBattleContext(mock_ctx)
    abc.area_btn_normal = ScreenArea(
        area_name='按键-普通攻击',
        pc_rect=Rect(1403, 916, 1505, 1017),
        template_sub_dir='battle',
        template_id='btn_normal_attack',
        template_match_threshold=0.7,
    )

    screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mock_mrl = MagicMock()
    mock_mrl.max = MagicMock()
    mock_tm.crop_and_match_template.return_value = mock_mrl

    assert abc.is_normal_attack_btn_available(screen) is True
    mock_tm.crop_and_match_template.assert_called_once_with(
        screen,
        Rect(1403, 916, 1505, 1017),
        'battle',
        'btn_normal_attack',
        threshold=0.9,
    )
