from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from one_dragon.base.matcher.template_matcher import TemplateMatcher
from one_dragon.base.screen.screen_loader import ScreenContext
from one_dragon.base.screen.screen_utils import OcrClickResultEnum, find_and_click_area
from one_dragon.base.screen.template_loader import TemplateLoader
from one_dragon.utils import cv2_utils


@pytest.fixture(scope='module')
def screen_loader() -> ScreenContext:
    loader = ScreenContext()
    loader.reload()
    return loader


@pytest.mark.parametrize(('filename', 'expected_center'), [
    ('screens/大世界/普通-PlayCover.webp', (1512, 101)),
    ('normal_world_basic.webp', (1555, 81)),
    ('normal_world_basic_2.webp', (1469, 81)),
    ('menu.webp', None),
])
def test_normal_world_compendium_entry(
    screen_loader: ScreenContext, filename: str, expected_center: tuple[int, int] | None,
) -> None:
    test_dir = Path(__file__).resolve().parent
    path = test_dir.parents[5] / filename if filename.startswith('screens/') else test_dir / filename
    screen = cv2_utils.read_image(str(path))
    assert screen is not None
    controller = Mock()
    controller.click.return_value = True
    ctx = SimpleNamespace(screen_loader=screen_loader, tm=TemplateMatcher(TemplateLoader()), controller=controller)
    route = screen_loader.get_screen_route('大世界-普通', '快捷手册-训练')
    assert route is not None and route.can_go
    hop = route.node_list[0]
    result = find_and_click_area(ctx, screen, hop.from_screen, hop.from_area)
    if expected_center is None:
        assert result == OcrClickResultEnum.OCR_CLICK_NOT_FOUND
        controller.click.assert_not_called()
    else:
        assert result == OcrClickResultEnum.OCR_CLICK_SUCCESS
        controller.click.assert_called_once()
        args, kwargs = controller.click.call_args
        assert kwargs['pc_alt'] is True
        assert abs(args[0].x - expected_center[0]) <= 5
        assert abs(args[0].y - expected_center[1]) <= 5
