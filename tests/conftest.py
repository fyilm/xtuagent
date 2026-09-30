"""测试全局夹具。

关键约定：**单元测试永远不打真实网络。**

联网兜底（`web_fallback_enabled`）默认在测试里关闭。原因有三：
1. 会真的花钱（搜索 API 按次计费）；
2. 结果随外部网页变化，测试必然 flaky；
3. 一个「跑测试」的动作不该产生出网流量。

需要验证联网兜底的用例，自行 monkeypatch 打开开关，并把 `web_search`
替换成桩函数——见 `tests/test_pipeline_webfallback.py`。
"""

import pytest

from xtuagent.core.config import settings


@pytest.fixture(autouse=True)
def _disable_web_fallback(monkeypatch):
    monkeypatch.setattr(settings, "web_fallback_enabled", False)
    yield
