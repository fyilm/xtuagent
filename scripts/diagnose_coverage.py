#!/usr/bin/env python
"""站点覆盖诊断：逐个探测目标站点，说明「为什么没抓到内容」。

爬完之后如果发现某些学院/部门站点贡献的文档数为 0，本脚本会告诉你
每个站点属于哪种情况，以及对应的处理办法：

- `OK`       —— 能正常抓到 HTML，没抓到多半是页数上限或栏目没铺开；
- `WAF-403`  —— 站点返回 403，普通请求被拦，需要浏览器渲染或换网络环境；
- `JS骨架`   —— 返回 200 但 HTML 极小、正文由前端异步加载，需要渲染；
- `网络错`   —— 超时 / DNS 失败，检查网络或该站点是否下线。

用法::

    uv run python scripts/diagnose_coverage.py
    uv run python scripts/diagnose_coverage.py --timeout 20
"""

import argparse
import collections
import os
import socket
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import requests

from xtuagent.core.config import settings
from xtuagent.core.logging import setup_logging
from xtuagent.crawler.site_config import TARGET_SITES
from xtuagent.crawler.spider import Spider

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# 判定为「骨架页」的 HTML 体积上限（字节）
SHELL_SIZE = 4000


def classify(status: int, html: str) -> str:
    if status == 403:
        return "WAF-403"
    if status != 200:
        return f"HTTP-{status}"
    if len(html) < SHELL_SIZE:
        return "JS骨架"
    return "OK"


def probe_network(timeout: int) -> None:
    """打印出口公网 IP 与代理设置。

    这一步很关键：**403 是按来源网络放行的**，所以先要确认「跑爬虫的这台机器
    到底从哪个 IP 出去」。在受管代理下运行的请求，其出口 IP 由代理决定，
    改本地网络并不会改变它。
    """
    print("\n=== 网络出口 ===")
    proxy_vars = {
        k: v for k, v in os.environ.items() if k.lower() in ("http_proxy", "https_proxy")
    }
    print(f"  代理环境变量：{proxy_vars or '（无）'}")

    # 本机在「出网」时会用哪块网卡：向外部地址建一个 UDP socket（不发包）
    # 再看内核选了哪个源地址。用它就能确认走的是有线还是无线。
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("223.5.5.5", 80))
            print(f"  本机出网源地址：{s.getsockname()[0]}（即实际选中的网卡）")
    except OSError as exc:
        print(f"  无法判定本机出网网卡：{exc}")
    for url in ("https://ipinfo.io/json", "https://ifconfig.me/all.json"):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            data = resp.json()
            ip = data.get("ip") or data.get("ip_addr")
            print(
                f"  出口公网 IP：{ip}"
                f"  {data.get('city', '')} {data.get('region', '')} {data.get('country', '')}"
                f"  {data.get('org', '')}".rstrip()
            )
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  查询出口 IP 失败（{type(exc).__name__}），继续…")


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断各目标站点的可抓取性")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--texts-dir", type=Path, default=settings.texts_dir)
    parser.add_argument("--skip-net", action="store_true", help="跳过出口 IP 探测")
    args = parser.parse_args()

    setup_logging()
    socket.setdefaulttimeout(args.timeout)

    if not args.skip_net:
        probe_network(args.timeout)

    counts = collections.Counter()
    if args.texts_dir.exists():
        for f in args.texts_dir.glob("*.txt"):
            counts[f.name.split("_")[0]] += 1

    spider = Spider(prefer_ipv4=False)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results = []
    for site in TARGET_SITES:
        host = site["url"].split("//")[1].split(".")[0]
        docs = counts.get(host, 0)
        try:
            resp = session.get(site["url"], timeout=args.timeout, allow_redirects=True)
            html = spider._decode_bytes(resp.content)
            kind = classify(resp.status_code, html)
        except Exception as exc:  # noqa: BLE001
            kind = f"网络错({type(exc).__name__})"
        results.append((site["name"], host, docs, kind))

    width = max(len(r[0]) for r in results) + 2
    print(f"\n{'站点'.ljust(width)}{'前缀':<10}{'文本数':>7}   状态")
    print("-" * (width + 34))
    for name, host, docs, kind in results:
        mark = "" if docs else "  ←"
        print(f"{name.ljust(width)}{host:<10}{docs:>7}   {kind}{mark}")

    by_kind = collections.Counter(k for *_, k in results)
    empty = [r for r in results if r[2] == 0]
    print("\n状态分布：", dict(by_kind))
    print(f"贡献 0 篇文本的站点：{len(empty)} / {len(results)}")

    if any(k == "WAF-403" for _, _, _, k in results):
        print(
            "\n提示：403 是按**来源网络**放行的，不是认证问题（加 Cookie 无效）。\n"
            "  1) 先对比上面打印的「出口公网 IP」：若本次是在受管代理/云环境里跑的，\n"
            "     出口 IP 由代理决定，改本地网络不会改变它；\n"
            "  2) 请在**你自己的终端**（不经过代理）里再跑一次本脚本，比对出口 IP。\n"
            "     若那时这些站点变成 OK，说明你的网络能直连，直接在本地终端跑\n"
            "     `uv run python scripts/crawl.py --fresh` 即可；\n"
            "  3) 若仍 403，可加 --render 用无头浏览器抓（需已安装 playwright）。"
        )
    if any(k == "JS骨架" for _, _, _, k in results):
        print("\n提示：`JS骨架` 站点的正文由前端异步加载，只能靠 --render 渲染后抓取。")


if __name__ == "__main__":
    main()
