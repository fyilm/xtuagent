#!/usr/bin/env python
"""端到端验证：知识库答不出来时，系统是否真的去网上把答案找回来。

用**真实 Chromium** 打开前端、真实键盘输入、真实点击按钮，并劫持 `fetch`
把服务端下发的每一条 SSE 事件原样记录下来——所有结论都基于真实网络往返，
不接受"单元测试过了所以线上一定对"。

用法：
    uv run python scripts/e2e_web_fallback.py --base http://127.0.0.1:8010
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _ROOT / "proof"

SSE_HOOK = r"""
window.__sse = [];
window.__netErr = [];
const _fetch = window.fetch.bind(window);
window.fetch = async function (...args) {
  const url = typeof args[0] === "string" ? args[0] : (args[0] && args[0].url) || "";
  const resp = await _fetch(...args);
  if (url.indexOf("/api/ask/stream") >= 0 && resp.body) {
    const clone = resp.clone();
    (async () => {
      try {
        const reader = clone.body.getReader();
        const dec = new TextDecoder("utf-8");
        let buf = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          let i;
          while ((i = buf.indexOf("\n\n")) >= 0) {
            const frame = buf.slice(0, i);
            buf = buf.slice(i + 2);
            const line = frame.split("\n").find((l) => l.startsWith("data: "));
            if (line) {
              try { window.__sse.push(JSON.parse(line.slice(6))); } catch (e) {}
            }
          }
        }
      } catch (e) { window.__netErr.push(String(e)); }
    })();
  }
  return resp;
};
"""

CASES = [
    {
        "id": "out_of_domain",
        "question": "今天长沙的天气怎么样？",
        "expect": "web",
        "why": "纯域外问题，知识库不可能有依据，必须联网找",
    },
    {
        "id": "kb_only",
        "question": "学分制是怎么回事？",
        "expect": "kb",
        "why": "知识库能答，绝不允许联网（否则白花钱）",
    },
    {
        "id": "kb_gap",
        "question": "校园网 VPN 怎么使用？",
        "expect": "any",
        "why": "知识库可能有简介但缺操作步骤，看实际命中情况",
    },
]


def wait_ready(base: str, timeout: int = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data = httpx.get(f"{base}/health", timeout=5).json()
            if data.get("status") == "ready":
                return data
            if data.get("status") == "error":
                sys.exit(f"服务预热失败：{data.get('error')}")
            print(f"  预热中…… ({data.get('status')})")
        except Exception as exc:  # noqa: BLE001
            print(f"  等待服务…… ({exc})")
        time.sleep(3)
    sys.exit("等待服务就绪超时")


def ask(page, question: str) -> dict:
    page.evaluate("window.__sse = []")
    page.click("#clearBtn")
    page.fill("#input", "")
    # 真实键盘输入，不用 JS 直接塞 value
    page.click("#input")
    page.type("#input", question, delay=12)
    page.click("#sendBtn")

    page.wait_for_function(
        "() => window.__sse.some(e => e.type === 'done' || e.type === 'error')",
        timeout=180000,
    )
    page.wait_for_timeout(400)

    events = page.evaluate("window.__sse")
    net_err = page.evaluate("window.__netErr")
    rendered = page.evaluate(
        """() => {
        const msgs = document.querySelectorAll('.msg.assistant');
        const last = msgs[msgs.length - 1];
        if (!last) return null;
        const sources = [...last.querySelectorAll('.source-item')].map((n) => ({
          badge: (n.querySelector('.source-badge') || {}).textContent || '',
          name:  (n.querySelector('.source-name') || {}).textContent || '',
          url:   (n.querySelector('a.source-name') || {}).href || '',
          title: (n.querySelector('.source-title') || {}).textContent || '',
        }));
        return {
          answer: (last.querySelector('.msg-content') || {}).innerText || '',
          meta:   (last.querySelector('.msg-meta') || {}).textContent || '',
          sources,
        };
      }"""
    )
    return {"events": events, "net_err": net_err, **rendered}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8010")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    OUT_DIR.mkdir(exist_ok=True)
    print(f"目标：{base}")
    health = wait_ready(base)
    print(f"服务就绪：知识库 {health['index_count']} 条 · 模型 {health['model']}\n")

    report = {"base": base, "cases": []}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 960},
            record_video_dir=str(OUT_DIR / "video"),
            record_video_size={"width": 1440, "height": 960},
        )
        page = context.new_page()
        page.add_init_script(SSE_HOOK)

        console_errors = []
        page.on(
            "console",
            lambda m: console_errors.append(m.text) if m.type == "error" else None,
        )

        page.goto(base, wait_until="domcontentloaded")
        page.wait_for_selector("#statusDot.ready", timeout=180000)
        build = page.evaluate("window.__XtuAgentBuild")
        print(f"前端构建号：{build}\n")

        for case in CASES:
            print(f"── [{case['id']}] {case['question']}")
            result = ask(page, case["question"])

            types = [e["type"] for e in result["events"]]
            deltas = "".join(
                e.get("text", "") for e in result["events"] if e["type"] == "delta"
            )
            done = next((e for e in result["events"] if e["type"] == "done"), {})
            mode = done.get("mode", "?")
            kb_src = sum(1 for s in result["sources"] if s["badge"] == "知识库")
            web_src = sum(1 for s in result["sources"] if s["badge"] == "网络")

            print(f"   事件序列：{' → '.join(types)}")
            print(f"   模式：{mode} ｜ 来源：知识库 {kb_src} 条 / 网络 {web_src} 条")
            print(f"   答案首行：{result['answer'].strip().splitlines()[0][:70] if result['answer'].strip() else '（空）'}")
            print(f"   状态栏：{result['meta']}")
            leaked = "暂未找到" in result["answer"]
            if leaked:
                print("   ⚠️ 拒答话术泄漏到了界面！")
            if result["net_err"]:
                print(f"   ⚠️ 前端读取流异常：{result['net_err']}")
            if mode == "web" and "非学校官方发布" not in result["answer"]:
                print("   ⚠️ 网络回答缺少免责声明！")

            # 展开来源卡片再截图，否则「知识库 / 网络」标签和跳转链接看不见
            page.evaluate(
                """() => document
                    .querySelectorAll('.msg.assistant .sources')
                    .forEach((n) => n.classList.add('open'))"""
            )
            page.wait_for_timeout(300)
            shot = OUT_DIR / f"e2e_{case['id']}.png"
            page.screenshot(path=str(shot), full_page=True)

            passed = True
            if case["expect"] == "web" and mode != "web":
                passed = False
            if case["expect"] == "kb" and mode != "kb":
                passed = False
            if leaked:
                passed = False
            if mode == "web" and "非学校官方发布" not in result["answer"]:
                passed = False

            report["cases"].append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "why": case["why"],
                    "expect": case["expect"],
                    "mode": mode,
                    "events": types,
                    "kb_sources": kb_src,
                    "web_sources": web_src,
                    "answer": result["answer"],
                    "meta": result["meta"],
                    "sources": result["sources"],
                    "refusal_leaked": leaked,
                    "passed": passed,
                    "screenshot": str(shot),
                }
            )
            print(f"   {'✅ 通过' if passed else '❌ 未达预期'}\n")

        context.close()  # 关闭后才能拿到录像
        browser.close()

    report["console_errors"] = console_errors
    report["build"] = build
    out = OUT_DIR / "e2e_web_fallback.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 64)
    ok = sum(1 for c in report["cases"] if c["passed"])
    print(f"结果：{ok}/{len(report['cases'])} 通过")
    if console_errors:
        print(f"浏览器控制台报错 {len(console_errors)} 条：{console_errors[:3]}")
    else:
        print("浏览器控制台无报错")
    print(f"证据：{out}")


if __name__ == "__main__":
    main()
