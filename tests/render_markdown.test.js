#!/usr/bin/env node
/*
 * 前端 Markdown 渲染回归测试
 *
 * 运行：node tests/render_markdown.test.js
 *
 * 背景：`renderMarkdown` 直接把模型输出渲染进 innerHTML，项目里此前没有任何前端测试。
 * 曾出过一个隐蔽 bug——模型在编号项之间夹空行，导致每个编号项被单独包成 <ol>，
 * 浏览器各自从 1 重新编号，页面上表现为「1. 1. 1. 1. 1.」。
 *
 * 本测试直接从 web/js/app.js 抽取被测函数执行，避免维护手抄副本造成漂移。
 */

const fs = require("fs");
const path = require("path");

const APP = path.join(__dirname, "..", "web", "js", "app.js");
const src = fs.readFileSync(APP, "utf8");

const start = src.indexOf("function escapeHtml");
const end = src.indexOf("/* ============ 消息渲染");
if (start < 0 || end < 0) {
  console.error("✗ 无法从 web/js/app.js 定位渲染函数，正则抽取需更新");
  process.exit(2);
}

const sandbox = {};
new Function("exports", src.slice(start, end) + "\nexports.renderMarkdown = renderMarkdown;")(sandbox);
const renderMarkdown = sandbox.renderMarkdown;

const CASES = [
  {
    name: "编号项首尾相连（原有正常行为）",
    input: "1. 甲\n2. 乙\n3. 丙",
    expect: { ol: 1, ul: 0, li: 3 },
  },
  {
    name: "编号项之间夹空行（曾经的 bug：会变成 5 个各含 1 项的 ol）",
    input: "1. 甲\n\n2. 乙\n\n3. 丙",
    expect: { ol: 1, ul: 0, li: 3 },
  },
  {
    name: "顶层短横线列表",
    input: "- 甲\n- 乙",
    expect: { ol: 0, ul: 1, li: 2 },
  },
  {
    name: "中文顿号编号 1、（无空格）",
    input: "1、甲\n2、乙",
    expect: { ol: 1, ul: 0, li: 2 },
  },
  {
    name: "缩进子项嵌套进父项",
    input: "1. 甲\n   - a\n   - b\n2. 乙",
    expect: { ol: 1, ul: 1, li: 4 },
  },
  {
    name: "列表与标题、段落共存",
    input: "## 标题\n正文一段\n\n1. 一\n2. 二",
    expect: { ol: 1, ul: 0, li: 2, h2: 1, p: 1 },
  },
  {
    name: "两段之间隔正文不应被合并",
    input: "- 甲\n\n中间一句\n\n- 乙",
    expect: { ol: 0, ul: 2, li: 2 },
  },
  {
    name: "代码块不被当成列表",
    input: "说明：\n\n```python\nprint('hi')\n```\n\n- 甲",
    expect: { ol: 0, ul: 1, li: 1, pre: 1 },
  },
  {
    name: "星号列表与加粗不互扰",
    input: "* **甲**：说明\n* 乙",
    expect: { ol: 0, ul: 1, li: 2, strong: 1 },
  },
  {
    name: "小数不被误判为编号列表",
    input: "3.14是圆周率\n2.5是另一个数",
    expect: { ol: 0, ul: 0, li: 0, p: 1 },
  },
  {
    name: "无空格的短横线不算列表",
    input: "a-b 是范围表达",
    expect: { ol: 0, ul: 0, li: 0, p: 1 },
  },
  {
    name: "URL 中的斜杠与冒号不被转义破坏",
    input: "- 访问 https://vpn.xtu.edu.cn?a=1&b=2",
    expect: { ol: 0, ul: 1, li: 1 },
  },
  {
    name: "HTML 注入被转义",
    input: "1. <script>alert(1)</script>",
    expect: { ol: 1, li: 1, script: 0 },
  },
];

const count = (html, tag) => (html.match(new RegExp(`<${tag}[ >]`, "g")) || []).length;

let failed = 0;
for (const c of CASES) {
  const html = renderMarkdown(c.input);
  const got = {};
  for (const tag of ["ol", "ul", "li", "h2", "p", "pre", "strong", "script"]) {
    got[tag] = count(html, tag);
  }
  const bad = Object.entries(c.expect).filter(([k, v]) => got[k] !== v);
  if (bad.length) {
    failed += 1;
    console.log(`✗ ${c.name}`);
    bad.forEach(([k, v]) => console.log(`     ${k}: 期望 ${v}，实际 ${got[k]}`));
    console.log(`     HTML: ${html}`);
  } else {
    console.log(`✓ ${c.name}`);
  }
}

console.log(`\n${CASES.length - failed}/${CASES.length} 通过`);
process.exit(failed ? 1 : 0);
