#!/usr/bin/env node
/*
 * 前端「回答区多语言」回归测试
 *
 * 运行：node tests/i18n.test.js
 *
 * 背景：后端会按提问语言作答（见 src/xtuagent/core/lang.py），但前端的来源卡片、
 * 结尾提示曾经写死中文——英文回答底下挂一句「知识库未收录，已联网检索 3 条」。
 * 本测试锁住两件事：
 *   1. detectLang 的判定顺序与后端一致（假名优先于汉字，否则日文会被认成中文）；
 *   2. 五种语言的文案表都没有缺项，且真被翻译过（不是把中文抄一遍）。
 *
 * 与 render_markdown.test.js 一样，直接从 web/js/app.js 抽取被测函数执行，
 * 避免维护手抄副本造成漂移。
 */

const fs = require("fs");
const path = require("path");

const APP = path.join(__dirname, "..", "web", "js", "app.js");
const src = fs.readFileSync(APP, "utf8");

const start = src.indexOf("const KANA =");
const end = src.indexOf("/* ============ 状态轮询");
if (start < 0 || end < 0) {
  console.error("✗ 无法从 web/js/app.js 定位多语言代码块，抽取标记需更新");
  process.exit(2);
}

const sandbox = {};
new Function(
  "exports",
  src.slice(start, end) +
    "\nexports.detectLang = detectLang;\n" +
    "exports.t = t;\n" +
    "exports.UI_TEXT = UI_TEXT;\n" +
    "exports.DEFAULT_LANG = DEFAULT_LANG;"
)(sandbox);
const { detectLang, t, UI_TEXT, DEFAULT_LANG } = sandbox;

let failed = 0;
const check = (name, ok, extra) => {
  if (ok) {
    console.log(`✓ ${name}`);
  } else {
    failed += 1;
    console.log(`✗ ${name}${extra ? `\n     ${extra}` : ""}`);
  }
};

/* ---------- 1. 语言识别 ---------- */

const LANG_CASES = [
  ["你好，绩点怎么算", "中文"],
  ["请问 GPA 怎么算", "中文"], // 中文里夹英文仍是中文
  ["How can I download MATLAB?", "英语"],
  ["図書館の開館時間を教えてください", "日语"], // 必须优先于汉字判定
  ["안녕하세요, 도서관 이용 시간이 궁금합니다", "韩语"],
  ["Где можно скачать MATLAB?", "俄语"],
  ["12345", DEFAULT_LANG], // 判不出来时退回默认
  ["", DEFAULT_LANG],
];

for (const [input, expect] of LANG_CASES) {
  const got = detectLang(input);
  check(`detectLang(${JSON.stringify(input.slice(0, 24))}) → ${expect}`, got === expect, `实际 ${got}`);
}

/* ---------- 2. 文案表完整性 ---------- */

const LANGS = Object.keys(UI_TEXT);
check(`文案表覆盖 5 种语言（实际 ${LANGS.length}）`, LANGS.length === 5, LANGS.join(","));
check("文案表包含默认语言", LANGS.includes(DEFAULT_LANG));

const KEYS = Object.keys(UI_TEXT[DEFAULT_LANG]);
for (const lang of LANGS) {
  const missing = KEYS.filter((k) => !UI_TEXT[lang][k]);
  check(`${lang}：${KEYS.length} 个键无缺项`, missing.length === 0, `缺 ${missing.join(", ")}`);
  const extra = Object.keys(UI_TEXT[lang]).filter((k) => !KEYS.includes(k));
  check(`${lang}：没有多余键`, extra.length === 0, `多 ${extra.join(", ")}`);
}

/* ---------- 3. 确实被翻译过 ---------- */

for (const lang of LANGS) {
  if (lang === DEFAULT_LANG) continue;
  const identical = KEYS.filter((k) => UI_TEXT[lang][k] === UI_TEXT[DEFAULT_LANG][k]);
  check(`${lang}：没有照抄中文的条目`, identical.length === 0, `照抄 ${identical.join(", ")}`);
}

/* ---------- 4. 取值与占位符 ---------- */

check("占位符被替换", t("英语", "metaWeb", { n: 3 }).includes("3"));
check("占位符不会残留", !t("日语", "kbSummary", { n: 2 }).includes("{n}"));
check("未知语言退回默认语言", t("世界语", "sourcesTitle") === UI_TEXT[DEFAULT_LANG].sourcesTitle);
check("未知键退回默认语言", t("英语", "不存在的键") === undefined);

console.log(`\n${failed === 0 ? "全部通过" : failed + " 项失败"}`);
process.exit(failed ? 1 : 0);
