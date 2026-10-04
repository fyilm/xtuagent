/* ============ XtuAgent 前端逻辑 ============ */

const API = {
  health: "/health",
  stats: "/api/stats",
  askStream: "/api/ask/stream",
  agent: "/api/agent",
};

// 前端构建号：由后端按静态资源内容自动算出并注入到 <meta name="x-build">。
// 用途：一眼确认浏览器加载的到底是不是最新 JS。
// 同一个号也用于资源 URL（index.html 里 ?v={{BUILD}}）——文件一变 URL 就变，
// 浏览器必然重新拉取，所以正常情况下不需要用户手动清缓存。
const APP_BUILD =
  (document.querySelector('meta[name="x-build"]') || {}).content || "dev";
window.__XtuAgentBuild = APP_BUILD;

const state = {
  ready: false,
  streaming: false,
  history: [],
};

const $ = (id) => document.getElementById(id);
const els = {
  statusCard: $("statusCard"),
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  agentToggle: $("agentToggle"),
  clearBtn: $("clearBtn"),
  messages: $("messages"),
  hero: $("hero"),
  input: $("input"),
  sendBtn: $("sendBtn"),
  toast: $("toast"),
};

/* 助手标记：与侧边栏品牌同一个图形，避免界面里出现两套视觉符号 */
const ASSISTANT_MARK =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" ' +
  'stroke-linecap="round" stroke-linejoin="round">' +
  '<path d="M22 10 12 5 2 10l10 5 10-5z"></path>' +
  '<path d="M6 12v5c0 1.7 2.7 3 6 3s6-1.3 6-3v-5"></path></svg>';

/* ============ 提问语言识别 ============ */

/* 与 `src/xtuagent/core/lang.py` 是同一套规则，改一处要同步另一处。
 *
 * 后端已经把 language 回传过来了（done 事件 / /api/agent 响应），
 * 这里还要在前端再判一次的原因：**打字机提示、错误提示在拿到响应之前就要显示**，
 * 那时候后端还没说话。
 *
 * 顺序不能变：假名 → 谚文 → 汉字。日文句子必然夹假名，
 * 而汉字区间会同时命中中日文，先判汉字会把日文提问认成中文。 */
const KANA = /[\u3040-\u309f\u30a0-\u30ff]/;
const HANGUL = /[\uac00-\ud7af\u1100-\u11ff]/;
const HAN = /[\u4e00-\u9fff\u3400-\u4dbf]/;
const CYRILLIC = /[\u0400-\u04ff]/;
const LATIN = /[A-Za-z]/;
const DEFAULT_LANG = "中文";

function detectLang(text) {
  if (!text) return DEFAULT_LANG;
  if (KANA.test(text)) return "日语";
  if (HANGUL.test(text)) return "韩语";
  if (HAN.test(text)) return "中文";
  if (CYRILLIC.test(text)) return "俄语";
  if (LATIN.test(text)) return "英语";
  return DEFAULT_LANG;
}

/* ============ 回答区文案 ============ */

/* 边界：**回答气泡里的所有文字都跟着回答语言走**，包括来源卡片、结尾提示、
 * 报错。应用外壳（按钮、占位符、状态条、侧边栏）保持中文——
 * 那些在提问之前就存在，跟回答语言无关，跟着变只会让人以为界面在乱跳。 */
const UI_TEXT = {
  中文: {
    sourcesTitle: "参考来源",
    kbSummary: "知识库 {n} 条",
    webSummary: "网络 {n} 条",
    badgeKb: "知识库",
    badgeWeb: "网络",
    sourceUnknown: "未知来源",
    sourceWeb: "网络来源",
    simScore: "相似度 {n}%",
    webPage: "网页",
    metaWeb: "知识库未收录，已联网检索 {n} 条",
    metaRefused: "知识库与网络均未找到依据",
    metaAgentWeb: "已联网检索 {n} 条",
    metaAgentNone: "未命中任何来源",
    agentHint: "正在检索知识库…",
    noAnswer: "（无回答）",
    errorPrefix: "出错了：",
    requestFailed: "请求失败",
  },
  英语: {
    sourcesTitle: "Sources",
    kbSummary: "knowledge base {n}",
    webSummary: "web {n}",
    badgeKb: "Knowledge base",
    badgeWeb: "Web",
    sourceUnknown: "Unknown source",
    sourceWeb: "Web source",
    simScore: "similarity {n}%",
    webPage: "Web page",
    metaWeb: "Not in the knowledge base — searched the web, {n} result(s)",
    metaRefused: "No basis found in the knowledge base or on the web",
    metaAgentWeb: "Searched the web — {n} result(s)",
    metaAgentNone: "No sources matched",
    agentHint: "Searching the knowledge base…",
    noAnswer: "(no answer)",
    errorPrefix: "Error: ",
    requestFailed: "Request failed",
  },
  日语: {
    sourcesTitle: "参考資料",
    kbSummary: "ナレッジベース {n}件",
    webSummary: "ウェブ {n}件",
    badgeKb: "ナレッジベース",
    badgeWeb: "ウェブ",
    sourceUnknown: "不明なソース",
    sourceWeb: "ウェブソース",
    simScore: "類似度 {n}%",
    webPage: "ウェブページ",
    metaWeb: "学内ナレッジベースに未収録のため、ウェブで {n} 件検索しました",
    metaRefused: "ナレッジベースにもウェブにも根拠が見つかりませんでした",
    metaAgentWeb: "ウェブで {n} 件検索しました",
    metaAgentNone: "該当するソースがありません",
    agentHint: "ナレッジベースを検索しています…",
    noAnswer: "（回答なし）",
    errorPrefix: "エラー：",
    requestFailed: "リクエストに失敗しました",
  },
  韩语: {
    sourcesTitle: "출처",
    kbSummary: "지식 베이스 {n}건",
    webSummary: "웹 {n}건",
    badgeKb: "지식 베이스",
    badgeWeb: "웹",
    sourceUnknown: "알 수 없는 출처",
    sourceWeb: "웹 출처",
    simScore: "유사도 {n}%",
    webPage: "웹 페이지",
    metaWeb: "교내 지식 베이스에 없어 웹에서 {n}건을 검색했습니다",
    metaRefused: "지식 베이스와 웹 모두에서 근거를 찾지 못했습니다",
    metaAgentWeb: "웹에서 {n}건을 검색했습니다",
    metaAgentNone: "일치하는 출처가 없습니다",
    agentHint: "지식 베이스를 검색하는 중…",
    noAnswer: "(답변 없음)",
    errorPrefix: "오류: ",
    requestFailed: "요청 실패",
  },
  俄语: {
    sourcesTitle: "Источники",
    kbSummary: "база знаний: {n}",
    webSummary: "интернет: {n}",
    badgeKb: "База знаний",
    badgeWeb: "Интернет",
    sourceUnknown: "Неизвестный источник",
    sourceWeb: "Веб-источник",
    simScore: "сходство {n}%",
    webPage: "Веб-страница",
    metaWeb: "Нет в базе знаний — поиск в интернете, {n} результат(ов)",
    metaRefused: "Ни в базе знаний, ни в интернете оснований не найдено",
    metaAgentWeb: "Поиск в интернете — {n} результат(ов)",
    metaAgentNone: "Подходящих источников не найдено",
    agentHint: "Поиск в базе знаний…",
    noAnswer: "(нет ответа)",
    errorPrefix: "Ошибка: ",
    requestFailed: "Запрос не выполнен",
  },
};

/* 取回答区文案。语言没收录时退回中文——宁可一句语言不对，也不要让界面空掉。 */
function t(lang, key, vars) {
  const table = UI_TEXT[lang] || UI_TEXT[DEFAULT_LANG];
  let s = table[key];
  if (s === undefined) s = UI_TEXT[DEFAULT_LANG][key];
  if (vars) {
    for (const k of Object.keys(vars)) s = s.split(`{${k}}`).join(vars[k]);
  }
  return s;
}

/* ============ 状态轮询 ============ */

function setStatus(kind, text) {
  els.statusDot.className = `dot ${kind}`;
  els.statusText.textContent = text;
}

async function pollHealth() {
  try {
    const resp = await fetch(API.health);
    const data = await resp.json();
    if (data.status === "ready") {
      state.ready = true;
      setStatus("ready", "服务就绪");
      els.sendBtn.disabled = false;
      return true;
    }
    if (data.status === "error") {
      setStatus("error", data.error || "服务异常，请查看后端日志");
      return true;
    }
    setStatus("loading", "模型预热中，首次启动约需 1 分钟");
    return false;
  } catch {
    setStatus("error", "无法连接服务，请确认后端已启动");
    return false;
  }
}

async function init() {
  setStatus("loading", "正在连接服务…");
  els.sendBtn.disabled = true;

  const done = await pollHealth();
  if (!done) {
    const timer = setInterval(async () => {
      const ready = await pollHealth();
      if (ready) clearInterval(timer);
    }, 2000);
  }
}

/* ============ Markdown 轻量渲染 ============ */

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(source) {
  const codeBlocks = [];
  let text = source.replace(/```[\w]*\n?([\s\S]*?)```/g, (_, code) => {
    codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    return `\u0000${codeBlocks.length - 1}\u0000`;
  });

  text = escapeHtml(text).replace(/\r\n?/g, "\n");
  text = text
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>");

  return restoreCode(buildBlocks(text), codeBlocks);
}

/*
 * 识别列表行：允许前导缩进；标记支持 - * 以及 1. 1、
 * 需区分「真列表」和正文，故：
 *   - `-`/`*` 后必须有空白（`-abc` 不是列表）
 *   - `1.` 后必须有空白（否则 `3.14是圆周率` 会被误判）
 *   - `1、` 后允许无空白（中文顿号习惯不空格，且顿号不可能是小数点）
 */
function parseListLine(line) {
  const m = line.match(/^([ \t]*)([-*]|\d+[.、])([ \t]*)(.*)$/);
  if (!m) return null;
  const [, ws, marker, sep, content] = m;
  if (!content) return null;
  const isBullet = /^[-*]$/.test(marker);
  if (isBullet && !sep) return null;
  if (/^\d+\.$/.test(marker) && !sep) return null;
  return {
    indent: ws.replace(/\t/g, "  ").length,
    type: isBullet ? "ul" : "ol",
    content,
  };
}

/*
 * 按行构建块级结构。
 * 关键点：编号项之间常夹空行，但不能因此把列表切断——
 * 否则每一项都会变成独立的 <ol>，浏览器从 1 重新编号（表现为「全是 1.」）。
 * 这里空行不打断列表，只有「非列表且非空」的行才终结列表。
 */
function buildBlocks(text) {
  const root = { children: [] };
  const listStack = [];
  let para = [];
  let container = root.children;

  const flushPara = () => {
    if (!para.length) return;
    container.push({ type: "raw", html: `<p>${para.join("<br>")}</p>` });
    para = [];
  };

  for (const line of text.split("\n")) {
    if (line.trim() === "") {
      if (listStack.length === 0) flushPara();
      continue;
    }

    const parsed = parseListLine(line);
    if (!parsed) {
      if (/^<h\d>/.test(line) || /^\u0000\d+\u0000$/.test(line.trim())) {
        flushPara();
        listStack.length = 0;
        container = root.children;
        container.push({ type: "raw", html: line.trim() });
        continue;
      }
      if (listStack.length) listStack.length = 0;
      container = root.children;
      para.push(line);
      continue;
    }

    flushPara();
    const { indent, type, content } = parsed;

    while (listStack.length && listStack[listStack.length - 1].indent > indent) {
      listStack.pop();
    }

    const top = listStack[listStack.length - 1];
    if (top && top.indent === indent && top.type === type) {
      const item = { text: content, children: [] };
      top.node.items.push(item);
      top.lastItem = item;
      container = item.children;
      continue;
    }

    let parentContainer;
    if (top && top.indent === indent) {
      parentContainer = listStack.pop().parentContainer;
    } else if (top) {
      parentContainer = top.lastItem ? top.lastItem.children : root.children;
    } else {
      parentContainer = root.children;
    }

    const node = { type, items: [] };
    parentContainer.push({ type: "list", node });
    const ctx = { indent, type, node, lastItem: null, parentContainer };
    listStack.push(ctx);
    const item = { text: content, children: [] };
    node.items.push(item);
    ctx.lastItem = item;
    container = item.children;
  }

  flushPara();
  return serialize(root.children);
}

function serialize(nodes) {
  return nodes
    .map((n) => {
      if (n.type === "raw") return n.html;
      const items = n.node.items
        .map((it) => `<li>${it.text}${serialize(it.children)}</li>`)
        .join("");
      return `<${n.node.type}>${items}</${n.node.type}>`;
    })
    .join("");
}

function restoreCode(html, codeBlocks) {
  return html.replace(/\u0000(\d+)\u0000/g, (_, i) => codeBlocks[i]);
}

/* ============ 消息渲染 ============ */

function hideHero() {
  if (els.hero) {
    els.hero.style.display = "none";
  }
}

function appendMessage(role, content) {
  hideHero();
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const body = document.createElement("div");
  body.className = "msg-body";

  const contentEl = document.createElement("div");
  contentEl.className = "msg-content";
  contentEl.innerHTML = role === "user" ? escapeHtml(content) : content;
  body.appendChild(contentEl);

  // 只有助手消息带标记：用户消息靠右对齐的浅灰气泡就够区分了，再加头像反而啰嗦
  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.innerHTML = ASSISTANT_MARK;
    wrapper.appendChild(avatar);
  }

  wrapper.appendChild(body);
  els.messages.appendChild(wrapper);
  scrollToBottom();
  return { wrapper, body, contentEl };
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

/* 来源卡片：知识库片段与网络检索结果统一展示，用「知识库 / 网络」标签区分。
 *
 * 网络来源的标题、站点名、URL 全部来自外部网页，一律用 textContent 赋值，
 * 绝不拼 HTML 字符串——否则一个恶意网页标题就能注入脚本。 */
function renderSources(body, sources, lang) {
  if (!sources || sources.length === 0) return;
  const kbCount = sources.filter((s) => s.kind !== "web").length;
  const webCount = sources.length - kbCount;
  const summary = [];
  if (kbCount) summary.push(t(lang, "kbSummary", { n: kbCount }));
  if (webCount) summary.push(t(lang, "webSummary", { n: webCount }));

  const card = document.createElement("div");
  card.className = "sources";
  const header = document.createElement("div");
  header.className = "sources-header";
  header.innerHTML = `<span class="arrow">▶</span><span></span>`;
  header.lastElementChild.textContent =
    `${t(lang, "sourcesTitle")} · ${summary.join(" / ")}`;
  header.addEventListener("click", () => card.classList.toggle("open"));

  const list = document.createElement("div");
  list.className = "sources-list";

  sources.forEach((s) => {
    const isWeb = s.kind === "web";
    const item = document.createElement("div");
    item.className = isWeb ? "source-item source-web" : "source-item";

    const head = document.createElement("div");
    head.className = "source-file";

    const badge = document.createElement("span");
    badge.className = "source-badge";
    badge.textContent = t(lang, isWeb ? "badgeWeb" : "badgeKb");

    const name = document.createElement(isWeb && s.url ? "a" : "span");
    name.className = "source-name";
    name.textContent =
      s.file || t(lang, isWeb ? "sourceWeb" : "sourceUnknown");
    if (isWeb && s.url) {
      name.href = s.url;
      name.target = "_blank";
      name.rel = "noopener noreferrer";
      name.title = s.url;
    }

    const score = document.createElement("span");
    score.className = "source-score";
    score.textContent = isWeb
      ? s.publish_date || t(lang, "webPage")
      : t(lang, "simScore", { n: ((s.score || 0) * 100).toFixed(0) });

    head.append(badge, name, score);
    item.appendChild(head);

    if (isWeb && s.title) {
      const title = document.createElement("div");
      title.className = "source-title";
      title.textContent = s.title;
      title.title = s.title;
      item.appendChild(title);
    }

    const snippet = document.createElement("div");
    snippet.className = "source-snippet";
    snippet.textContent = s.snippet || "";
    item.appendChild(snippet);

    list.appendChild(item);
  });

  card.appendChild(header);
  card.appendChild(list);
  body.appendChild(card);
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  setTimeout(() => els.toast.classList.remove("show"), 3200);
}

/* ============ 问答主流程 ============ */

function makeTyping() {
  const typing = document.createElement("div");
  typing.className = "typing";
  typing.innerHTML = "<span></span><span></span><span></span>";
  return typing;
}

function appendMeta(body, text) {
  const meta = document.createElement("div");
  meta.className = "msg-meta";
  meta.textContent = text;
  body.appendChild(meta);
}

async function send() {
  const question = els.input.value.trim();
  if (!question || state.streaming) return;
  if (!state.ready) {
    showToast("服务尚未就绪，请稍候");
    return;
  }

  state.streaming = true;
  els.sendBtn.disabled = true;
  els.input.value = "";
  autoResize();

  appendMessage("user", question);
  const { body, contentEl } = appendMessage("assistant", "");
  const typing = makeTyping();
  contentEl.appendChild(typing);

  // 提问语言在这里判一次带进 ctx：打字机提示、报错都要在响应回来之前就定语言。
  const ctx = { question, lang: detectLang(question), body, contentEl, typing };

  try {
    if (els.agentToggle.checked) {
      await runAgentMode(ctx);
    } else {
      await runStreamMode(ctx);
    }
  } finally {
    if (typing.parentNode) typing.remove();
    // 兜底清理：流中途异常时，过渡提示可能没被收掉，别让它永远挂在那儿
    body.querySelectorAll(".msg-status").forEach((n) => n.remove());
    state.streaming = false;
    els.sendBtn.disabled = false;
    els.input.focus();
  }
}

/* ---------- 模式一：普通 RAG 流式问答 ---------- */

async function runStreamMode(ctx) {
  const { question, body, contentEl, typing } = ctx;
  let answerText = "";
  let sources = [];
  let webSources = [];
  let statusEl = null;

  const clearStatus = () => {
    if (statusEl && statusEl.parentNode) statusEl.remove();
    statusEl = null;
  };
  const setStatus = (text) => {
    clearStatus();
    statusEl = document.createElement("div");
    statusEl.className = "msg-status";
    statusEl.textContent = text;
    body.appendChild(statusEl);
    scrollToBottom();
  };

  const resp = await fetch(API.askStream, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // language 交给后端按提问内容自动判定（见 serving/schemas.py 的 _resolve_language）
    body: JSON.stringify({ question, language: "auto" }),
  });

  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    typing.remove();
    const msg = detail.detail || `${t(ctx.lang, "requestFailed")} (${resp.status})`;
    contentEl.innerHTML =
      `<p style="color:#f87171">${escapeHtml(t(ctx.lang, "errorPrefix") + msg)}</p>`;
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop();

    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      let event;
      try {
        event = JSON.parse(line.slice(6));
      } catch {
        continue;
      }
      handleStreamEvent(event, {
        body,
        contentEl,
        typing,
        lang: ctx.lang,
        get answerText() { return answerText; },
        set answerText(v) { answerText = v; },
        get sources() { return sources; },
        set sources(v) { sources = v; },
        get webSources() { return webSources; },
        set webSources(v) { webSources = v; },
        setStatus,
        clearStatus,
      });
    }
  }
}

/* 结尾提示只在「值得让用户知道」时才出现。
 * 正常走知识库的情况，来源卡片里已经写明条数，这里再补一行「耗时 x.xs · 检索 N 条」
 * 属于重复信息，只会让界面变吵。只有联网兜底/完全无依据这类需要用户警觉的情况才提示。
 * 文案跟着回答语言走（lang），否则英文回答底下挂一句中文提示。 */
function describeDone(event, kbCount, webCount, lang) {
  if (event.mode === "web") {
    return t(lang, "metaWeb", { n: webCount });
  }
  if (event.mode === "refused") {
    return t(lang, "metaRefused");
  }
  return "";
}

function handleStreamEvent(event, ctx) {
  // 后端在 done 事件里回报它实际用的语言；中途的事件先按提问语言渲染。
  const lang = event.language || ctx.lang;
  if (event.type === "sources") {
    ctx.sources = event.sources || [];
  } else if (event.type === "web_sources") {
    ctx.webSources = event.sources || [];
  } else if (event.type === "reset") {
    // 模型先吐了拒答话术，随后才判定该转联网 —— 把已渲染内容整个抹掉重来。
    // 抹掉是必须的：留着「暂未找到相关信息」再补一段网络答案，用户会以为自相矛盾。
    // 状态文字不在这里写死：紧跟其后的 status 事件会带来本地化版本。
    ctx.answerText = "";
    ctx.sources = [];
    ctx.webSources = [];
    if (ctx.typing.parentNode) ctx.typing.remove();
    ctx.contentEl.innerHTML = "";
    ctx.body.querySelectorAll(".sources, .msg-meta").forEach((n) => n.remove());
    ctx.clearStatus();
    scrollToBottom();
  } else if (event.type === "status") {
    ctx.setStatus(event.text);
  } else if (event.type === "delta") {
    ctx.clearStatus();
    if (ctx.typing.parentNode) ctx.typing.remove();
    ctx.answerText += event.text;
    ctx.contentEl.innerHTML = renderMarkdown(ctx.answerText) + '<span class="cursor"></span>';
    scrollToBottom();
  } else if (event.type === "done") {
    ctx.clearStatus();
    if (ctx.typing.parentNode) ctx.typing.remove();
    ctx.contentEl.innerHTML = renderMarkdown(ctx.answerText);
    const all = ctx.sources.concat(ctx.webSources);
    if (all.length) renderSources(ctx.body, all, lang);
    const note = describeDone(event, ctx.sources.length, ctx.webSources.length, lang);
    if (note) appendMeta(ctx.body, note);
    scrollToBottom();
  } else if (event.type === "error") {
    ctx.clearStatus();
    if (ctx.typing.parentNode) ctx.typing.remove();
    ctx.contentEl.innerHTML =
      `<p style="color:#f87171">${escapeHtml(t(lang, "errorPrefix") + event.message)}</p>`;
  }
}

/* ---------- 模式二：工具增强（ReAct Agent，非流式） ---------- */

async function runAgentMode(ctx) {
  const { question, body, contentEl, typing } = ctx;
  typing.remove();

  const hint = document.createElement("div");
  hint.className = "msg-status";
  hint.textContent = t(ctx.lang, "agentHint");
  contentEl.appendChild(hint);
  scrollToBottom();

  let data;
  try {
    const resp = await fetch(API.agent, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // language 交给后端按提问内容自动判定
      body: JSON.stringify({ question, language: "auto" }),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      const msg = detail.detail || `${t(ctx.lang, "requestFailed")} (${resp.status})`;
      throw new Error(msg);
    }
    data = await resp.json();
  } catch (error) {
    contentEl.innerHTML =
      `<p style="color:#f87171">${escapeHtml(t(ctx.lang, "errorPrefix") + error.message)}</p>`;
    return;
  }

  const lang = data.language || ctx.lang;
  contentEl.innerHTML = renderMarkdown(data.answer || t(lang, "noAnswer"));
  if (data.sources && data.sources.length) renderSources(body, data.sources, lang);
  const kbCount = (data.sources || []).filter((s) => s.kind !== "web").length;
  const webCount = (data.sources || []).length - kbCount;
  // 与流式模式保持一致：来源条数由来源卡片表达，这里只提需要用户警觉的情况
  const parts = [];
  if (webCount) parts.push(t(lang, "metaAgentWeb", { n: webCount }));
  if (!kbCount && !webCount) parts.push(t(lang, "metaAgentNone"));
  if (parts.length) appendMeta(body, parts.join(" · "));
  scrollToBottom();
}


/* ============ 输入框 ============ */

function autoResize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 180) + "px";
}

els.input.addEventListener("input", autoResize);
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
els.sendBtn.addEventListener("click", send);

els.clearBtn.addEventListener("click", () => {
  els.messages.innerHTML = "";
  if (els.hero) {
    els.hero.style.display = "flex";
    els.messages.appendChild(els.hero);
  }
});

/* ============ 启动 ============ */

init();
els.input.focus();
