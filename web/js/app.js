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
  statusSub: $("statusSub"),
  langSelect: $("langSelect"),
  agentToggle: $("agentToggle"),
  examples: $("examples"),
  clearBtn: $("clearBtn"),
  messages: $("messages"),
  hero: $("hero"),
  input: $("input"),
  sendBtn: $("sendBtn"),
  toast: $("toast"),
  techNote: $("techNote"),
};

const EXAMPLES = [
  "考试作弊会有什么处分？",
  "如何办理休学手续？",
  "奖学金申请的流程是什么？",
  "休学后还能复学吗？",
  "学分制是怎么回事？",
  "校园网 VPN 怎么使用？",
];

/* ============ 状态轮询 ============ */

function setStatus(kind, title, sub) {
  els.statusDot.className = `dot ${kind}`;
  els.statusText.textContent = title;
  els.statusSub.textContent = sub || "";
}

async function pollHealth() {
  try {
    const resp = await fetch(API.health);
    const data = await resp.json();
    if (data.status === "ready") {
      state.ready = true;
      setStatus("ready", "服务就绪", `知识库 ${data.index_count} 条 · ${data.model}`);
      els.sendBtn.disabled = false;
      return true;
    }
    if (data.status === "error") {
      setStatus("error", "服务异常", data.error || "请运行自检脚本排查");
      return true;
    }
    setStatus("loading", "模型预热中…", "首次启动约需 1 分钟");
    return false;
  } catch {
    setStatus("error", "无法连接服务", "请确认后端已启动");
    return false;
  }
}

async function init() {
  setStatus("loading", "正在连接服务…", "初始化中");
  els.sendBtn.disabled = true;

  const done = await pollHealth();
  if (!done) {
    const timer = setInterval(async () => {
      const ready = await pollHealth();
      if (ready) clearInterval(timer);
    }, 2000);
  }

  try {
    const resp = await fetch(API.stats);
    const data = await resp.json();
    const model = (data.meta && data.meta.embedding_model) || data.embedding_model || "bge-base-zh";
    const short = model.split("/").pop();
    els.techNote.textContent =
      `${short} · 阈值 ${data.score_threshold ?? "-"} · ${data.model || "GLM"} · 构建 ${APP_BUILD}`;
  } catch {
    els.techNote.textContent = `构建 ${APP_BUILD}`;
  }
}

/* ============ 示例问题 ============ */

function renderExamples() {
  EXAMPLES.forEach((q) => {
    const btn = document.createElement("button");
    btn.className = "example-item";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      if (!state.ready || state.streaming) return;
      els.input.value = q;
      send();
    });
    els.examples.appendChild(btn);
  });
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

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = role === "user" ? "🧑" : "🎓";

  const body = document.createElement("div");
  body.className = "msg-body";

  const contentEl = document.createElement("div");
  contentEl.className = "msg-content";
  contentEl.innerHTML = role === "user" ? escapeHtml(content) : content;

  body.appendChild(contentEl);
  wrapper.appendChild(avatar);
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
function renderSources(body, sources) {
  if (!sources || sources.length === 0) return;
  const kbCount = sources.filter((s) => s.kind !== "web").length;
  const webCount = sources.length - kbCount;
  const summary = [];
  if (kbCount) summary.push(`知识库 ${kbCount} 条`);
  if (webCount) summary.push(`网络 ${webCount} 条`);

  const card = document.createElement("div");
  card.className = "sources";
  const header = document.createElement("div");
  header.className = "sources-header";
  header.innerHTML = `<span class="arrow">▶</span><span></span>`;
  header.lastElementChild.textContent = `参考来源 · ${summary.join(" / ")}`;
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
    badge.textContent = isWeb ? "网络" : "知识库";

    const name = document.createElement(isWeb && s.url ? "a" : "span");
    name.className = "source-name";
    name.textContent = s.file || (isWeb ? "网络来源" : "未知来源");
    if (isWeb && s.url) {
      name.href = s.url;
      name.target = "_blank";
      name.rel = "noopener noreferrer";
      name.title = s.url;
    }

    const score = document.createElement("span");
    score.className = "source-score";
    score.textContent = isWeb
      ? s.publish_date || "网页"
      : `相似度 ${((s.score || 0) * 100).toFixed(0)}%`;

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

  const ctx = { question, body, contentEl, typing };

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
    body: JSON.stringify({ question, language: els.langSelect.value }),
  });

  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    typing.remove();
    contentEl.innerHTML = `<p style="color:#f87171">出错了：${escapeHtml(detail.detail || `请求失败 (${resp.status})`)}</p>`;
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

function describeDone(event, kbCount, webCount) {
  const bits = [`耗时 ${((event.elapsed_ms || 0) / 1000).toFixed(1)}s`];
  if (event.mode === "web") {
    bits.push(`知识库未收录 · 已联网检索 ${webCount} 条`);
  } else if (event.mode === "refused") {
    bits.push("知识库与网络均未找到依据");
  } else {
    bits.push(`检索 ${kbCount} 条来源`);
  }
  return bits.join(" · ");
}

function handleStreamEvent(event, ctx) {
  if (event.type === "sources") {
    ctx.sources = event.sources || [];
  } else if (event.type === "web_sources") {
    ctx.webSources = event.sources || [];
  } else if (event.type === "reset") {
    // 模型先吐了拒答话术，随后才判定该转联网 —— 把已渲染内容整个抹掉重来。
    // 抹掉是必须的：留着「暂未找到相关信息」再补一段网络答案，用户会以为自相矛盾。
    ctx.answerText = "";
    ctx.sources = [];
    ctx.webSources = [];
    if (ctx.typing.parentNode) ctx.typing.remove();
    ctx.contentEl.innerHTML = "";
    ctx.body.querySelectorAll(".sources, .msg-meta").forEach((n) => n.remove());
    ctx.setStatus("校内知识库未收录，正在联网检索…");
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
    if (all.length) renderSources(ctx.body, all);
    appendMeta(ctx.body, describeDone(event, ctx.sources.length, ctx.webSources.length));
    scrollToBottom();
  } else if (event.type === "error") {
    ctx.clearStatus();
    if (ctx.typing.parentNode) ctx.typing.remove();
    ctx.contentEl.innerHTML = `<p style="color:#f87171">生成失败：${escapeHtml(event.message)}</p>`;
  }
}

/* ---------- 模式二：工具增强（ReAct Agent，非流式） ---------- */

async function runAgentMode(ctx) {
  const { question, body, contentEl, typing } = ctx;
  typing.remove();

  const hint = document.createElement("div");
  hint.className = "msg-meta";
  hint.textContent = "工具增强模式：正在检索知识库…";
  contentEl.appendChild(hint);
  scrollToBottom();

  let data;
  try {
    const resp = await fetch(API.agent, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, language: els.langSelect.value }),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `请求失败 (${resp.status})`);
    }
    data = await resp.json();
  } catch (error) {
    contentEl.innerHTML = `<p style="color:#f87171">出错了：${escapeHtml(error.message)}</p>`;
    return;
  }

  contentEl.innerHTML = renderMarkdown(data.answer || "（无回答）");
  if (data.sources && data.sources.length) renderSources(body, data.sources);
  const kbCount = (data.sources || []).filter((s) => s.kind !== "web").length;
  const webCount = (data.sources || []).length - kbCount;
  const parts = [`耗时 ${((data.elapsed_ms || 0) / 1000).toFixed(1)}s`];
  if (kbCount) parts.push(`知识库 ${kbCount} 条`);
  if (webCount) parts.push(`联网 ${webCount} 条`);
  if (!kbCount && !webCount) parts.push("未命中任何来源");
  appendMeta(body, `工具增强模式 · ${parts.join(" · ")}`);
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

renderExamples();
init();
els.input.focus();
