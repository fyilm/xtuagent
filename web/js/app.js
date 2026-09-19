/* ============ XtuAgent 前端逻辑 ============ */

const API = {
  health: "/health",
  stats: "/api/stats",
  askStream: "/api/ask/stream",
};

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
  examples: $("examples"),
  clearBtn: $("clearBtn"),
  messages: $("messages"),
  hero: $("hero"),
  input: $("input"),
  sendBtn: $("sendBtn"),
  toast: $("toast"),
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
    if (data.meta) {
      document.querySelector(".tech-note").textContent =
        `${data.meta.embedding_model} · ${data.count} 向量 · GLM-4-Flash`;
    }
  } catch { /* 忽略 */ }
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

  text = escapeHtml(text);
  text = text
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>");

  text = text.replace(/(?:^|\n)((?:[-*] .+(?:\n|$))+)/g, (_, list) => {
    const items = list.trim().split("\n")
      .map((line) => `<li>${line.replace(/^[-*] /, "")}</li>`).join("");
    return `\n<ul>${items}</ul>`;
  });
  text = text.replace(/(?:^|\n)((?:\d+\. .+(?:\n|$))+)/g, (_, list) => {
    const items = list.trim().split("\n")
      .map((line) => `<li>${line.replace(/^\d+\. /, "")}</li>`).join("");
    return `\n<ol>${items}</ol>`;
  });

  const restore = (chunk) => chunk.replace(/\u0000(\d+)\u0000/g, (_, i) => codeBlocks[i]);
  return text
    .split(/\n{2,}/)
    .map((part) => {
      const trimmed = part.trim();
      if (!trimmed) return "";
      if (/^<(h\d|ul|ol|pre)/.test(trimmed)) return restore(trimmed);
      return `<p>${restore(trimmed).replace(/\n/g, "<br>")}</p>`;
    })
    .join("");
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

function renderSources(body, sources) {
  if (!sources || sources.length === 0) return;
  const card = document.createElement("div");
  card.className = "sources";
  const header = document.createElement("div");
  header.className = "sources-header";
  header.innerHTML = `<span class="arrow">▶</span><span>参考来源 · ${sources.length} 条</span>`;
  header.addEventListener("click", () => card.classList.toggle("open"));

  const list = document.createElement("div");
  list.className = "sources-list";
  sources.forEach((s) => {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <div class="source-file">
        <span>${escapeHtml(s.file)}</span>
        <span class="source-score">相似度 ${(s.score * 100).toFixed(0)}%</span>
      </div>
      <div class="source-snippet">${escapeHtml(s.snippet)}</div>`;
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

/* ============ 流式问答 ============ */

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

  const typing = document.createElement("div");
  typing.className = "typing";
  typing.innerHTML = "<span></span><span></span><span></span>";
  contentEl.appendChild(typing);

  let answerText = "";
  let sources = [];

  try {
    const resp = await fetch(API.askStream, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        language: els.langSelect.value,
      }),
    });

    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `请求失败 (${resp.status})`);
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
        handleEvent(event, { body, contentEl, typing, get answerText() { return answerText; },
          set answerText(v) { answerText = v; },
          get sources() { return sources; },
          set sources(v) { sources = v; },
        });
      }
    }
  } catch (error) {
    typing.remove();
    contentEl.innerHTML = `<p style="color:#f87171">出错了：${escapeHtml(error.message)}</p>`;
  } finally {
    typing.remove();
    state.streaming = false;
    els.sendBtn.disabled = false;
    els.input.focus();
  }

  function handleEvent(event, ctx) {
    if (event.type === "sources") {
      ctx.sources = event.sources || [];
    } else if (event.type === "delta") {
      if (ctx.typing.parentNode) ctx.typing.remove();
      ctx.answerText += event.text;
      ctx.contentEl.innerHTML = renderMarkdown(ctx.answerText) + '<span class="cursor"></span>';
      scrollToBottom();
    } else if (event.type === "done") {
      if (ctx.typing.parentNode) ctx.typing.remove();
      ctx.contentEl.innerHTML = renderMarkdown(ctx.answerText);
      if (ctx.sources.length) renderSources(ctx.body, ctx.sources);
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      meta.textContent = `耗时 ${(event.elapsed_ms / 1000).toFixed(1)}s · 检索 ${ctx.sources.length} 条来源`;
      ctx.body.appendChild(meta);
      scrollToBottom();
    } else if (event.type === "error") {
      if (ctx.typing.parentNode) ctx.typing.remove();
      ctx.contentEl.innerHTML = `<p style="color:#f87171">生成失败：${escapeHtml(event.message)}</p>`;
    }
  }
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
