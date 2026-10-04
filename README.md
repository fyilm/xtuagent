# XtuAgent · 基于 RAG 的校园智能学业问答系统

面向高校场景的私域知识库问答系统：爬取学校官网公开信息 → 构建本地向量索引 →
结合大模型生成**可追溯来源**的学业问答。所有数据与模型都在本机运行，
检索不到时会**如实拒答**而不是编造。

---

## 1. 系统架构

```
                     ┌──────────────── 离线阶段 ────────────────┐
  校园官网 ──爬虫──▶ data/texts/*.txt ──分块──▶ 嵌入(bge) ──▶ FAISS 索引
                     └──────────────────────────────────────────┘
                                          │
                     ┌──────────────── 在线阶段 ────────────────┐
  浏览器 ──▶ FastAPI ──▶ RAGPipeline ──▶ 向量检索 + 阈值过滤 ──▶ 智谱 GLM ──▶ SSE 流式回答
                     └──────────────────────────────────────────┘
```

| 目录 | 职责 |
|---|---|
| `src/xtuagent/core/` | 配置中心、日志、领域异常 |
| `src/xtuagent/crawler/` | 站点配置、BFS 爬虫、文件下载、PDF/文本抽取 |
| `src/xtuagent/ingest/` | 文档加载、中文语义分块、索引构建编排 |
| `src/xtuagent/rag/` | 嵌入服务、FAISS 向量库、检索、LLM 工厂、RAG 流水线 |
| `src/xtuagent/serving/` | FastAPI 应用、服务容器（预热状态机）、请求/响应模型 |
| `src/xtuagent/agent.py` | ReAct 工具调用 Agent（知识库驱动，无模拟数据） |
| `web/` | 原生 HTML/CSS/JS 前端（流式渲染 + 来源展示） |
| `scripts/` | 爬取 / 建库 / 启动 / 自检 / 评估 入口 |
| `tests/` | 单元测试（使用假嵌入，秒级完成，不加载模型） |

---

## 2. 环境准备

依赖 [`uv`](https://docs.astral.sh/uv/) 管理，Python ≥ 3.11。

```bash
uv sync --all-extras      # 创建 .venv 并安装全部依赖
```

### 2.1 配置环境变量

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

编辑 `.env`，填入智谱开放平台 Key（https://open.bigmodel.cn/ ）：

```ini
ZHIPU_API_KEY=你的真实Key

# 嵌入模型：指向本地离线模型目录（见 2.2），必须配置
EMBEDDING_MODEL=D:/xtuagent/models/bge-base-zh
```

> **为什么必须用本地模型目录？**
> `src/xtuagent/core/config.py` 会强制设置 `TRANSFORMERS_OFFLINE=1` /
> `HF_HUB_OFFLINE=1`，且 `EmbeddingService` 以 `local_files_only=True` 加载模型
> —— 目的是让服务在断网/内网环境下也能稳定启动。因此模型必须**提前落盘**。

### 2.2 准备嵌入模型（离线）

默认使用 `BAAI/bge-base-zh`（768 维）。把它下载到 `models/bge-base-zh/`：

```bash
# 方式一：能直连 HuggingFace 时
uv run python -c "from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-base-zh', local_dir='models/bge-base-zh')"

# 方式二：国内镜像（推荐；若镜像抖动可重试）
HF_ENDPOINT=https://hf-mirror.com uv run python -c "from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-base-zh', local_dir='models/bge-base-zh')"
```

只需以下文件即可推理：`config.json`、`config_sentence_transformers.json`、
`modules.json`、`sentence_bert_config.json`、`1_Pooling/config.json`、
`special_tokens_map.json`、`tokenizer.json`、`tokenizer_config.json`、
`vocab.txt`、`model.safetensors`（`pytorch_model.bin` 与前者等价，可不下载）。

> 下载失败排查：Windows 上若 `snapshot_download` 生成的快照文件为 0 字节
> （符号链接不可用导致），改为逐文件直取：
> `https://hf-mirror.com/BAAI/bge-base-zh/resolve/main/<文件名>`

---

## 3. 运行

### 3.1 一键控制台（Windows）

```bash
start.bat
```
菜单包含：启动服务 / 重建索引 / 爬取官网 / 系统自检 / 运行测试。

### 3.2 手动执行

```bash
uv run python scripts/crawl.py        # 1. 爬取官网文档 → data/texts/
uv run python scripts/build_index.py  # 2. 构建 FAISS 索引 → data/index/
uv run python scripts/serve.py        # 3. 启动服务 → http://127.0.0.1:8000
```

#### 关于爬取耗时与断点续爬（重要）

整站爬取**很慢**：全校域下有几十个子站、上千个页面，加上礼貌延时，
一次完整爬取通常需要十几分钟到几十分钟。

- 进度保存在 `data/.crawl_state.json`（已访问 URL + **待爬队列**）。
  **中断后直接重新运行 `crawl.py` 即可从断点继续**，不会从头再爬。
- 单次运行有全局页数上限 `CRAWL_MAX_TOTAL_PAGES`（默认 3000），
  到上限会停下并保留待爬队列，下次运行继续。
- 原始 HTML 会缓存到 `data/raw_html/`，**命中缓存的页面不会再发网络请求**。
  因此「加深爬取层级」「改进抽取规则」都接近零成本。
- 要彻底重来：`uv run python scripts/crawl.py --fresh`。

> ⚠️ **改了 `crawl_max_depth` 必须加 `--fresh`。** 深度只在「处理某个页面时」
> 才会用来决定是否展开它的链接；已访问过的页面不会重新处理，所以直接续爬
> 会沿用旧的浅层结果，新的深度根本不生效。

> ⚠️ **不要在中途就急着建索引。** 爬到一半建成索引，会导致知识库过小、
> 大量问题检索不到——这是实践中很容易踩的坑。

运行 `crawl.py` 时注意日志里的进度行：

```
进度：本次已处理 240 页 / 累计 240 页 / 待爬 1873 个 / 落盘文本 186 篇
```

确认爬取真正跑完（日志出现 `爬取完成`）后再执行 `build_index.py`。

构建索引时嵌入计算在 CPU 上进行，耗时取决于语料规模；
进度每 128 个片段落盘一次，**中断后重跑会自动续算**（语料变更时自动失效重算）。

#### 正文抽取：为什么必须剥离导航（重要）

校园门户每个页面顶部都挂着一段**几乎一模一样的导航菜单**。如果只是粗暴地
按标签删掉 HTML 再取文本，这段菜单就会被当成「正文」写进语料——结果是
索引里塞满「首页 学校概况 院系设置 …」这类零信息量的样板文字，检索时
频繁命中导航块而非真正的内容页，答案自然又空又不准。

现在的抽取策略（`crawler/spider.py`）按精度从高到低：

1. **优先取正文容器**：博达（VSB）CMS 的 `#vsb_content` / `.v_news_content`，
   以及常见的 `.article-content`、`#content` 等，命中即直接取该容器文本；
2. **回退到 `<body>`**：先按 `class/id` 词元删掉导航、页脚、面包屑、侧栏等容器，
   再取剩余文本。

> ⚠️ **坑：不要把 `<form>` 加进必删标签。** 博达 CMS 的 `#vsb_content`
> 恰恰嵌套在 `<form>` 内部，删掉 `form` 会把正文连坐删除，抽取结果只剩
> 侧边栏和「最新内容」列表。另外正文容器必须在删除 nav/header **之前**
> 查找，否则正文可能被其父级容器一并删掉。

同时 `loader.py` 会**按内容指纹去重**：同一页面常因多个 URL 别名被落盘成
多份完全相同的文本，去重后避免重复内容稀释检索结果。

#### 附件（PDF）下载：URL 里没有后缀

培养方案、学生手册、考试安排这类关键信息常常是**附件**。注意校内用的是
博达 CMS，附件链接长这样：

```
/system/_content/download.jsp?urltype=news.DownloadAttachUrl&owner=…&wbfileid=…
```

**URL 里既没有文件名也没有 `.pdf`**，所以：

- 只按后缀名判断「是不是附件」会把校内附件**整批漏掉**（本项目早期版本
  下载到的附件数就是 0）；
- 下载时也不能拿 URL 路径当文件名——所有附件的 path 都是
  `system/_content/download.jsp`，会全部撞成同一个名字，结果只存下 1 个。

现在的做法：按特征串识别间接下载入口；下载后优先从 `Content-Disposition`
取服务端给出的真实文件名，用 `wbfileid` 之类的 ID 兜底保证唯一，并用文件头
魔数（`%PDF-` 等）校验内容确实是附件而不是登录页/错误页。

#### 微信公众号自动采集：绕过 403 的主要手段

13 个学院/部门站点被 WAF 按**来源网络**拦成 403（见 8.1），而同样内容大量
存在于各学院的微信公众号里——教师介绍、专业介绍、招生录取方案、学院动态，
覆盖面往往比官网更全。`crawl_wechat.py` 走「搜狗微信搜索 → 还原跳转 →
抓 `mp.weixin.qq.com` 正文」，全自动：

```bash
uv run python scripts/crawl_wechat.py --dry-run          # 只搜索并列出，验证通道可用
uv run python scripts/crawl_wechat.py                     # 用内置词表（覆盖被 403 的单位）
uv run python scripts/crawl_wechat.py --keywords "湘潭大学 化学学院,湘潭大学 商学院"
uv run python scripts/crawl_wechat.py --pages 3 --max-per-keyword 20 --delay 3
```

落盘到 `data/raw_html/wechat_*.html`，**与 `crawl.py` 缓存格式一致**，
所以 `extract_texts.py` + `build_index.py` 无需任何改动即可消费。清单写在
`data/wechat/manifest.jsonl`（标题/公众号/日期/字数），同时充当断点续爬状态，
重复文章自动跳过。

三个实现细节值得记下来：

1. **真实地址是 JS 拼出来的。** 搜狗结果的 `href` 是 `/link?url=…`，
   页面上没有静态 `Location`，而是 `url += '片段'` 逐段拼接后
   `window.location.replace(url)`，还夹着 `@` 占位符要 replace 掉。
   必须取完所有 `url += '…'` 片段再拼接还原，否则拿到的是空壳。
2. **公众号名不在 `<a class="account">` 里。** 实测结构是
   `<div class="s-p"><span class="all-time-y2">公众号</span>…`，class 名会变，
   只能按结构取。日期多数由 `document.write(timeConvert('时间戳'))` 写入，
   旧条目则是纯文本 `2024-04-03`，两种都要认。
3. **必须压缩后再落盘。** 微信文章页实测 **3.5MB**，正文只有几 KB，其余全是
   内联脚本与样式。落盘前剥掉 `script`/`style`/注释可压到约 100KB（30 倍），
   而正文位于 `#js_content` 的静态 HTML 中，**不影响后续重抽**。

> ⚠️ 搜狗的 `/link` 是**临时**跳转链，几小时后失效。脚本设计为「搜到即抓」，
> 只持久化抓到的正文，不持久化链接。

> ⚠️ **搜狗限流有两种形态，只防一种会静默丢数据。** 除了「返回 200 再塞验证码」，
> 它还会**直接回 `403`/`429`**。早期版本只认前者，于是限流被当成「该关键词没有
> 结果」，主循环一路把剩余关键词全烧掉——**实测一次连丢 9 个关键词**（且恰好是
> 最需要的材料/计算机/机械/土木/外国语/历史文化/商学院/公管/马克思）。
> 现在 `_RATE_LIMIT_CODES` 把 403/429/503 一并纳入反爬判定，并由
> `search_with_retry()` 按 `--cooldown`（默认 45s）**2 倍退避重试**
> `--retries`（默认 3）次——限流是短时的，实测几秒到几十秒即恢复。
> 另外关键词之间也补了 `--search-delay` 间隔：配额在首页填满时会直接跳出
> 翻页循环，若不在关键词交界处等待，两个搜索请求会紧挨着发出去，正是限流诱因。
> 无论如何，断点续爬清单能保证**重跑只补没抓到的，不重复抓**（`失败`/`过短`
> 的条目会重试，已落盘的按标题与链接跳过）。

#### 官网之外的渠道（手工灌入）

`crawl.py` 只爬 `xtu.edu.cn` 域，`crawl_wechat.py` 覆盖公众号。知乎/贴吧等
其余站外内容，以及手头已有的 PDF/Word 文件、浏览器另存的页面，用
`ingest_urls.py` 灌入：

```bash
# URL 列表（每行一个，# 开头为注释）
uv run python scripts/ingest_urls.py urls.txt --source wechat

# 直接给 URL
uv run python scripts/ingest_urls.py https://mp.weixin.qq.com/s/xxxx --source wechat

# 本地文件（pdf/docx/md/txt）
uv run python scripts/ingest_urls.py ./docs/*.pdf --source handbook

# 浏览器「另存为」的整页 HTML
uv run python scripts/ingest_urls.py ./saved_pages/*.html --source portal
```

灌完同样跑 `build_index.py` 重建索引。抽取器已内置微信公众号正文容器
（`#js_content`）、知乎（`.RichText`）、贴吧（`.d_post_content`）等选择器，
支持 `.pdf`、`.docx`（含表格）、`.html`、`.md`、`.txt`。

### 3.3 迁移到另一台主机（离线拷贝索引）

索引构建要在 CPU 上做嵌入，语料上千篇时耗时可观。如果只是想在另一台机器上
跑起来，**不必重新构建**，把已经建好的产物拷过去即可。

需要拷两样东西，缺一不可：

| 内容 | 路径 | 大小（参考） | 说明 |
|---|---|---|---|
| 向量索引 | `data/index/` | 约 27 MB | `index.faiss` + `index.pkl` + `index_meta.json` |
| 嵌入模型 | `models/bge-base-zh/` | 约 391 MB | 查询时要靠它把问题转成向量，**缺了服务起不来** |

另外目标主机还需要 `.env`（`ZHIPU_API_KEY`）与项目代码（`git clone` 即可）。

```bash
# 在当前主机打包（排除 .build_progress.npz —— 那是断点续算的临时缓存）
cd data/index && zip -r ../../xtuagent_index_$(date +%Y%m%d).zip . -x ".build_progress.npz"

# 在目标主机解压回原位
cd <项目根>/data && unzip xtuagent_index_YYYYMMDD.zip -d index/
```

然后在目标主机上启动验证：

```bash
uv run python scripts/serve.py
curl -s http://127.0.0.1:8000/api/stats      # 向量数/文档数应与源主机一致
```

> ⚠️ 注意 `index_meta.json` 里的 `embedding_model` 记录的是**构建时的绝对路径**。
> 目标主机若把模型放在别处，要同步改 `.env` 里的 `EMBEDDING_MODEL`（默认
> `BAAI/bge-base-zh`，本项目在本机改成了本地目录）。**索引与模型必须配对**——
> 不同模型编码出的向量与查询向量不在同一空间，检索结果会失真。另外
> `config.py` 强制离线加载（`TRANSFORMERS_OFFLINE=1` / `local_files_only=True`），
> 所以模型**必须已经落盘**，不能指望运行时自动下载。

---

## 4. 前端使用

两种问答模式，可在左侧栏切换：

| 模式 | 接口 | 特点 |
|---|---|---|
| 普通模式（默认） | `POST /api/ask/stream` | SSE 流式输出，逐字渲染，展示来源卡片 |
| 工具增强模式 | `POST /api/agent` | ReAct Agent 自主调用检索/规定原文工具，适合多步查找，非流式 |

**两种模式都不是「只能答知识库」**，它们都能回答知识库以外的問題，
区别在于**怎么发现「答不了」**：

| 提问类型 | 普通模式 | 工具增强模式 |
|---|---|---|
| 知识库有的学业问题 | 检索 → 生成（`mode=kb`） | 模型自行选工具检索（`mode=kb`） |
| 寒暄 / 常识 / 闲聊 | 直接作答 | 直接作答 |
| 知识库没有、网上有 | 模型判定答不了 → **服务端强制**转联网（`mode=web`） | 模型**自行**改用 `web_lookup` |
| 都没有 | 拒答话术（`mode=refused`） | 拒答话术 |

这个差别是理解工具增强模式的关键：

- **普通模式靠「服务端判定」**：模型说答不了 → `is_unanswerable()` 认出来 →
  服务端**强制**转联网。不指望模型主动做事，所以稳定。
- **工具增强模式靠「模型自主」**：联不联网由模型自己决定调不调 `web_lookup`。
  模型一偷懒就漏掉——所以 `agent.py` 补了一道确定性兜底：
  模型说答不了却又没读过任何网络资料时，服务端直接替它调一次联网（`_web_fallback`）。

5 个工具，4 个查知识库、1 个联网：

| 工具 | 作用 |
|---|---|
| `search_knowledge_base` | 开放式检索学业规定 / 政策 / 办事指南 |
| `course_schedule` / `exam_schedule` | 查课程安排 / 期末考试安排 |
| `regulation_lookup` | 查规定原文条款 |
| `web_lookup` | 知识库没有依据时到互联网检索（受 `WEB_FALLBACK_ENABLED` 控制） |

### 4.1 改前端后为什么能立刻生效（不用清缓存）

前端是最容易被浏览器缓存坑到的地方：代码明明改了，页面跑的还是旧版，
于是浪费大量时间怀疑自己改错了地方。本项目做了两层处理，**改完重启即可，不需要用户清缓存**：

1. **静态资源不缓存**：`serving/app.py` 的 `NoCacheStaticFiles` 给 `/js`、`/css`
   等资源加 `Cache-Control: no-cache, must-revalidate`，每次请求都会向服务端校验
   （未变更走 304，开销可忽略）。
2. **资源地址带内容指纹**：`GET /` 由路由返回 `index.html`，把里面的
   `{{BUILD}}` 替换成 `static_build_id()` 算出的构建号（`web/js/*.js` + `web/css/*.css`
   内容的 sha1 前 10 位）。文件一变，URL 就从 `/js/app.js?v=53ed71975a` 变成新值，
   浏览器眼里是全新的文件，必然重新拉取。

**所以：改前端 → 重启服务 → 普通刷新即可。** 版本号是自动算的，不需要手工维护。

排查"页面到底跑的是哪一版"：

```bash
curl http://127.0.0.1:8000/api/build     # 服务端当前构建号
```

页面上也会显示：左下角技术说明栏的 `构建 xxxxxxxxxx`，
或在浏览器控制台执行 `window.__XtuAgentBuild`。两者不一致就说明页面是旧的，刷新即可。

前端 Markdown 渲染有独立回归测试（注意是 node 脚本，`pytest` 不会收集）：

```bash
node tests/render_markdown.test.js
```

---

## 5. 检索质量与相关性阈值

对外暴露的 `score` 是**真实余弦相似度**（文档与查询向量均做 L2 归一化，
由 `cos = 1 - L2²/2` 还原），取值 `[0, 1]`。

### 5.1 阈值默认是关闭的（重要）

`retriever_score_threshold` **默认 `0.0`，即不过滤**，检索稳定返回 top_k 条。
这是刻意的设计，原因如下：

BGE 中文模型的余弦分数被压缩在很窄的区间内，且区间位置随语料大幅变化。
实测某校园语料：

| 类型 | top-1 相似度区间 |
|---|---|
| 语料内问题 | 0.619 ~ 0.696 |
| 语料外问题 | 0.646 ~ 0.669 |

**两组完全重叠**——此时任何固定阈值要么把结果全滤掉（查不出东西），
要么形同虚设。所以"该不该拒答"交给提示词判断，而不是靠一个拍脑袋的数字。

判出来答不了之后怎么办，见 [5.5 联网兜底](#55-答不出来怎么办联网兜底)。

### 5.2 先标定，再决定是否开启

不要凭直觉设阈值。运行标定工具，它会打印两组分布并直接给出结论：

```bash
uv run python scripts/eval_qa.py --calibrate   # 无需 API Key
```

- 输出「两组可分离」→ 按建议值设 `RETRIEVER_SCORE_THRESHOLD`。
- 输出「分布重叠」→ **保持 0**，优先去改善语料覆盖度（重新爬取）。

### 5.3 其他检索参数

```ini
# .env
RETRIEVER_TOP_K=5              # 送入模型的片段数
RETRIEVER_MAX_PER_SOURCE=3     # 同一文件最多几个片段，防止上下文被单篇长文占满
RETRIEVER_SCORE_THRESHOLD=0    # 0 = 关闭过滤
```

> 注意：相似度只用于排序和（可选的）过滤，**不会写进提示词**。
> 分数量纲不可靠时把它交给模型，反而会诱导模型把不相关内容当成依据。

### 5.4 分块必须按 token 计数（不要改回字符）

`CHUNK_SIZE` / `CHUNK_OVERLAP` 的**单位是 token，不是字符**。
`ingest/splitter.py` 的长度函数走嵌入模型自己的 tokenizer。

这条规矩是有代价换来的。bge-base-zh 的 `max_position_embeddings` 是 **512**，
而**中文大约 1 字 1 token**。早先的 `length_function=len` 数的是字符，于是
`CHUNK_SIZE=800` 实际是 800 个 token，**超上限 56%**，超出部分被 tokenizer
**静默截断**。实测（1821 篇语料）：

| | 按字符切 800（旧） | 按 token 切 500（现） |
|---|---|---|
| 文本块数 | 4507 | 6006 |
| token 中位 / 最大 | 532 / 773 | 448 / 502 |
| **超出 512 上限的块** | **52.0%** | **0.00%** |
| 被截断块平均编码内容 | 仅前 77.6% | 100% |

> ⚠️ **这个坑为什么难发现**：截断只影响「**召回**」，不影响「**生成**」——
> 完整块文本仍存在索引里，命中后会原样交给 LLM。所以答案看起来总是对的，
> 症状是**有些问题干脆检索不到**（答案恰好落在块的尾部 22% 时），
> 而不是答案出错。transformers 其实一直在打警告
> （`Token indices sequence length is longer than the specified maximum
> sequence length (710 > 512)`），只是淹没在日志里。

`CHUNK_SIZE` 取 **500** 而不是 512，是给 tokenizer 自动补的 `[CLS]`/`[SEP]`
留 2 个 token 余量。`tests/test_splitter.py` 里有回归用例把
「默认长度函数必须是 token 计数」和「任何块都不得超过模型上限」钉死了——
改回 `len` 会立刻挂测试。

> 顺带一提：如果将来需要**多语言**检索或**超长上下文**（8192），才值得换成
> `BAAI/bge-m3`（568M 参数、2.2GB、1024 维）。当前语料中文字符占 69.2%、
> 非中文文档仅 4.1%，纯为多语言换模型是划不来的。

### 5.5 答不出来怎么办：联网兜底

知识库只覆盖校内站点。对「今天长沙天气」这类域外问题它必然没有依据；
对「校园网 VPN 怎么用」也常只有《校园网简介》而没有操作步骤。
改造前这类问题一律回一句「根据现有知识库暂未找到相关信息」——
这不是回答，是把服务窗口关了。

现在的策略是**两层**：

```
用户提问
   ↓
向量检索 top_k 条（阈值 0，不过滤）
   ↓
拼进提示词 → 模型作答
   ↓
模型判定「答不了」───────────────┐
   ↓ 答得了                    ↓
 直接返回（mode=kb）      调智谱搜索 API 联网检索
   ↓                          ↓
 不生任何搜索费用         用网络资料重新作答（mode=web）
                              ↓ 联网也失败
                          退回原拒答话术（mode=refused）
```

**触发点是「模型的判定」，不是相似度分数。** 阈值分布重叠（见 5.1），
分数分不开域内域外；唯一可靠的信号是模型读完之后明确说答不了。
判定逻辑在 `rag/pipeline.py:is_unanswerable`，**刻意做得保守**：
只在回答开头一段范围内匹配拒答特征，宁可漏判（保持原拒答行为），
不可误判（把本来答得好好的内容换成网络答案）。

判定分三路，从严到宽：

1. **照抄句指纹**：提示词里直接给出「答不了就照抄这句」，我们记住那句话的指纹
   （压掉空白后的前 12 字符）再比对。最精确，且天然支持任意语言。
2. **中文正则**：沿用原有设计，应付模型改写（窗口 30 字）。
3. **外文正则**：英/日/韩/俄的常见拒答说法，窗口放宽到 200 字符——
   同样一句「没找到信息」，中文十来个字说完，英文要六十个字符，
   窗口给短了英文拒答根本扫不到。

否定词必须和「被否定的对象」同时出现才算拒答：`cannot find` 不能单独成判据，
否则 `You cannot submit the form late` 这种正常回答会被误判。

几个必须知道的实现细节：

1. **免责声明由服务端强制注入**，不交给模型。实测 `glm-4-flash` 会漏掉
   提示词里"先说明来源性质"这条要求，直接开讲天气——而"这是不是学校官方口径"
   恰恰是用户最不能搞错的事。凡是不能容忍被跳过的内容，都不该指望模型自觉。

2. **流式输出有 24 字符的扣留窗口**（`_HOLDBACK_CHARS`）。拒答话术只有 37 个字，
   如果直接流给用户，会看到「暂未找到…」闪一下再消失。扣留前 24 字，
   在窗口内识别出拒答就一个字都不发，直接转联网。正常回答只多等一个 chunk。
   若拒答较长、已越过窗口才判定，则发 `reset` 事件让前端清空重来。

3. **联网失败绝不放大故障**：检索失败、生成失败、超时，任意一环出问题都
   降级回原来的拒答话术，行为与改造前一致。

4. **成本**：`search_std` 约 0.01 元/次，且只在真拒答时触发。
   `tests/test_pipeline_webfallback.py::test_ask_normal_answer_never_triggers_web_search`
   把"能答的问题不联网"钉成了回归测试。

配置：

```ini
# .env
WEB_FALLBACK_ENABLED=true         # 关掉即完全回到改造前的行为
WEB_SEARCH_ENGINE=search_std      # search_std 约 0.01 元/次；search_pro 更贵更全
WEB_SEARCH_COUNT=5                # 送入模型的网络资料条数
WEB_SEARCH_TIMEOUT=20
```

> 坑：该搜索 API 对模糊问题会做**多意图拆解**（响应里的 `search_intent`），
> 于是 `count=5` 也可能返回 10 条。`web_search()` 会按配置硬截断——
> 10 条 × 800 字足以把模型淹掉。

验证：

```bash
uv run pytest tests/test_websearch.py tests/test_pipeline_webfallback.py -q

# 真实浏览器端到端（需先启动服务）
uv run python scripts/e2e_web_fallback.py --base http://127.0.0.1:8000
```

第二个脚本用真实 Chromium 打字、点击，并劫持 `fetch` 把服务端下发的每一条 SSE
事件原样落盘到 `proof/e2e_web_fallback.json`，同时截全页图——
结论基于真实网络往返，不接受"单元测试过了所以线上一定对"。

### 5.6 用什么语言回答

**按提问语言自动决定，不让用户选**。识别在 `core/lang.py`，按字符集粗判
（假名 → 谚文 → 汉字 → 西里尔 → 拉丁），零依赖、纯离线。
顺序不能改：日文必然夹假名，而汉字区间会同时命中中日文，
先判汉字会把日文提问认成中文。

语言只影响**话术**，不影响检索与是否联网：

| 受影响 | 不受影响 |
| --- | --- |
| 提示词里「用哪种语言回答」 | 检索条数、相似度阈值 |
| 拒答时要求模型照抄的那句话 | 是否触发联网兜底 |
| 兜底答复 / 联网免责声明 / 联网状态提示 | 成本底线（答得出来就不联网） |

所有话术集中在 `core/messages.py`，五种语言各一份。
**提示词与拒答识别共用同一份文案**——这两处一旦分家，
改了话术忘了改识别规则，外文提问的联网兜底就会静默失效
（真实事故：语言改成自动识别后，模型开始用英文拒答，
而当时的拒答正则是清一色中文，于是英文提问永远走不到联网兜底）。

前端的分工是：**回答气泡里的文字跟着回答语言走**（来源卡片、结尾提示、报错），
**应用外壳保持中文**（按钮、占位符、状态条）。后者在提问之前就存在，
跟回答语言无关，跟着变只会让人觉得界面在乱跳。

**已知限制：外文提问的检索质量明显更差。** 语料和嵌入模型都是中文的，
实测英文问「What is the GPA warning threshold?」返回的 top-5 全是无关页面
（一篇报纸引用、英文首页导航、期刊引用、密码重置通知），而同一问题用中文问
能命中《考核成绩与绩点》。分数还都在 0.75–0.78，毫无区分度——
这正是 §5.1「阈值必须设 0」的成因。
后果：外文提问更容易走联网兜底（每次约 0.01 元），且网络答案未必是本校口径。
想让外文提问真正命中知识库，得换多语言嵌入模型（§5.4 提到的 `BAAI/bge-m3`）。

---

## 6. API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 服务状态（`idle`/`loading`/`ready`/`error`）与索引条目数 |
| GET | `/api/stats` | 模型、索引元数据、当前检索参数、联网兜底开关 |
| POST | `/api/ask` | 同步问答，返回答案 + 来源 + 耗时 |
| POST | `/api/ask/stream` | SSE 流式问答 |
| POST | `/api/agent` | 工具增强问答（Agent），返回答案 + 命中来源 |

`/api/ask` 与 `/api/agent` 的响应都带 `mode` 字段：

| mode | 含义 |
|---|---|
| `kb` | 由本地知识库回答 |
| `web` | 知识库未收录，联网检索后回答（回答开头已强制加免责声明） |
| `refused` | 知识库与网络都没找到依据，返回拒答话术 |

来源 `sources[]` 统一结构，靠 `kind` 区分两类：

```json
[
  { "file": "学分制管理规定.txt", "snippet": "…", "score": 0.83, "kind": "kb" },
  { "file": "天气网", "title": "长沙天气", "url": "http://changsha.tianqi.com/",
    "snippet": "…", "score": 0.0, "kind": "web", "publish_date": "2026/09/30" }
]
```

> `kind="web"` 时 `score` 无意义（网络结果没有余弦相似度），前端不展示；
> `file` 是站点名，`url` 可点击跳转。

`/api/ask/stream` 的事件序列：

| 事件 | 时机 | 载荷 |
|---|---|---|
| `sources` | 一开始，知识库检索完成 | `sources`（`kind="kb"`） |
| `delta` | 流式正文 | `text` |
| `reset` | 已下发过正文才判定为拒答 | 无（前端清空当前气泡） |
| `web_sources` | 联网检索完成 | `sources`（`kind="web"`） |
| `status` | 联网兜底过程提示 | `text` |
| `done` | 结束 | `elapsed_ms`、`mode`、`sources` |
| `error` | 生成失败 | `message` |

正常问答只有 `sources → delta… → done`；走联网兜底时才会出现
`reset`/`web_sources`/`status`。

请求体：

```json
{ "question": "如何办理休学手续？", "language": "中文", "top_k": 5 }
```

> `top_k` 为可选项，用于单次请求覆盖默认检索条数。

---

## 7. 自检与评估

```bash
uv run python scripts/doctor.py                # 全链路自检（配置/数据/索引/模型/LLM）
uv run python scripts/doctor.py --skip-llm     # 无 Key 时跳过 LLM 连通性测试

uv run python scripts/eval_qa.py               # 完整评估：检索 + 生成 + 拒答
uv run python scripts/eval_qa.py --skip-llm    # 仅评估检索层，无需 API Key
uv run python scripts/eval_qa.py --calibrate   # 标定相关性阈值（域内/域外分档）

uv run python scripts/diagnose_coverage.py     # 站点覆盖诊断（哪些站没抓到、为什么）

uv run pytest tests/ -v                        # 单元测试
```

`eval_qa.py` 输出三类指标：

- **Recall@k**：预期要点是否出现在检索上下文中（不依赖大模型，可复现）
- **回答命中率**：生成答案是否覆盖预期要点且未触发兜底
- **正确拒答率**：语料外问题是否被如实拒答（衡量幻觉倾向）

---

## 8. 已知限制

### 8.1 站点覆盖：**域名过时**才是主因（不是网络）

跑 `uv run python scripts/diagnose_coverage.py` 可以逐个看到某站点是哪种情况。

> ⚠️ **重要修正（2026-09-30）**：本节早先的结论是「13 个学院站被 WAF 按来源网络
> 拦截，只能换出口网络」。**这个结论是错的。** 真正的原因是那些学院**换了域名**，
> 我们配的是学校改版后遗留的旧域名。同一时刻、同一出口 IP 下，用现行域名访问
> 返回的是 200 正常页面。

| 学院 | 旧域名（403） | 现行域名（200 ✅） |
|---|---|---|
| 计算机学院 | `cie` | **`jwxy`**（计算机学院·网络空间安全学院） |
| 商学院 | `bs` | **`business`** |
| 哲学与历史文化学院 | `lsxy` | **`bqsy`**（碧泉书院） |
| 化学学院 | `chem` | **`hxxy`** |
| 材料科学与工程学院 | `mse` | **`clxy`** |
| 外国语学院 | `fl` | **`wgyxy`** |
| 物理与光电工程学院 | `phy` | **`wlxy`** |
| 化工学院 | — | **`hgxy`** |
| 土木工程学院 | `cee` | **`cem`** |
| 公共管理学院 | `gggl` | **`glxy`** |
| 机械工程与力学学院 | `mee` | **`jxgc`** |
| 马克思主义学院 | `marx` | **`mks`** |
| 法学学部 | — | **`fxxb`** |
| 学生工作部 | `xgb` / `xsc` | **`xtuxgb`** |
| 招生网 | `zhaosheng` | **`zs`** |
| 兴湘学院 | — | **`xxxy`** |

这些域名已全部写进 `src/xtuagent/crawler/site_config.py`（新旧域名都保留：新域名
在当前网络可用，旧域名留给校园网出口的用户）。

**怎么找现行域名——不要靠猜。** 研究生院和招生网都有「各学院工作方案链接汇总」
页面，逐条列出了全部学院的官网地址，例如
<https://yjsc.xtu.edu.cn/info/1084/10250.htm>。各学院的微信公众号文章里也常带
自己的官网链接。学校改版后旧域名往往还解析、但被 WAF 挡掉，这就制造了
「像是网络问题」的假象。

仍抓不到的只有两类：

| 现象 | 站点 | 原因 | 办法 |
|---|---|---|---|
| `JS骨架` | 图书馆 `lib`（200，仅 2.5KB） | 正文由前端异步加载 | `--render` 渲染 |
| `403` | `xgb` / `xsc` / `tsg` / `library` 等 | 暂未找到可达的替代域名 | 走 `crawl_wechat.py`，或人工另存页面后 `ingest_urls.py` |

```bash
# 用无头浏览器兜底抓 JS 页（需已安装 playwright）
uv run python scripts/crawl.py --render --fresh
```

> `--render` 只对「普通请求拿不到正文」的页面才启用浏览器，正常页面仍走
> 轻量请求。注意渲染慢一到两个数量级，整站重爬耗时会明显拉长。
> **注意：`--render` 解决不了真 403。** 它换的是客户端，不是网络出口；
> 若站点确实是按出口网段判定的，渲染只会同样拿到 403 页。


#### 旧域名确实是 403 —— 但换个域名就通了

先把观测留下（这些实测都真实发生过）：

| 试法 | 结果 |
|---|---|
| 走沙箱代理出口 `202.105.108.205` | 旧域名全 `403` |
| **清空全部代理变量、直连以太网卡 `192.168.26.39`** | 旧域名**仍全 `403`** |
| 加假 Cookie / `Referer` / `X-Forwarded-For` / 换 HTTP 明文 / 完整浏览器头 | 全部 `403` |
| 换无线（WLAN `192.168.119.135`）或换有线（以太网） | 出口 IP 相同，无差别 |
| 同一时刻访问 `www` / `jwc` / `math` / `art` / `law` / `hjzy` | `200`，25–69 KB 正常 HTML |

> 有人会以为是「客户端特征不够像浏览器」，于是去加请求头或上 `--render`
> —— 都没用：加请求头、加 Cookie、换 HTTP 明文一律还是 403。
>
> 但**当时由此推出的「只能换网络」是错的**。真正的原因是这些 vhost 属于
> 学校改版前的旧域名，服务端只对校内网段放行；而学院**现在有全新的域名**，
> 对外完全正常。也就是说：**问题在「访问了哪个域名」，不在「从哪个 IP 访问」。**

所以正确的动作是：

1. **换域名，不换网络。** 去研究生院/招生网的「各学院工作方案链接汇总」页
   取现行域名（如 <https://yjsc.xtu.edu.cn/info/1084/10250.htm>），
   或从学院公众号文章里的官网链接拿。改 `site_config.py` 后直接增量爬：
   ```bash
   uv run python scripts/diagnose_coverage.py          # 看现状
   uv run python scripts/crawl.py --max-total-pages 6000
   ```
   不需要 `--fresh`：待爬队列为空时新种子会自动入队，已访问的 URL 不会重爬。
2. **实在找不到替代域名的站点**（如 `xgb`/`tsg`），再走公众号通道
   `uv run python scripts/crawl_wechat.py`（见 3.2）或人工另存页面后
   `ingest_urls.py` 灌入。

#### 登录态能解决 403 吗？——不能

**403 是服务端对该域名的访问策略，不是认证问题。** 实测（同一客户端、同一时刻）：

| 站点 | 结果 |
|---|---|
| `www` / `jwc` / `math` / `art` / `law` / `hjzy` | `200`，25–69 KB 正常 HTML |
| `bs` / `cie` / `portal` / `my`（旧域名） | `403`，固定 199 字节 Apache 默认错误页 |

并逐一验证过：加**假 Cookie**、加 `Referer`、加 `X-Forwarded-For`（伪装校内 IP）、
换 HTTP 明文、带完整浏览器请求头——**全部仍是 403**。

**所以用信息门户的登录态并不能解锁这些旧域名**，也不必为此提供账号密码。
遇到 403 请先按上一条**换域名**。

#### 确实需要登录时才用 Cookie

若某些页面**必须登录**（而不是被网络策略挡住），可从浏览器导出 Cookie
存成本地文件后交给爬虫。**凭据只留在你本机**，工具不收集也不上传：

```bash
# 用扩展「Get cookies.txt」或浏览器开发者工具导出，存成 .cookies.txt
uv run python scripts/crawl.py --cookie-file .\my.cookies.txt
uv run python scripts/ingest_urls.py urls.txt --cookie-file .\my.cookies.txt
```

支持 Netscape（`cookies.txt`）与 JSON（Playwright `storageState`）两种格式。
`*.cookies.txt` 已加入 `.gitignore`。**Cookie 等同于登录态，泄露即可被冒充**，
用完请及时删除。

#### 最稳的兜底：把页面「另存为」再灌入

对付抓不到的站点（403 / JS 渲染 / 需登录），最可靠的办法是人工保存：

1. 浏览器里打开目标页面，`Ctrl+S` 保存为「网页，仅 HTML」；
2. 把保存下来的 `.html` 放进一个目录；
3. 灌入语料：

```bash
uv run python scripts/ingest_urls.py .\saved_pages\*.html --source portal
uv run python scripts/build_index.py
```

抽取器会照常剥离导航、取正文容器，所以另存的整页 HTML 直接可用，
不需要手工清理。

### 8.2 附件（PDF）受验证码保护

校内附件的下载入口是博达 CMS 的 `download.jsp?urltype=news.DownloadAttachUrl`，
访问后会返回一个**要求填写图形验证码**的中间页，而不是文件本身。

**本项目不会去绕过验证码**（这既是站点的反自动化措施，也涉及合规风险）。
可行做法：手工下载需要的文件（培养方案、学生手册等），放到一个目录后灌入：

```bash
uv run python scripts/ingest_urls.py .\manual_docs\*.pdf --source handbook
```

### 8.3 磁盘加密（DLP）会让下载的文件无法读取

若本机装有企业文档加密客户端（文件头形如 `%TSD-Header` / `%ESD-Header`），
**下载的附件会被自动加密**，外部程序读到的是密文：

- 表现为 `pdfplumber` 报 `No /Root object! - Is this really a PDF?`；
- 这不是代码缺陷，无法在应用层解决，需在系统层面放行或换设备处理。

### 8.4 其他

- **`.doc`（老版二进制 Word）无法解析**：请另存为 `.docx` 或 `.pdf`；
  `.docx` 现已支持（含表格，因为培养方案的关键信息常在表格里）。
- **Agent 模式对「答不了」的判断不如普通模式可靠**：普通模式由服务端强制转联网，
  工具增强模式则依赖模型自己决定调不调 `web_lookup`。实测还会出现
  「检索到一堆无关内容就直接罗列出来收尾」的情况（问「摩尔定律」搜到的是人名文章）。
  已在提示词里写明「无关检索结果等同于没找到」，并把「我无法直接回答…」
  纳入拒答识别以触发确定性兜底；但受限于 `glm-4-flash` 的能力，仍不保证每次都走到联网。
  **需要稳定兜底的场景请用普通模式。**
- **`langchain-community` 已进入维护期**：`FAISS` 与 `ChatZhipuAI` 后续需迁移到
  独立集成包，升级时留意弃用警告。
