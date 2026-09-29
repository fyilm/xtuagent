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
> 只持久化抓到的正文，不持久化链接。另外搜狗有速率限制，连续高频请求会触发
> 验证码；脚本内置 `--delay` 与验证码熔断，命中后**立即停止并保留已抓内容**。

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

---

## 4. 前端使用

两种问答模式，可在左侧栏切换：

| 模式 | 接口 | 特点 |
|---|---|---|
| 普通模式（默认） | `POST /api/ask/stream` | SSE 流式输出，逐字渲染，展示来源卡片 |
| 工具增强模式 | `POST /api/agent` | ReAct Agent 自主调用检索/规定原文工具，适合多步查找，非流式 |

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

---

## 6. API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 服务状态（`idle`/`loading`/`ready`/`error`）与索引条目数 |
| GET | `/api/stats` | 模型、索引元数据、当前检索参数 |
| POST | `/api/ask` | 同步问答，返回答案 + 来源 + 耗时 |
| POST | `/api/ask/stream` | SSE 流式问答（`sources` → `delta` → `done`） |
| POST | `/api/agent` | 工具增强问答（Agent），返回答案 + 命中来源 |

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

### 8.1 站点覆盖：26 个目标站点里有 14 个抓不到

跑 `uv run python scripts/diagnose_coverage.py` 可以逐个看到某站点是哪种情况。
当前实测分布：**12 个正常 / 13 个被 WAF 拦截 / 1 个是 JS 骨架页**。

| 现象 | 占比 | 原因 | 办法 |
|---|---|---|---|
| `OK` | 12/26 | 正常 | — |
| `WAF-403` | 13/26 | 站点按**来源网络**放行，非浏览器/非校内网段一律 `403` | 换到出口 IP 不同的网络重跑；**或直接走 `crawl_wechat.py`（推荐，见 3.2）** |
| `JS骨架` | 1/26（图书馆） | 返回 200 但 HTML 仅 2.5KB，正文由前端异步加载 | 只能 `--render` 渲染 |

被 403 的 13 个站点里包含**计算机、土木、机械、材料、物理、化学、外国语、
历史文化、商学院、公共管理、马克思主义、学工部、招生网**——数量不少，
值得优先解决。

```bash
# 用无头浏览器兜底抓 JS 页 / 403 站点（需已安装 playwright）
uv run python scripts/crawl.py --render --fresh
```

> `--render` 只对「普通请求拿不到正文」的页面才启用浏览器，正常页面仍走
> 轻量请求。注意渲染慢一到两个数量级，整站重爬耗时会明显拉长。
> **注意：`--render` 解决不了 403。** 它换的是客户端，不是网络出口；
> 而 403 是按出口网段判定的，渲染只会同样拿到 403 页。

#### 403 已确认是出口网段问题，不是本机能改的

实测结论（**已排除所有本机因素**）：

| 试法 | 结果 |
|---|---|
| 走沙箱代理出口 `202.105.108.205` | 13 站全 `403` |
| **清空全部代理变量、直连以太网卡 `192.168.26.39`** | 13 站**仍全 `403`** |
| 加假 Cookie / `Referer` / `X-Forwarded-For` / 换 HTTP 明文 / 完整浏览器头 | 全部 `403` |
| 换无线（WLAN `192.168.119.135`）或换有线（以太网） | 出口 IP 相同，无差别 |
| 同一时刻访问 `www` / `jwc` / `math` / `art` / `law` / `hjzy` | `200`，25–69 KB 正常 HTML |

第三行是本项目**最容易被误解**的一点：有人会以为是「客户端特征不够像浏览器」，
于是去加请求头或上 `--render`——都没用。决定性证据是最后一行：**同一秒、同一
客户端**，一部分站点 200、一部分站点 403。区别只可能是服务端看到的**来源 IP**。

同时注意**磁盘加密（DLP，见 8.3）与此无关**：DLP 是落盘层，管的是文件写进硬盘
后的形态，完全不碰 HTTP 请求，改不了出口 IP。

因此正确的动作只有两个：

1. **换一个出口 IP 不同的网络**（移动热点、别的办公网、VPN）——到新环境的
   **第一件事**是先看出口 IP 有没有变：
   ```bash
   curl -s ipinfo.io/ip        # 若仍是 202.105.108.205，说明是同一个出口，换了也白换
   ```
   变了的话跑 `uv run python scripts/diagnose_coverage.py`，`WAF-403` 消失即说明通了，
   然后 `uv run python scripts/crawl.py --fresh` 全量重爬。
2. **不等网络，直接走公众号通道**：`uv run python scripts/crawl_wechat.py`
   （见 3.2）。这条通道**在当前网络下已验证可用**，实测已抓到教师介绍、
   招生录取方案、学院通知等官网拿不到的内容。

#### 登录态能解决 403 吗？——不能

**403 是网络层策略，不是认证问题。** 实测（同一客户端、同一时刻）：

| 站点 | 结果 |
|---|---|
| `www` / `jwc` / `math` / `art` / `law` / `hjzy` | `200`，25–69 KB 正常 HTML |
| `bs` / `cie` / `portal` / `my` | `403`，固定 199 字节 Apache 默认错误页 |

并逐一验证过：加**假 Cookie**、加 `Referer`、加 `X-Forwarded-For`（伪装校内 IP）、
换 HTTP 明文、带完整浏览器请求头——**全部仍是 403**。

也就是说：这些 vhost 是按**来源网络**放行的（多半只允许校园网），
跟「有没有登录」无关。**所以用信息门户的登录态并不能解锁这 13 个站点**，
也不必为此提供账号密码。

> 如果你能在浏览器里正常打开 `https://bs.xtu.edu.cn`，那么**同一台机器上**
> 直接跑 `crawl.py` 就能抓到——因为限制的是网络位置，不是身份。
> 这也意味着 `--render` 大概率也用不上。

#### 先确认请求从哪个 IP 出去

既然 403 取决于来源网络，排查第一步就是看**出口公网 IP**：

```bash
uv run python scripts/diagnose_coverage.py
```

它会先打印代理环境变量与出口 IP，再逐个列出站点状态。实测对比：

| 运行方式 | 出口 IP | `bs.xtu.edu.cn` |
|---|---|---|
| Agent 沙箱内（走托管代理） | `202.105.108.205`（广东电信 / GCP 转发） | `403` |
| 强制不用代理 | 同上（沙箱强制走托管网络） | `403` |

**结论**：在受管代理/云环境里，出口 IP 由代理决定，**换本地网络不会改变它**。
要真正验证「你的网络能不能直连」，请在**你自己的终端**（不经过该代理）里
跑一次上面的命令：如果那些站点变成 `OK`，就可以直接本地终端跑
`uv run python scripts/crawl.py --fresh` 全量抓取。

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
- **Agent 模式**：工具全部基于本地知识库，知识库未收录的内容会明确返回「未找到」。
- **Agent 模式**：工具全部基于本地知识库，知识库未收录的内容会明确返回「未找到」。
- **`langchain-community` 已进入维护期**：`FAISS` 与 `ChatZhipuAI` 后续需迁移到
  独立集成包，升级时留意弃用警告。
