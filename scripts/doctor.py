#!/usr/bin/env python
"""系统自检：配置 / 目录 / 索引 / 模型 / LLM / 检索全链路诊断。"""

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.core.config import settings

PASS = "[通过]"
FAIL = "[失败]"
WARN = "[警告]"

results = []


def api_key_configured() -> bool:
    """复用 LLM 工厂的校验逻辑，避免两处占位符判断发生漂移。"""
    from xtuagent.rag.llm import validate_api_key

    try:
        validate_api_key()
        return True
    except Exception:  # noqa: BLE001
        return False


def check(name: str, ok: bool, detail: str = "", warning: bool = False) -> bool:
    tag = PASS if ok else (WARN if warning else FAIL)
    results.append((tag, name, detail))
    print(f"{tag} {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="XtuAgent 系统自检")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 连通性测试")
    parser.add_argument("--skip-model", action="store_true", help="跳过嵌入模型加载测试")
    args = parser.parse_args()

    print("=" * 56)
    print("  XtuAgent 系统自检")
    print("=" * 56)

    # 1. 配置
    key_ok = api_key_configured()
    check(
        "智谱 API Key 已配置",
        key_ok,
        f"模型: {settings.zhipu_model}"
        if key_ok
        else "请在 .env 中填入真实的 ZHIPU_API_KEY（当前为空或仍是占位符）",
    )

    # 2. 目录与数据
    settings.ensure_dirs()
    texts = sorted(
        p for p in settings.texts_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".txt", ".md", ".pdf")
    )
    check(
        "文本数据目录",
        len(texts) > 0,
        f"{len(texts)} 个可入库文件（{settings.texts_dir}）"
        if texts
        else f"未找到任何 .txt/.md/.pdf 文件（{settings.texts_dir}），请先运行 scripts/crawl.py",
    )

    # 3. 索引文件
    index_file = settings.index_dir / "index.faiss"
    meta_file = settings.index_dir / "index_meta.json"
    check("索引文件存在", index_file.exists(), str(index_file))
    meta = None
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        check(
            "索引元数据",
            True,
            f"模型 {meta.get('embedding_model')} / {meta.get('vector_count')} 向量 / "
            f"{meta.get('documents')} 文档 / 构建于 {meta.get('created_at')}",
        )
        check(
            "元数据与配置一致",
            meta.get("embedding_model") == settings.embedding_model,
            f"{meta.get('embedding_model')} vs {settings.embedding_model}",
        )
    else:
        check("索引元数据", False, "index_meta.json 不存在（可运行 build_index.py 重建）", warning=True)

    # 4. 嵌入模型
    embedding = None
    if not args.skip_model:
        try:
            from xtuagent.rag.embeddings import EmbeddingService

            start = time.time()
            service = EmbeddingService.instance()
            service.warmup()
            elapsed = time.time() - start
            check(
                "嵌入模型加载",
                True,
                f"{Path(settings.embedding_model).name}，耗时 {elapsed:.1f}s，维度 {service.dimension}",
            )
            embedding = service
        except Exception as exc:  # noqa: BLE001
            check("嵌入模型加载", False, str(exc))
    else:
        print(f"{WARN} 嵌入模型加载 — 已跳过")

    # 5. 索引加载 + 检索
    if not index_file.exists():
        print(f"{WARN} 索引加载 — 已跳过（索引尚未构建，请先运行 scripts/build_index.py）")
    elif embedding is None:
        print(f"{WARN} 索引加载 — 已跳过（嵌入模型未就绪）")
    else:
        try:
            from xtuagent.rag.vector_store import load_index

            start = time.time()
            vs = load_index()
            check("索引加载", True, f"{vs.count} 条向量，耗时 {time.time() - start:.1f}s")

            probe = "学分是什么"
            start = time.time()
            docs = vs.search(probe, top_k=3)
            cost = time.time() - start
            if docs:
                check(
                    "检索测试",
                    True,
                    f"「{probe}」命中 {len(docs)} 条，耗时 {cost:.2f}s，"
                    f"最高相关度 {docs[0].score:.2f}（{docs[0].source}）",
                )
            else:
                check(
                    "检索测试",
                    False,
                    f"「{probe}」无结果：全部低于阈值 {settings.retriever_score_threshold}。"
                    "若语料确实相关，可下调 RETRIEVER_SCORE_THRESHOLD",
                    warning=True,
                )
        except Exception as exc:  # noqa: BLE001
            check("索引加载", False, str(exc))

    # 6. LLM 连通性
    if args.skip_llm:
        print(f"{WARN} LLM 连通性 — 已跳过")
    elif not key_ok:
        print(f"{WARN} LLM 连通性 — 已跳过（API Key 未配置）")
    else:
        try:
            from xtuagent.rag.llm import create_llm

            start = time.time()
            llm = create_llm()
            reply = llm.invoke("回复两个字：正常")
            check("LLM 连通性", bool(reply), f"回复: {str(reply.content)[:20]}，耗时 {time.time() - start:.1f}s")
        except Exception as exc:  # noqa: BLE001
            check("LLM 连通性", False, str(exc))

    # 7. 检索参数（便于复现实验设置）
    print(
        f"{WARN} 检索参数 — top_k={settings.retriever_top_k}，"
        f"阈值={settings.retriever_score_threshold}，"
        f"同源上限={settings.retriever_max_per_source}，"
        f"chunk={settings.chunk_size}/{settings.chunk_overlap}"
    )

    # 汇总
    print("=" * 56)
    failed = [r for r in results if r[0] == FAIL]
    print(f"自检完成：{len(results) - len(failed)}/{len(results)} 项通过")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
