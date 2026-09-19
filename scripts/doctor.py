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
    check("智谱 API Key 已配置", bool(settings.zhipu_api_key), f"模型: {settings.zhipu_model}")

    # 2. 目录与数据
    settings.ensure_dirs()
    texts = list(settings.texts_dir.glob("*.txt"))
    check("文本数据目录", len(texts) > 0, f"{len(texts)} 个文本文件（{settings.texts_dir}）")

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
            check("嵌入模型加载", True, f"bge-base-zh，耗时 {elapsed:.1f}s，维度 {service.dimension}")
            embedding = service
        except Exception as exc:  # noqa: BLE001
            check("嵌入模型加载", False, str(exc))
    else:
        print(f"{WARN} 嵌入模型加载 — 已跳过")

    # 5. 索引加载 + 检索
    if embedding is not None and index_file.exists():
        try:
            from xtuagent.rag.vector_store import load_index

            start = time.time()
            vs = load_index()
            check("索引加载", True, f"{vs.count} 条向量，耗时 {time.time() - start:.1f}s")

            start = time.time()
            docs = vs.search("学分是什么", top_k=3)
            check(
                "检索测试",
                len(docs) > 0,
                f"命中 {len(docs)} 条，耗时 {time.time() - start:.2f}s，"
                f"首条: {docs[0].source if docs else '无'}",
            )
        except Exception as exc:  # noqa: BLE001
            check("索引加载", False, str(exc))
    elif not args.skip_model:
        print(f"{WARN} 索引加载 — 已跳过（模型未就绪）")

    # 6. LLM 连通性
    if not args.skip_llm:
        try:
            from xtuagent.rag.llm import create_llm

            start = time.time()
            llm = create_llm()
            reply = llm.invoke("回复两个字：正常")
            check("LLM 连通性", bool(reply), f"回复: {str(reply.content)[:20]}，耗时 {time.time() - start:.1f}s")
        except Exception as exc:  # noqa: BLE001
            check("LLM 连通性", False, str(exc))
    else:
        print(f"{WARN} LLM 连通性 — 已跳过")

    # 汇总
    print("=" * 56)
    failed = [r for r in results if r[0] == FAIL]
    print(f"自检完成：{len(results) - len(failed)}/{len(results)} 项通过")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
