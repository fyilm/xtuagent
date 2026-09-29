#!/usr/bin/env python
"""RAG 问答评估：阈值标定 + 分层度量（检索 / 生成 / 拒答）。

================================ 设计说明 ================================

早期版本只用「答案里是否出现预期关键词」当准确率，有两个问题：
1. 幻觉答案只要碰巧含关键词就会被判对；
2. 无法区分「检索没找到」和「检索找到了但模型没答好」。

更关键的是**阈值标定**：`retriever_score_threshold` 不该拍脑袋设定。
本脚本第 0 层会把「语料内问题」与「语料外问题」的 top-1 相似度分布都打出来，
并直接告诉你这两组是否可分离、建议阈值取多少。

经验事实：BGE 中文模型的余弦分数常被压缩在窄区间，若两组分布重叠，
**任何固定阈值都无效**，此时应保持阈值为 0（关闭过滤），
把拒答交给提示词，并优先去改善语料覆盖度。

用法：
    python scripts/eval_qa.py --calibrate    # 仅做阈值标定（无需 API Key）
    python scripts/eval_qa.py --skip-llm     # 检索层 + 标定，无需 API Key
    python scripts/eval_qa.py                # 完整评估（需要 ZHIPU_API_KEY）
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.core.config import settings  # noqa: E402
from xtuagent.core.logging import setup_logging  # noqa: E402
from xtuagent.rag.pipeline import FALLBACK_ANSWER, RAGPipeline  # noqa: E402
from xtuagent.rag.vector_store import load_index  # noqa: E402

# 语料内问题：应能被检索到并正确回答
TEST_QUESTIONS = [
    {"question": "如何申请奖学金？", "category": "政策咨询", "expected_keywords": ["奖学金", "申请", "评审"]},
    {"question": "毕业论文的提交流程是什么？", "category": "学业流程", "expected_keywords": ["论文", "答辩", "提交"]},
    {"question": "MATLAB软件在哪里可以下载？", "category": "软件使用", "expected_keywords": ["MATLAB", "软件", "下载"]},
    {"question": "考试作弊会受到什么处分？", "category": "纪律规定", "expected_keywords": ["作弊", "处分", "警告"]},
    {"question": "如何办理休学手续？", "category": "学籍管理", "expected_keywords": ["休学", "办理", "手续"]},
    {"question": "图书馆的开放时间是什么？", "category": "校园设施", "expected_keywords": ["图书馆", "开放"]},
    {"question": "学分制是怎么回事？", "category": "学籍管理", "expected_keywords": ["学分", "学制"]},
    {"question": "校园网VPN怎么使用？", "category": "网络服务", "expected_keywords": ["VPN", "校园网"]},
]

# 语料外问题：知识库不可能覆盖，正确行为是拒答而非编造
OUT_OF_DOMAIN_QUESTIONS = [
    "今天长沙的天气怎么样？",
    "请帮我写一段快速排序的代码。",
    "推荐几部最近的科幻电影。",
]


def _hit(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in keywords)


def _is_fallback(text: str) -> bool:
    return FALLBACK_ANSWER[:12] in text


def calibrate(vs) -> None:
    """第 0 层：打印两组问题的 top-1 相似度分布，判断阈值是否可设定。"""
    print("\n【第 0 层】阈值标定 —— 语料内 vs 语料外的 top-1 相似度分布\n")

    def top1(question: str) -> float:
        docs = vs.search(question, top_k=1, min_score=0.0)
        return docs[0].score if docs else 0.0

    inside = [(q["question"], top1(q["question"])) for q in TEST_QUESTIONS]
    outside = [(q, top1(q)) for q in OUT_OF_DOMAIN_QUESTIONS]

    print("  语料内问题：")
    for q, s in inside:
        print(f"    {s:.4f}  {q}")
    print("  语料外问题：")
    for q, s in outside:
        print(f"    {s:.4f}  {q}")

    in_min = min(s for _, s in inside)
    out_max = max(s for _, s in outside)
    print(f"\n  语料内最低 {in_min:.4f} | 语料外最高 {out_max:.4f} | 间隔 {in_min - out_max:+.4f}")

    if in_min - out_max > 0.02:
        suggest = round((in_min + out_max) / 2, 2)
        print(f"  结论：两组可分离，建议 RETRIEVER_SCORE_THRESHOLD={suggest}")
        print("        （注意：单次抽样，建议换一批问题复测确认稳定性）")
    else:
        print("  结论：两组分布重叠，**任何固定阈值都不可靠**。")
        print("        建议保持 RETRIEVER_SCORE_THRESHOLD=0（关闭过滤），")
        print("        由提示词负责拒答；根本解法是扩充语料覆盖度（重新爬取）。")


def main() -> None:
    parser = argparse.ArgumentParser(description="XtuAgent RAG 评估")
    parser.add_argument("--calibrate", action="store_true", help="仅做阈值标定")
    parser.add_argument("--skip-llm", action="store_true", help="跳过生成层评估（无需 API Key）")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖检索条数")
    parser.add_argument("--min-recall", type=float, default=0.6, help="检索命中率合格线")
    args = parser.parse_args()

    setup_logging(level="WARNING")
    vs = load_index()
    rag = RAGPipeline(vector_store=vs)

    print("=" * 66)
    print(f"  XtuAgent RAG 评估 · 索引 {vs.count} 条向量")
    print(f"  当前阈值 {settings.retriever_score_threshold} · top_k {args.top_k or settings.retriever_top_k}")
    print("=" * 66)

    calibrate(vs)
    if args.calibrate:
        return

    # ---------- 第一层：检索质量（不依赖大模型） ----------
    print("\n【第一层】检索质量 Recall@k —— 预期关键词是否出现在检索上下文中\n")
    retrieved_hits = 0
    latencies: list[float] = []

    for index, case in enumerate(TEST_QUESTIONS, 1):
        start = time.time()
        docs = rag.retrieve(case["question"], top_k=args.top_k)
        latencies.append(time.time() - start)

        context = "\n".join(d.content for d in docs)
        ok = bool(docs) and _hit(context, case["expected_keywords"])
        retrieved_hits += int(ok)
        best = f"{docs[0].score:.2f}" if docs else "—"
        print(
            f"  [{index}/{len(TEST_QUESTIONS)}] {'命中' if ok else '未命中'} | "
            f"{case['category']:<6} | 召回 {len(docs)} 条 | 最高分 {best} | {case['question']}"
        )

    recall = retrieved_hits / len(TEST_QUESTIONS)
    avg_retrieval_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0.0
    print(f"\n  检索命中率 Recall@k = {retrieved_hits}/{len(TEST_QUESTIONS)} = {recall:.1%}")
    print(f"  平均检索耗时 = {avg_retrieval_ms:.0f} ms")
    if recall < args.min_recall:
        print("  ⚠ 检索命中率偏低。优先检查语料覆盖：相关页面是否真的爬到了？")

    answer_accuracy = None
    refusal_rate = None

    # ---------- 第二层/第三层：生成质量与拒答行为 ----------
    if not args.skip_llm:
        print("\n【第二层】生成质量 —— 答案是否包含预期要点且未触发兜底\n")
        answer_hits = 0
        for index, case in enumerate(TEST_QUESTIONS, 1):
            answer = rag.ask(case["question"], top_k=args.top_k).text
            ok = _hit(answer, case["expected_keywords"]) and not _is_fallback(answer)
            answer_hits += int(ok)
            print(f"  [{index}/{len(TEST_QUESTIONS)}] {'正确' if ok else '存疑'} | {case['question']}")
            print(f"      {answer[:110]}{'...' if len(answer) > 110 else ''}")

        answer_accuracy = answer_hits / len(TEST_QUESTIONS)
        print(f"\n  回答命中率 = {answer_hits}/{len(TEST_QUESTIONS)} = {answer_accuracy:.1%}")

        print("\n【第三层】拒答行为 —— 语料外问题应如实拒答而非编造\n")
        refusals = 0
        for case in OUT_OF_DOMAIN_QUESTIONS:
            answer = rag.ask(case).text
            refused = _is_fallback(answer)
            refusals += int(refused)
            print(f"  [{'已拒答' if refused else '疑似编造'}] {case}")
            print(f"      {answer[:110]}{'...' if len(answer) > 110 else ''}")

        refusal_rate = refusals / len(OUT_OF_DOMAIN_QUESTIONS)
        print(f"\n  正确拒答率 = {refusals}/{len(OUT_OF_DOMAIN_QUESTIONS)} = {refusal_rate:.1%}")
    else:
        print("\n【第二层/第三层】已跳过（--skip-llm）")

    # ---------- 汇总 ----------
    print("\n" + "=" * 66)
    print("  汇总")
    print("=" * 66)
    print(f"  检索命中率 Recall@k : {recall:.1%}")
    if answer_accuracy is not None:
        print(f"  回答命中率          : {answer_accuracy:.1%}")
        print(f"  语料外正确拒答率    : {refusal_rate:.1%}（幻觉率 {1 - refusal_rate:.1%}）")
    print("=" * 66)

    passed = recall >= args.min_recall
    if answer_accuracy is not None:
        passed = passed and answer_accuracy >= 0.6 and refusal_rate >= 0.6
    print("  结论：" + ("达标" if passed else "未达标：先看第 0 层的语料覆盖与阈值标定结论"))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
