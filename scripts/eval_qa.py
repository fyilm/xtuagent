#!/usr/bin/env python
"""问答准确率评估：8 题标准测试集。"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.core.logging import setup_logging
from xtuagent.rag.pipeline import RAGPipeline
from xtuagent.rag.vector_store import load_index

TEST_QUESTIONS = [
    {
        "question": "如何申请奖学金？",
        "category": "政策咨询",
        "expected_keywords": ["奖学金", "申请", "评审"],
    },
    {
        "question": "毕业论文的提交流程是什么？",
        "category": "学业流程",
        "expected_keywords": ["论文", "答辩", "提交"],
    },
    {
        "question": "MATLAB软件在哪里可以下载？",
        "category": "软件使用",
        "expected_keywords": ["MATLAB", "软件", "下载"],
    },
    {
        "question": "考试作弊会受到什么处分？",
        "category": "纪律规定",
        "expected_keywords": ["作弊", "处分", "警告"],
    },
    {
        "question": "如何办理休学手续？",
        "category": "学籍管理",
        "expected_keywords": ["休学", "办理", "手续"],
    },
    {
        "question": "图书馆的开放时间是什么？",
        "category": "校园设施",
        "expected_keywords": ["图书馆", "开放"],
    },
    {
        "question": "学分制是怎么回事？",
        "category": "学籍管理",
        "expected_keywords": ["学分", "学制"],
    },
    {
        "question": "校园网VPN怎么使用？",
        "category": "网络服务",
        "expected_keywords": ["VPN", "校园网"],
    },
]


def main() -> None:
    setup_logging()
    vs = load_index()
    rag = RAGPipeline(vector_store=vs)

    total = len(TEST_QUESTIONS)
    passed = 0

    for index, test in enumerate(TEST_QUESTIONS, 1):
        answer = rag.ask(test["question"]).text
        matched = any(kw.lower() in answer.lower() for kw in test["expected_keywords"])
        if matched:
            passed += 1
        status = "PASS" if matched else "FAIL"
        print(f"\n[{index}/{total}] {status} | {test['category']}")
        print(f"  Q: {test['question']}")
        print(f"  A: {answer[:180]}{'...' if len(answer) > 180 else ''}")

    accuracy = passed / total * 100
    print(f"\n{'=' * 60}")
    print(f"准确率: {passed}/{total} = {accuracy:.1f}%")
    sys.exit(0 if accuracy >= 85 else 1)


if __name__ == "__main__":
    main()
