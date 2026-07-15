import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.vector_store import load_vector_store
from xtuagent.rag_chain import RAGChain


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
        "expected_keywords": ["考试", "作弊", "处分"],
    },
    {
        "question": "如何办理休学手续？",
        "category": "学籍管理",
        "expected_keywords": ["休学", "办理", "手续"],
    },
    {
        "question": "What is the procedure for international student registration?",
        "category": "外籍学生",
        "expected_keywords": ["international", "student", "register"],
    },
    {
        "question": "图书馆的开放时间是什么？",
        "category": "校园设施",
        "expected_keywords": ["图书馆", "开放时间"],
    },
    {
        "question": "学分制是怎么回事？",
        "category": "学籍管理",
        "expected_keywords": ["学分", "学分制"],
    },
]


def evaluate():
    vs = load_vector_store()
    if vs is None:
        print("FAIL: vector store not found. Run ingest first.")
        return

    rag = RAGChain(vector_store=vs)

    total = len(TEST_QUESTIONS)
    passed = 0

    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        expected = test["expected_keywords"]

        answer = rag.ask(question)
        matched = any(kw.lower() in answer.lower() for kw in expected)

        status = "PASS" if matched else "FAIL"
        if matched:
            passed += 1

        print(f"\n[{i}/{total}] {status} | {test['category']}")
        print(f"  Q: {question}")
        print(f"  Expected keywords: {expected}")
        print(f"  A: {answer[:200]}{'...' if len(answer) > 200 else ''}")

    accuracy = passed / total * 100
    print(f"\n{'='*60}")
    print(f"Accuracy: {passed}/{total} = {accuracy:.1f}%")

    if accuracy >= 85:
        print("PASS: accuracy >= 85%")
    else:
        print("NOTE: accuracy below 85%, consider improving knowledge base or prompt")


if __name__ == "__main__":
    evaluate()
