import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.config import config
from xtuagent.vector_store import load_vector_store

txts = list(Path("data/texts").glob("*.txt"))
vs = load_vector_store()

print("=== XtuAgent 系统状态 ===")
print(f"LLM: {config.llm_provider} / {config.zhipu_model}")
print(f"知识文档: {len(txts)} 个文本文件")
print(f"向量库: {vs.index.ntotal if vs else 0} 条记录")
print(f"嵌入模型: {config.embedding_model}")
print(f"向量库路径: {config.chroma_persist_dir}")
print(f"智谱API: {'已配置' if config.zhipu_api_key else '未配置'}")
print(f"千帆API: {'已配置' if config.qianfan_api_key else '未配置'}")
