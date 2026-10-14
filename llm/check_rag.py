import logging

logging.basicConfig(level=logging.INFO)

from llm import rag
from llm import tools as T

print("funcții în rag.py:", [n for n in dir(rag) if not n.startswith("_")])
print()
for rezultat in T.tool_cauta_manual("Edge-Ring", 3):
    print(rezultat["doc"], "| pagina:", rezultat["page"])
    print("   ", rezultat["text"][:200].replace("\n", " "))