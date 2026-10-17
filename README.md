# Semiconductor Wafer Diagnostics & LLM Decision Agent

This project provides an automated inspection, classification, and diagnostics system for semiconductor silicon wafers based on the WM-811K benchmark dataset.

## Features

- **Visual Classification**: CNN model for detecting defect patterns (Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, and none).
- **RAG Technical Knowledge Retrieval**: FAISS semantic retrieval over process manufacturing manuals and standard operating procedures.
- **Root Cause & Drift Diagnostics**: Real-time lot drift alarm verification and equipment check actions grounded in domain knowledge.
- **Autonomous LLM Decision Agent**: ReAct reasoning loop with tool calling for structured technical diagnostics (`analyze(wafer_json)`).

For detailed setup and module documentation, see [`llm/README.md`](llm/README.md).