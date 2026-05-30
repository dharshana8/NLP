# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for custom PDF data.

## Stack
- LangChain - PDF loading & text splitting
- FAISS - Vector database
- HuggingFace Embeddings - `all-MiniLM-L6-v2`

## Setup

```bash
pip install -r requirement.txt
```

## Usage

1. Place your PDF file in the `vijaygpt/` folder
2. Update `pdf_path` in `rag_pipeline.py`
3. Update `query` with your question
4. Run:

```bash
python rag_pipeline.py
```
