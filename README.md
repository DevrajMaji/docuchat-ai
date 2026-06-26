# 🧠 DocuChat AI — End-to-End RAG Document Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-green)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3-orange)](https://groq.com)

An end-to-end RAG system — upload PDFs and chat with them using LLaMA3 via Groq.

## 🏗️ Architecture

```
PDF Upload → PyPDF2 Extraction → Recursive Chunking
    → all-MiniLM-L6-v2 Embeddings → FAISS Index
    → Query → Cosine Similarity → Top-K Context
    → Groq LLaMA3 → Answer + Source Attribution
```

## 📁 Structure

```
├── app.py                    # Streamlit UI
├── Dockerfile                # HF Spaces deployment
├── requirements.txt
└── src/
    ├── document_processor.py # PDF extraction + chunking
    ├── vector_store.py       # FAISS + embeddings
    └── rag_engine.py         # RAG + Groq API
```

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🐳 Docker

```bash
docker build -t docuchat-ai .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key docuchat-ai
```

## 🤗 HF Spaces Deploy

1. New Space → SDK: **Docker**
2. Upload all files
3. Settings → Secrets → add `GROQ_API_KEY`
4. Done ✅

Free Groq key: [console.groq.com](https://console.groq.com)

## DocuChat Ai is live

LINK : https://huggingface.co/spaces/Devraj12344/docuchat-ai

## 🧑‍💻 Author

**Devraj Maji** — AI & ML, Brainware University
