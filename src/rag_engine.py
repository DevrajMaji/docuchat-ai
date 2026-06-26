"""
rag_engine.py
─────────────
Orchestrates the full RAG pipeline:
  1. Retrieve relevant chunks via VectorStore
  2. Build a structured prompt with context
  3. Call Groq LLM API
  4. Return answer + source metadata
"""

import os
import requests
from typing import List, Dict, Any
from src.vector_store import VectorStore


SYSTEM_PROMPT = """You are DocuChat AI, an expert document analyst.
Your job is to answer the user's question based ONLY on the provided context chunks.

Rules:
- Be concise and accurate.
- If the answer is not in the context, say "I couldn't find this in the uploaded documents."
- Always cite which page(s) or document the information came from when possible.
- Never fabricate information.
"""


class RAGEngine:
    """
    Full RAG pipeline: retrieve → prompt → generate → return.
    """

    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        vector_store: VectorStore,
        groq_api_key: str,
        model_name: str = "llama3-8b-8192",
        top_k: int = 3,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        self.vs = vector_store
        self.api_key = groq_api_key
        self.model_name = model_name
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Return top-k relevant chunks with metadata."""
        results = self.vs.search(query, top_k=self.top_k)
        retrieved = []
        for chunk, score in results:
            retrieved.append({
                "text": chunk.text,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", "?"),
                "score": round(score, 4),
            })
        return retrieved

    # ── Prompt building ───────────────────────────────────────────────────────
    def build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        context_str = ""
        for i, c in enumerate(context_chunks, 1):
            context_str += (
                f"\n[Context {i} | Source: {c['source']} | Page {c['page']} "
                f"| Relevance: {c['score']}]\n{c['text']}\n"
            )

        return f"""Use the following document excerpts to answer the question.

{context_str}

Question: {query}

Answer:"""

    # ── LLM call ──────────────────────────────────────────────────────────────
    def call_groq(self, user_message: str) -> str:
        """Call Groq chat completions API and return text response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(
                self.GROQ_URL, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                return "❌ Invalid Groq API key. Please check your key."
            elif response.status_code == 429:
                return "❌ Groq rate limit hit. Please wait a moment and retry."
            else:
                return f"❌ Groq API error: {str(e)}"
        except Exception as e:
            return f"❌ Request failed: {str(e)}"

    # ── Main answer method ────────────────────────────────────────────────────
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Full pipeline:
          query → retrieve → build_prompt → call_groq → return dict

        Returns:
            {
                "answer": str,
                "sources": list[str],   # unique "filename (p.N)" strings
                "chunks": list[dict],   # raw retrieved chunks
            }
        """
        # 1. Retrieve
        chunks = self.retrieve(query)

        if not chunks:
            return {
                "answer": "No relevant content found in the uploaded documents.",
                "sources": [],
                "chunks": [],
            }

        # 2. Build prompt
        prompt = self.build_prompt(query, chunks)

        # 3. Generate
        answer_text = self.call_groq(prompt)

        # 4. Deduplicate sources
        sources = list({
            f"{c['source']} (p.{c['page']})"
            for c in chunks
        })

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks": chunks,
        }
