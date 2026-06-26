"""
document_processor.py
─────────────────────
Handles PDF text extraction and recursive character text splitting.
"""

import re
import PyPDF2
import io
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    """Extract text from PDFs and split into overlapping chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── PDF extraction ────────────────────────────────────────────────────────
    def extract_text_from_pdf(self, file) -> List[dict]:
        """Return list of {page, text} dicts from an uploaded PDF file object."""
        pages = []
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                if text.strip():
                    pages.append({"page": i + 1, "text": text})
        except Exception as e:
            raise RuntimeError(f"PDF read error: {e}")
        return pages

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    # ── Chunking ──────────────────────────────────────────────────────────────
    def split_text(self, text: str, metadata: dict = None) -> List[Chunk]:
        """
        Recursive character splitter that tries to break on paragraphs,
        then sentences, then words, then characters.
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = self._recursive_split(text, separators)
        result = []
        for i, c in enumerate(chunks):
            meta = dict(metadata or {})
            meta["chunk_index"] = i
            result.append(Chunk(text=c, metadata=meta))
        return result

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []

        sep = separators[0]
        remaining_seps = separators[1:]

        if len(text) <= self.chunk_size:
            return [text]

        if sep == "":
            # Hard split by character
            return self._merge_splits(
                [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
            )

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    # Recurse with next separator
                    sub = self._recursive_split(part, remaining_seps)
                    chunks.extend(sub)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return self._merge_splits(chunks)

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge small splits and add overlap between chunks."""
        merged = []
        current = ""
        for s in splits:
            if not s.strip():
                continue
            if len(current) + len(s) + 1 <= self.chunk_size:
                current = (current + " " + s).strip()
            else:
                if current:
                    merged.append(current)
                current = s

        if current:
            merged.append(current)

        # Add overlap: prepend tail of previous chunk to next
        if self.chunk_overlap > 0 and len(merged) > 1:
            overlapped = [merged[0]]
            for i in range(1, len(merged)):
                prev_tail = merged[i-1][-self.chunk_overlap:]
                overlapped.append(prev_tail + " " + merged[i])
            return overlapped

        return merged

    # ── Main entry point ──────────────────────────────────────────────────────
    def process_pdf(self, file) -> List[Chunk]:
        """Full pipeline: extract → chunk → return Chunk list."""
        filename = getattr(file, "name", "unknown.pdf")
        pages = self.extract_text_from_pdf(file)
        all_chunks = []
        for page_data in pages:
            chunks = self.split_text(
                page_data["text"],
                metadata={"source": filename, "page": page_data["page"]}
            )
            all_chunks.extend(chunks)
        return all_chunks
