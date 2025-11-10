# -*- coding: utf-8 -*-
"""
Paju Embedding Tuner
--------------------
Recursive + Late Hybrid Chunking + BM25/FAISS Hybrid Retrieval
Grid-search style "tuning" for embedding-related hyperparameters (chunk size, late alpha, tokenizer).
Designed for administrative (public sector) documents.
"""

import os
import re
import json
import glob
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable

import numpy as np

# 3rd-party (install: pip install sentence-transformers faiss-cpu rank-bm25 pypdf)
from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi

try:
    import faiss  # faiss-cpu
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

from sentence_transformers import SentenceTransformer

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("paju-embed-tuner")

# -----------------------------
# Utils
# -----------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def postprocess_pdf_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"-\s*\n\s*", "", s)  # hyphen line breaks
    s = re.sub(r"[ \t]*\n+[ \t]*", " ", s)  # newlines -> space
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def recursive_chunk(text: str, max_chars: int = 1200) -> List[str]:
    """
    Paragraph -> sentence -> clause style recursive chunking (char-based).
    Adjust 'max_chars' to approximate token limits (e.g., ~1200 chars ~ 600-800 tokens).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        # sentence split (fixed-width lookbehind only)
        sentences = re.split(r'(?<=[.!?])\s+|(?<=다\.)\s+', para)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 > max_chars:
                if buf.strip():
                    chunks.append(buf.strip())
                buf = s
            else:
                buf = (buf + " " + s).strip()
        if buf.strip():
            chunks.append(buf.strip())

    # clause-level refinement
    refined: List[str] = []
    for c in chunks:
        if len(c) > max_chars:
            parts = re.split(r'(,|;|:)\s*', c)
            buf = ""
            for p in parts:
                if len(buf) + len(p) + 1 > max_chars:
                    if buf.strip():
                        refined.append(buf.strip())
                    buf = p
                else:
                    buf = (buf + " " + p).strip()
            if buf.strip():
                refined.append(buf.strip())
        else:
            refined.append(c)

    # keep only reasonable chunks
    return [normalize_text(x) for x in refined if len(x) >= 50]

def ko_tokenize(text: str) -> List[str]:
    """Lightweight tokenizer: whitespace + bi-gram fallback for Korean."""
    ws = text.lower().split()
    bigrams = [w[i:i+2] for w in ws for i in range(max(1, len(w)-1))]
    return ws + bigrams

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class DocRecord:
    doc_id: str
    title: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

# -----------------------------
# Corpus builder with Late Chunking
# -----------------------------
class TextCorpus:
    def __init__(self, embed_model_name: str):
        self.embed_model_name = embed_model_name
        self.embed_model = SentenceTransformer(embed_model_name)
        self.records: List[DocRecord] = []
        self.chunks: List[str] = []
        self.chunk_meta: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None  # (N, D)

    def add_pdf(self, file_path: str, max_chars: int = 1200, late_alpha: float = 0.5) -> int:
        """
        PDF -> text -> recursive chunking -> late hybrid embedding.
        late_alpha: weight for document-level embedding (0..1).
        """
        reader = PdfReader(file_path)
        raw = "".join(postprocess_pdf_text(page.extract_text() or "") for page in reader.pages)
        text = normalize_text(raw)
        doc_id = Path(file_path).stem

        # chunks
        chunks = recursive_chunk(text, max_chars=max_chars)
        if not chunks:
            logger.warning("No chunks produced for %s", file_path)
            self.records.append(DocRecord(doc_id=doc_id, title=doc_id, text=text, meta={"empty": True}))
            return 0

        # doc embedding
        doc_emb = self.embed_model.encode([text], normalize_embeddings=True)[0]

        # chunk embeddings (late hybrid)
        chunk_embs: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        for i, ch in enumerate(chunks):
            ch_emb = self.embed_model.encode([ch], normalize_embeddings=True)[0]
            hybrid = (1.0 - late_alpha) * ch_emb + late_alpha * doc_emb
            norm = np.linalg.norm(hybrid)
            if norm > 0:
                hybrid = hybrid / norm
            chunk_embs.append(hybrid.astype("float32"))
            metas.append({"doc_id": doc_id, "chunk_id": i, "type": "pdf", "path": file_path})

        self.records.append(DocRecord(doc_id=doc_id, title=doc_id, text=text, meta={"path": file_path}))
        self.chunks.extend(chunks)
        self.chunk_meta.extend(metas)

        M = np.vstack(chunk_embs).astype("float32")
        self.embeddings = M if self.embeddings is None else np.vstack([self.embeddings, M])
        return len(chunks)

# -----------------------------
# Retriever
# -----------------------------
class HybridRetriever:
    def __init__(
        self,
        corpus: TextCorpus,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        use_rrf: bool = True,
        rrf_k: float = 60.0
    ):
        if not corpus.chunks:
            raise ValueError("Corpus is empty. Load PDFs first.")
        self.corpus = corpus
        self._tok = tokenizer or ko_tokenize
        tokenized = [self._tok(c) for c in corpus.chunks]
        self.bm25 = BM25Okapi(tokenized)

        self.has_faiss = _FAISS_OK
        if self.has_faiss:
            self.index = faiss.IndexFlatIP(corpus.embeddings.shape[1])
            self.index.add(corpus.embeddings.astype("float32"))
        else:
            logger.warning("FAISS not available. Using BM25 only.")

        self.use_rrf = use_rrf
        self.rrf_k = rrf_k

    @staticmethod
    def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
        # smaller rank = better
        return (-scores).argsort().argsort().astype("float32")

    def _rrf(self, bm25_rank: np.ndarray, faiss_rank: Optional[np.ndarray]) -> np.ndarray:
        k = self.rrf_k
        rrf = 1.0 / (k + bm25_rank)
        if faiss_rank is not None:
            rrf += 1.0 / (k + faiss_rank)
        return rrf

    def search(self, query: str, k: int = 5) -> List[int]:
        # BM25
        token_q = self._tok(query)
        bm25_scores = self.bm25.get_scores(token_q)
        bm25_rank = self._ranks_from_scores(bm25_scores)

        faiss_rank = None
        if self.has_faiss:
            q_emb = self.corpus.embed_model.encode([query], normalize_embeddings=True).astype("float32")
            D, I = self.index.search(q_emb, min(max(50, k*8), len(self.corpus.chunks)))
            faiss_scores = np.zeros_like(bm25_scores, dtype="float32")
            faiss_scores[I[0]] = D[0]
            faiss_rank = self._ranks_from_scores(faiss_scores)

        if self.use_rrf:
            fused = self._rrf(bm25_rank, faiss_rank)
            cand = fused.argsort()[:k]
        else:
            # min-max fusion as fallback
            b = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-8)
            if self.has_faiss:
                f = (faiss_scores - faiss_scores.min()) / (faiss_scores.ptp() + 1e-8)
            else:
                f = np.zeros_like(b)
            hybrid = 0.5 * b + 0.5 * f
            cand = np.argsort(-hybrid)[:k]
        return cand.tolist()

    def build_context(self, query: str, k: int = 5, max_chars: int = 2400) -> str:
        idxs = self.search(query, k=k)
        lines = []
        total = 0
        for idx in idxs:
            meta = self.corpus.chunk_meta[idx]
            tag = f"[{meta.get('type','pdf')}/{meta.get('doc_id','doc')}#{meta.get('chunk_id',idx)}]"
            line = f"{tag} {self.corpus.chunks[idx]}"
            total += len(line)
            if total > max_chars:
                break
            lines.append(line)
        return "\n\n".join(lines)

# -----------------------------
# Evaluation helpers
# -----------------------------
def load_eval_queries(path: str) -> List[Dict[str, Any]]:
    """
    JSON lines or JSON file format:
    [
      {"query":"...", "relevant_keywords":["...","..."]},
      {"query":"...", "relevant_keywords":["..."]}
    ]
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def precision_at_k(texts: List[str], keywords: List[str]) -> float:
    """
    Simple proxy metric: a retrieved chunk is 'relevant' if it contains any keyword.
    You may replace with manual labels later.
    """
    if not texts or not keywords:
        return 0.0
    hits = 0
    for t in texts:
        if any(kw.lower() in t.lower() for kw in keywords):
            hits += 1
    return hits / len(texts)

# -----------------------------
# Main tuning routine
# -----------------------------
from pathlib import Path

def build_corpus_from_pdfs(pdf_dir: str,
                           embed_model: str,
                           max_chars: int,
                           late_alpha: float) -> TextCorpus:
    corpus = TextCorpus(embed_model_name=embed_model)
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_files:
        logger.warning("No PDF found in %s", pdf_dir)
    for p in pdf_files:
        n = corpus.add_pdf(p, max_chars=max_chars, late_alpha=late_alpha)
        logger.info("Loaded %s -> %d chunks", os.path.basename(p), n)
    logger.info("Corpus: %d docs / %d chunks", len(corpus.records), len(corpus.chunks))
    return corpus

def run_tuning(pdf_dir: str,
               eval_json: Optional[str],
               embed_model: str,
               grid_max_chars: List[int],
               grid_late_alpha: List[float],
               k: int,
               out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    queries = load_eval_queries(eval_json) if eval_json else []

    for max_chars in grid_max_chars:
        for late_alpha in grid_late_alpha:
            logger.info("=== Building corpus: max_chars=%d, late_alpha=%.2f ===", max_chars, late_alpha)
            corpus = build_corpus_from_pdfs(pdf_dir, embed_model, max_chars, late_alpha)
            retriever = HybridRetriever(corpus, tokenizer=ko_tokenize, use_rrf=True)

            # quick evaluation
            metrics = {"p_at_k": []}
            if queries:
                for item in queries:
                    q = item["query"]
                    keywords = item.get("relevant_keywords", [])
                    idxs = retriever.search(q, k=k)
                    texts = [corpus.chunks[i] for i in idxs]
                    p = precision_at_k(texts, keywords)
                    metrics["p_at_k"].append(p)

            avg_p = float(np.mean(metrics["p_at_k"])) if metrics["p_at_k"] else None
            row = {
                "max_chars": max_chars,
                "late_alpha": late_alpha,
                "k": k,
                "avg_precision_at_k": avg_p,
                "num_docs": len(corpus.records),
                "num_chunks": len(corpus.chunks)
            }
            results.append(row)
            logger.info("Result: %s", row)

            # save FAISS index and artifacts
            if _FAISS_OK and corpus.embeddings is not None and len(corpus.chunks) > 0:
                dim = corpus.embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(corpus.embeddings.astype("float32"))
                faiss_path = os.path.join(out_dir, f"faiss_{max_chars}_{int(late_alpha*100)}.index")
                faiss.write_index(index, faiss_path)

                # save corpus artifacts
                with open(os.path.join(out_dir, f"chunks_{max_chars}_{int(late_alpha*100)}.json"), "w", encoding="utf-8") as f:
                    json.dump(corpus.chunks, f, ensure_ascii=False, indent=2)
                with open(os.path.join(out_dir, f"chunk_meta_{max_chars}_{int(late_alpha*100)}.json"), "w", encoding="utf-8") as f:
                    json.dump(corpus.chunk_meta, f, ensure_ascii=False, indent=2)

    # save results
    res_path = os.path.join(out_dir, "tuning_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved tuning results -> %s", res_path)

def main():
    parser = argparse.ArgumentParser(description="Paju Embedding Tuner")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--embed_model", type=str, default="intfloat/multilingual-e5-large",
                        help="HF model name (e.g., intfloat/e5-large, intfloat/multilingual-e5-large)")
    parser.add_argument("--grid_max_chars", type=str, default="800,1200,1600",
                        help="Comma-separated values for recursive chunk size (char-based)")
    parser.add_argument("--grid_late_alpha", type=str, default="0.3,0.5,0.7",
                        help="Comma-separated values for late chunking alpha (0..1)")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval for evaluation")
    parser.add_argument("--eval_json", type=str, default="", help="Optional eval JSON with queries/keywords")
    parser.add_argument("--out_dir", type=str, default="./embed_tuning_out", help="Output directory")
    args = parser.parse_args()

    grid_max_chars = [int(x) for x in args.grid_max_chars.split(",") if x]
    grid_late_alpha = [float(x) for x in args.grid_late_alpha.split(",") if x]

    run_tuning(
        pdf_dir=args.pdf_dir,
        eval_json=args.eval_json if args.eval_json else None,
        embed_model=args.embed_model,
        grid_max_chars=grid_max_chars,
        grid_late_alpha=grid_late_alpha,
        k=args.k,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()
