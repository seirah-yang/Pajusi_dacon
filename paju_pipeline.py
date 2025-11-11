# -*- coding: utf-8 -*-
"""
파주시 행정문서 자동화 E2E 파이프라인 (v10, single file)
- Ingest (PDF/DOCX/TXT/HTML/CSV/JSON)
- Chunking (Sentence + Recursive), PDF page mapping
- Embedding (Late Hybrid: chunk ⊕ doc)
- Hybrid Retrieval (BM25 + FAISS, MMR/RRF)
- Generation (LLM JSON schema; stub fallback)
- Validation (schema, reference coverage, UNIEVAL proxy/GPT)
- Visualization (bar/radar + JSON/CSV)
- Orchestrator (run_pipeline)

Dependencies (pip):
  pip install pypdf pdfplumber python-docx beautifulsoup4 lxml rank-bm25 sentence-transformers faiss-cpu pandas matplotlib pydantic
"""

from __future__ import annotations
import os, re, json, time, math, csv, hashlib, logging
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import List, Dict, Any, Optional, Tuple, Callable

# ------------------ Optional Imports ------------------
try:
    from PyPDF2 import PdfReader  # pypdf
except Exception:
    PdfReader = None
try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except Exception:
    _PDFPLUMBER_OK = False
try:
    from docx import Document
except Exception:
    Document = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
from rank_bm25 import BM25Okapi

# Embedding & FAISS
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import faiss  # faiss-cpu or faiss-gpu
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

# ------------------ Logging ------------------
logger = logging.getLogger("paju_admin_e2e")
logger.setLevel(logging.INFO)

# =========================
# Utils: Text normalization & chunking
# =========================
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _postprocess_pdf_text(s: str) -> str:
    """PDF 텍스트 후처리: 하이픈 줄바꿈/개행/공백 보정"""
    if not s:
        return ""
    s = re.sub(r"-\s*\n\s*", "", s)            # 하이픈 줄바꿈 제거
    s = re.sub(r"[ \t]*\n+[ \t]*", " ", s)     # 줄바꿈 → 공백
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def sentence_chunking(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """문장 단위 청킹 (한글 '다.' 포함)"""
    if not text:
        return []
    sents = re.split(r'(?<=[.!?])\s+|(?<=다\.)\s+', text)
    chunks, buf = [], []
    for sent in (x.strip() for x in sents if x.strip()):
        prospective = " ".join(buf + [sent]).strip()
        if len(prospective) <= chunk_size:
            buf.append(sent); continue
        if buf:
            chunk = normalize_text(" ".join(buf))
            if len(chunk) >= 50: chunks.append(chunk)
        # overlap 유지
        keep, kept = [], 0
        for s in reversed(buf):
            keep.append(s); kept += len(s) + 1
            if kept >= overlap: break
        buf = list(reversed(keep)) if keep else []
        if len(sent) >= chunk_size:
            chunks.append(normalize_text(sent)); buf = []
        else:
            buf.append(sent)
    if buf:
        chunk = normalize_text(" ".join(buf))
        if len(chunk) >= 50 and (not chunks or chunks[-1] != chunk):
            chunks.append(chunk)
    return chunks

def recursive_chunk(text: str, max_tokens: int = 500) -> List[str]:
    """문단 → 문장 → 절 단위 재귀 청킹 (의미 단절 최소화)"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for para in paragraphs:
        if len(para) <= max_tokens:
            chunks.append(para); continue
        sentences = re.split(r'(?<=[.!?]|다\.)\s+', para)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) > max_tokens:
                chunks.append(buf.strip()); buf = s
            else:
                buf += (" " + s) if buf else s
        if buf: chunks.append(buf.strip())
    # 절 단위 보정
    refined: List[str] = []
    for c in chunks:
        if len(c) > max_tokens:
            parts = re.split(r'(,|;|:)\s*', c)
            buf = ""
            for p in parts:
                if len(buf) + len(p) > max_tokens:
                    refined.append(buf.strip()); buf = p
                else:
                    buf += (" " + p) if buf else p
            if buf: refined.append(buf.strip())
        else:
            refined.append(c)
    return [normalize_text(x) for x in refined if len(x) > 30]

# =========================
# Data classes
# =========================
@dataclass
class DocRecord:
    doc_id: str
    title: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: int
    page: Optional[int] = None  # 1-based for PDFs
    text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

# =========================
# Tokenizer (Korean-friendly)
# =========================
def ko_tokenize(text: str) -> List[str]:
    ws = text.lower().split()
    bigrams = [w[i:i+2] for w in ws for i in range(max(1, len(w)-1))]
    return ws + bigrams

# =========================
# Corpus with multi-format loaders + page mapping + late embedding
# =========================
class TextCorpus:
    def __init__(self, embed_model_name: str = "intfloat/multilingual-e5-large"):
        self.records: List[DocRecord] = []
        self.chunks: List[str] = []
        self.chunk_meta: List[Dict[str, Any]] = []
        self.embed_model_name = embed_model_name
        self.embed_model = SentenceTransformer(embed_model_name) if SentenceTransformer else None
        self.embeddings: Optional[np.ndarray] = None

    # ---------- PDF ----------
    def _extract_pdf_pages(self, file_path: str) -> List[Tuple[int, str]]:
        pages: List[Tuple[int, str]] = []
        # 1) PyPDF2
        if PdfReader is not None:
            try:
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages, start=1):
                    t = page.extract_text() or ""
                    t = _postprocess_pdf_text(t)
                    if t: pages.append((i, t))
            except Exception as e:
                logger.warning("PdfReader failed for %s: %s", file_path, e)
        # 2) pdfplumber fallback
        if _PDFPLUMBER_OK and not pages:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for i, p in enumerate(pdf.pages, start=1):
                        t = p.extract_text() or ""
                        t = _postprocess_pdf_text(t)
                        if t: pages.append((i, t))
            except Exception as e:
                logger.warning("pdfplumber failed for %s: %s", file_path, e)
        return pages

    def add_pdf(self, file_path: str, chunk_size: int = 500, overlap: int = 50, recursive_first: bool = True) -> int:
        file_path = str(file_path)
        doc_id = Path(file_path).stem
        pages = self._extract_pdf_pages(file_path)
        if not pages:
            logger.error("No text extracted from PDF: %s", file_path)
            self.records.append(DocRecord(doc_id=doc_id, title=doc_id, text="", meta={"type":"pdf","path":file_path,"empty":True}))
            return 0
        full_text = normalize_text(" ".join(t for _, t in pages))
        self.records.append(DocRecord(doc_id=doc_id, title=doc_id, text=full_text, meta={"type":"pdf","path":file_path,"pages":len(pages)}))

        created = 0
        for page_num, page_text in pages:
            if recursive_first:
                base_chunks = recursive_chunk(page_text, max_tokens=chunk_size)
            else:
                base_chunks = sentence_chunking(page_text, chunk_size=chunk_size, overlap=overlap)
            for i, ch in enumerate(base_chunks):
                self.chunks.append(ch)
                self.chunk_meta.append({"doc_id": doc_id, "chunk_id": i, "type":"pdf", "page": page_num, "path": file_path})
                created += 1
        return created

    # ---------- DOCX ----------
    def add_docx(self, file_path: str, chunk_size: int = 500, overlap: int = 50, recursive_first: bool = True) -> int:
        if Document is None:
            logger.error("python-docx is not installed.")
            return 0
        doc_id = Path(file_path).stem
        doc = Document(file_path)
        text = normalize_text("\n".join(p.text for p in doc.paragraphs))
        self.records.append(DocRecord(doc_id=doc_id, title=doc_id, text=text, meta={"type":"docx","path":file_path}))
        base_chunks = recursive_chunk(text, chunk_size) if recursive_first else sentence_chunking(text, chunk_size, overlap)
        created = 0
        for i, ch in enumerate(base_chunks):
            self.chunks.append(ch)
            self.chunk_meta.append({"doc_id": doc_id, "chunk_id": i, "type":"docx", "path": file_path})
            created += 1
        return created

    # ---------- TXT ----------
    def add_text(self, file_path: str, chunk_size: int = 500, overlap: int = 50, recursive_first: bool = True) -> int:
        doc_id = Path(file_path).stem
        with open(file_path, "r", encoding="utf-8") as f:
            text = normalize_text(f.read())
        self.records.append(DocRecord(doc_id=doc_id, title=doc_id, text=text, meta={"type":"text","path":file_path}))
        base_chunks = recursive_chunk(text, chunk_size) if recursive_first else sentence_chunking(text, chunk_size, overlap)
        created = 0
        for i, ch in enumerate(base_chunks):
            self.chunks.append(ch)
            self.chunk_meta.append({"doc_id": doc_id, "chunk_id": i, "type":"text", "path": file_path})
            created += 1
        return created

    # ---------- HTML ----------
    def add_html(self, file_path: str, chunk_size: int = 500, overlap: int = 50, recursive_first: bool = True) -> int:
        if BeautifulSoup is None:
            logger.error("beautifulsoup4 is not installed.")
            return 0
        doc_id = Path(file_path).stem
        html = Path(file_path).read_text("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        # remove boilerplate
        for tag in soup(["script","style","noscript","header","footer","nav","aside"]):
            tag.decompose()
        text = normalize_text(soup.get_text("\n"))
        self.records.append(DocRecord(doc_id=doc_id, title=doc_id, text=text, meta={"type":"html","path":file_path}))
        base_chunks = recursive_chunk(text, chunk_size) if recursive_first else sentence_chunking(text, chunk_size, overlap)
        created = 0
        for i, ch in enumerate(base_chunks):
            self.chunks.append(ch)
            self.chunk_meta.append({"doc_id": doc_id, "chunk_id": i, "type":"html", "path": file_path})
            created += 1
        return created

    # ---------- CSV (summarize to text) ----------
    def add_csv(self, file_path: str, max_rows: int = 5, chunk_size: int = 700) -> int:
        import pandas as pd
        doc_id = Path(file_path).stem
        df = pd.read_csv(file_path)
        # simple textualization
        cols = ", ".join(df.columns[:12])
        head = df.head(max_rows).to_dict(orient="records")
        head_txt = "\n".join([json.dumps(r, ensure_ascii=False) for r in head])
        text = normalize_text(f"CSV 요약: 컬럼={cols}\n상위 {max_rows}행 샘플:\n{head_txt}")
        self.records.append(DocRecord(doc_id=doc_id, title=doc_id, text=text, meta={"type":"csv","path":file_path,"rows":len(df)}))
        chunks = recursive_chunk(text, max_tokens=chunk_size)
        created = 0
        for i, ch in enumerate(chunks):
            self.chunks.append(ch)
            self.chunk_meta.append({"doc_id": doc_id, "chunk_id": i, "type":"csv", "path": file_path})
            created += 1
        return created

    # ---------- JSON (summarize to text) ----------
    def add_json(self, file_path: str, chunk_size: int = 700) -> int:
        doc_id = Path(file_path).stem
        data = json.loads(Path(file_path).read_text("utf-8"))
        # flatten small JSONs; otherwise summarize keys
        def _flatten(d, prefix=""):
            items = []
            if isinstance(d, dict):
                for k, v in d.items():
                    items += _flatten(v, f"{prefix}{k}.")
            elif isinstance(d, list):
                for i, v in enumerate(d[:20]):
                    items += _flatten(v, f"{prefix}{i}.")
            else:
                items.append(f"{prefix[:-1]}={d}")
            return items
        flattened = _flatten(data)
        text = normalize_text("JSON 요약:\n" + "\n".join(flattened[:300]))
        self.records.append(DocRecord(doc_id=doc_id, title=doc_id, text=text, meta={"type":"json","path":file_path}))
        chunks = recursive_chunk(text, max_tokens=chunk_size)
        created = 0
        for i, ch in enumerate(chunks):
            self.chunks.append(ch)
            self.chunk_meta.append({"doc_id": doc_id, "chunk_id": i, "type":"json", "path": file_path})
            created += 1
        return created

    # ---------- Embedding (Late Hybrid) ----------
    def build_embeddings(self) -> int:
        """각 문서의 doc_emb와 chunk_emb를 평균하여 전역 문맥 주입"""
        if self.embed_model is None:
            logger.error("SentenceTransformer not available.")
            return 0
        if not self.chunks or not self.records:
            logger.warning("No data to embed.")
            return 0
        # map doc_id -> doc_text embedding
        doc_text_map = {r.doc_id: r.text for r in self.records}
        doc_ids = list(doc_text_map.keys())
        doc_embs = {}
        for did in doc_ids:
            v = self.embed_model.encode([doc_text_map[did]], normalize_embeddings=True)[0]
            doc_embs[did] = v
        emb_list = []
        for i, ch in enumerate(self.chunks):
            meta = self.chunk_meta[i]
            did = meta.get("doc_id")
            ch_emb = self.embed_model.encode([ch], normalize_embeddings=True)[0]
            if did in doc_embs:
                hybrid = (ch_emb + doc_embs[did]) / 2.0
            else:
                hybrid = ch_emb
            hybrid = hybrid / (np.linalg.norm(hybrid) + 1e-12)
            emb_list.append(hybrid.astype("float32"))
        self.embeddings = np.vstack(emb_list)
        return len(emb_list)

    # ---------- Index save/load ----------
    def save_index(self, out_dir: str) -> None:
        out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path / "embeddings.npz", E=self.embeddings if self.embeddings is not None else np.zeros((0,0),dtype="float32"))
        (out_path / "chunk_meta.json").write_text(json.dumps(self.chunk_meta, ensure_ascii=False, indent=2), "utf-8")
        (out_path / "records.json").write_text(json.dumps([r.__dict__ for r in self.records], ensure_ascii=False, indent=2), "utf-8")
        (out_path / "model.txt").write_text(self.embed_model_name, "utf-8")
        logger.info("Index saved to %s", out_path)

    def load_index(self, in_dir: str) -> None:
        in_path = Path(in_dir)
        if (in_path / "embeddings.npz").exists():
            self.embeddings = np.load(in_path / "embeddings.npz")["E"]
        if (in_path / "chunk_meta.json").exists():
            self.chunk_meta = json.loads((in_path / "chunk_meta.json").read_text("utf-8"))
        if (in_path / "records.json").exists():
            recs = json.loads((in_path / "records.json").read_text("utf-8"))
            self.records = [DocRecord(**r) for r in recs]
        if (in_path / "model.txt").exists() and SentenceTransformer:
            name = (in_path / "model.txt").read_text("utf-8").strip()
            if name != self.embed_model_name:
                self.embed_model_name = name
                self.embed_model = SentenceTransformer(name)
        logger.info("Index loaded from %s", in_path)

    def summary(self) -> Dict[str, int]:
        info = {"documents": len(self.records), "chunks": len(self.chunks)}
        logger.info(" Documents: %d |  Chunks: %d", info["documents"], info["chunks"])
        return info

# =========================
# Hybrid Retriever (BM25 + FAISS + MMR + RRF)
# =========================
class HybridRetriever:
    def __init__(self, corpus: TextCorpus, tokenizer: Optional[Callable[[str], List[str]]] = None, use_rrf: bool = True, faiss_weight: float = 0.5):
        if not corpus.chunks:
            raise ValueError("Error : corpus is empty. Load and chunk documents first.")
        self.corpus = corpus
        self._tok = tokenizer or ko_tokenize
        tokenized = [self._tok(c) for c in corpus.chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.use_rrf = use_rrf
        self.faiss_weight = faiss_weight
        self.has_faiss = _FAISS_OK and (corpus.embeddings is not None) and (corpus.embeddings.size > 0)
        if self.has_faiss:
            self.index = faiss.IndexFlatIP(corpus.embeddings.shape[1])
            self.index.add(corpus.embeddings.astype("float32"))
        else:
            logger.warning("FAISS disabled. Using BM25-only.")

    @staticmethod
    def _rrf(rank_list_len: int, bm25_ranks: np.ndarray, faiss_ranks: Optional[np.ndarray]) -> np.ndarray:
        k = 60.0
        N = rank_list_len
        rrf = np.zeros(N, dtype="float32")
        rrf += 1.0 / (k + bm25_ranks)
        if faiss_ranks is not None:
            rrf += 1.0 / (k + faiss_ranks)
        return rrf

    def _mmr(self, cand_idx: List[int], q_emb: Optional[np.ndarray], top_k: int, lambda_: float = 0.7) -> List[int]:
        if q_emb is None or top_k >= len(cand_idx) or self.corpus.embeddings is None:
            return cand_idx[:top_k]
        D = self.corpus.embeddings[cand_idx]
        rel = D @ q_emb.reshape(-1, 1)
        selected, rest = [], list(range(len(cand_idx)))
        while len(selected) < top_k and rest:
            if not selected:
                best = int(np.argmax(rel[rest])); selected.append(rest.pop(best)); continue
            S = D[[rest], :] @ D[[selected], :].T
            div = S.max(axis=1, keepdims=True)
            mmr = lambda_ * rel[rest] - (1 - lambda_) * div
            best = int(np.argmax(mmr)); selected.append(rest.pop(best))
        return [cand_idx[i] for i in selected]

    def build_context(self, query: str, k: int = 6, max_chars: int = 2400) -> str:
        token_q = self._tok(query)
        bm25_scores = self.bm25.get_scores(token_q)
        bm25_rank = (-bm25_scores).argsort().argsort().astype("float32")

        faiss_rank = None
        q_emb = None
        if self.has_faiss and self.corpus.embed_model is not None:
            q_emb = self.corpus.embed_model.encode([query], normalize_embeddings=True).astype("float32")[0]
            D, I = self.index.search(q_emb[None, :], min(k*10, len(self.corpus.chunks)))
            faiss_scores = np.zeros_like(bm25_scores, dtype="float32")
            faiss_scores[I[0]] = D[0]
            faiss_rank = (-faiss_scores).argsort().argsort().astype("float32")

        if self.use_rrf:
            fused = self._rrf(len(bm25_scores), bm25_rank, faiss_rank)
            cand_idx = fused.argsort()[: max(k*8, k)]
        else:
            b = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-8)
            if self.has_faiss:
                f = (faiss_scores - faiss_scores.min()) / (faiss_scores.ptp() + 1e-8)
            else:
                f = np.zeros_like(b)
            hybrid = (1.0 - self.faiss_weight) * b + self.faiss_weight * f
            cand_idx = np.argsort(-hybrid)[: max(k*8, k)]

        if self.has_faiss and q_emb is not None:
            top_idx = self._mmr(cand_idx.tolist(), q_emb, top_k=k, lambda_=0.7)
        else:
            top_idx = cand_idx[:k]

        ctx_lines, total = [], 0
        for idx in top_idx:
            meta = self.corpus.chunk_meta[idx] if idx < len(self.corpus.chunk_meta) else {}
            doc = meta.get("doc_id", "doc")
            typ = meta.get("type", "chunk")
            cid = meta.get("chunk_id", idx)
            page = meta.get("page")
            tag = f"[{typ}/{doc}#{cid}" + (f":p{page}]" if page else "]")
            line = f"{tag} {self.corpus.chunks[idx]}"
            total += len(line)
            if total > max_chars: break
            ctx_lines.append(line)
        return "\n\n".join(ctx_lines)

# =========================
# Generation schemas & functions
# =========================
class Recommendation(BaseModel):
    title: str
    reason: str

class ActionItem(BaseModel):
    task: str
    owner: str = "TBD"
    due: str = "TBD"

class GenOutput(BaseModel):
    summary: str = ""
    body: str = ""
    recommendations: List[Recommendation] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

def build_prompt(section_name: str, context_text: str, constraints: List[str], style: str = "행정 공문서 체계(배경-목적-추진-기대효과), 근거 명시, 간결·정확") -> Dict[str, str]:
    system = (
        "당신은 지자체 행정문서 작성 전문가입니다. "
        "반드시 JSON만 출력하세요. 문장 외 텍스트나 주석은 금지합니다."
    )
    user = (
        f"[섹션]: {section_name}\n\n"
        f"[컨텍스트(근거)]:\n{context_text}\n\n"
        f"[제약]:\n- " + "\n- ".join(constraints) + "\n\n"
        f"[스타일]: {style}\n\n"
        "요구 출력(JSON 스키마): {\n"
        ' "summary": str,\n'
        ' "body": str,\n'
        ' "recommendations": [{"title": str, "reason": str}, ...],\n'
        ' "action_items": [{"task": str, "owner": str, "due": str}, ...],\n'
        ' "references": [str, ...],\n'
        ' "meta": {"section": str, "timestamp": int, "provenance": [str, ...]}\n'
        "}\n"
        "주의: JSON 외 장식/서문/코드는 절대 출력하지 마세요."
    )
    return {"system": system, "user": user}

# Toggle these to use real GPT
_GPT_OK = False
GPT_MODEL = "gpt-4o-mini"
client = None  # set your OpenAI client before use

def call_llm_generate(section_name: str, context_chunks: List[str], constraints: Optional[List[str]] = None, client=None, model: str = GPT_MODEL, temperature: float = 0.2) -> Dict[str, Any]:
    constraints = constraints or [
        "공문서 톤으로 작성",
        "모든 주장 뒤에 근거 문장을 반영",
        "요약은 2~4문장, 본문은 7~12문장",
        "references에는 근거 태그만 기입([type/doc#chunk:p])",
    ]
    ctx_text = "\n\n".join(context_chunks)
    provenance = []
    for line in context_chunks:
        if line.startswith("[") and "]" in line:
            tag = line.split("]")[0].strip("[]")
            provenance.append(tag)
    prompt = build_prompt(section_name, ctx_text, constraints)
    ts = int(time.time() * 1000)
    ctx_hash = hashlib.md5(ctx_text.encode("utf-8")).hexdigest()[:8]

    if client is not None and _GPT_OK:
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            obj = GenOutput(**data)
            obj.meta.update({
                "section": section_name, "timestamp": ts, "ctx_hash": ctx_hash,
                "provenance": provenance, "model": model
            })
            return obj.model_dump()
        except Exception as e:
            logger.warning("LLM call failed, using stub. %s", e)

    # stub fallback
    dummy = GenOutput(
        summary="하이브리드 검색과 Late·Recursive 청킹을 통해 근거 중심의 행정문서 자동화를 구현합니다.",
        body=(
            "본 문서는 BM25와 의미 임베딩 검색을 결합하여 규정·조례·서식을 정밀 탐색하고, "
            "Late Chunking으로 전역 문맥을 반영한 근거 기반 서술을 제공합니다. "
            "생성 단계는 행정 문체를 준수하며 배경-목적-추진-기대효과의 구조를 따릅니다."
        ),
        recommendations=[
            Recommendation(title="RPA 단계적 도입", reason="보고서 집계→결재 연동→원장 반영 순으로 리스크 분산"),
            Recommendation(title="근거 문구 자동 인용", reason="심사·감사 대응을 위한 출처 가시성 강화"),
        ],
        action_items=[
            ActionItem(task="결재메일 자동화 스크립트 배포", owner="정보통신과", due="2025-11-15"),
            ActionItem(task="벡터DB 인덱스 주간 리빌드", owner="데이터정책과", due="매주 금"),
        ],
        references=[tag for tag in provenance] or ["pdf/paju_plan#0:p1"],
        meta={"section": section_name, "timestamp": ts, "ctx_hash": ctx_hash, "provenance": provenance, "model": "stub"},
    )
    return dummy.model_dump()

def call_llm_stub(prompt_text: str) -> str:
    """단순 스텁(JSON 문자열)"""
    obj = GenOutput(
        summary="스텁: 입력 컨텍스트를 바탕으로 행정문서 초안의 구조를 생성합니다.",
        body="스텁 본문: 배경-목적-추진-기대효과의 순서로 서술하며, 인용 태그는 컨텍스트에서 추출해 references에 포함합니다.",
        recommendations=[Recommendation(title="스텁 권고", reason="테스트 환경")],
        action_items=[ActionItem(task="스텁 작업", owner="TBD", due="TBD")],
        references=["stub/source#0:p1"],
        meta={"section": "stub", "timestamp": int(time.time()*1000), "provenance": ["stub/source#0:p1"], "model": "stub"},
    )
    return json.dumps(obj.model_dump(), ensure_ascii=False)

# =========================
# Validation & UNIEVAL proxy/GPT
# =========================
def _pretty_pydantic_error(e: ValidationError) -> str:
    msgs = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", []))
        typ = err.get("type", "")
        msg = err.get("msg", "")
        msgs.append(f"{loc}: {msg} ({typ})")
    return "; ".join(msgs) if msgs else str(e)

def validate_json_payload(js_or_obj: Any) -> Dict[str, Any]:
    try:
        obj = json.loads(js_or_obj) if isinstance(js_or_obj, str) else js_or_obj
    except json.JSONDecodeError as e:
        raise ValueError(f"[ValidationError] JSON 파싱 실패: {e}")
    try:
        validated = GenOutput(**obj)
        return validated.model_dump()
    except ValidationError as e:
        raise ValueError(f"[ValidationError] 스키마 오류: {_pretty_pydantic_error(e)}")

_REF_APA_RE = re.compile(r"\([\w가-힣]+,\s*20\d{2}\)")
_REF_TAG_RE = re.compile(r"\[([a-zA-Z]+)/([^\]#]+)#(\d+)(:p\d+)?\]")

def reference_metrics(text: str) -> Dict[str, Any]:
    if not text or not isinstance(text, str):
        return {"apa_hits": 0, "tag_hits": 0, "unique_sources": 0, "coverage": 0.0, "num_sentences": 0}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]|다\.)\s+', text) if s.strip()]
    apa_hits = len(_REF_APA_RE.findall(text))
    tag_matches = list(_REF_TAG_RE.finditer(text))
    tag_hits = len(tag_matches)
    src_keys = set()
    tag_sent_hits = 0
    for m in tag_matches:
        typ, doc, chunk, page = m.groups()
        src_keys.add(f"{typ}/{doc}")
    for s in sentences:
        if _REF_APA_RE.search(s) or _REF_TAG_RE.search(s):
            tag_sent_hits += 1
    coverage = tag_sent_hits / max(1, len(sentences))
    return {"apa_hits":apa_hits, "tag_hits":tag_hits, "unique_sources":len(src_keys), "coverage":round(coverage,4), "num_sentences":len(sentences)}

def _ngram_redundancy(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n + 1: return 0.0
    grams = {}
    for i in range(len(tokens) - n + 1):
        g = " ".join(tokens[i:i+n])
        grams[g] = grams.get(g, 0) + 1
    rep = sum(1 for c in grams.values() if c > 1)
    return rep / max(1, len(grams))

def _lexical_diversity(text: str) -> float:
    tokens = [t.lower() for t in re.findall(r"\w+", text)]
    if not tokens: return 0.0
    return len(set(tokens)) / len(tokens)

def _clip01(x: Optional[float]) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)): return 0.0
    try: v = float(x)
    except Exception: return 0.0
    return max(0.0, min(1.0, v))

def offline_unieval_proxy(draft_text: str) -> Dict[str, Any]:
    if not draft_text or not draft_text.strip():
        return {"Accuracy":0.0,"Relevance":0.0,"Consistency":0.0,"Fluency":0.0,"Redundancy":1.0,"Comment":"빈 텍스트"}
    L = len(draft_text)
    len_score = 1.0 if 800 <= L <= 4000 else max(0.2, min(1.0, L/800 if L < 800 else 4000/L))
    div = _lexical_diversity(draft_text)
    red = _ngram_redundancy(draft_text, n=3)
    refs = reference_metrics(draft_text); cov = refs["coverage"]
    accuracy = _clip01(0.6 + 0.2*cov)
    relevance = _clip01(0.6 + 0.3*cov)
    consistency = _clip01(0.5 + 0.3*len_score - 0.2*red)
    fluency = _clip01(0.5 + 0.4*div - 0.2*red)
    redundancy = _clip01(red)
    return {"Accuracy":round(accuracy,3),"Relevance":round(relevance,3),"Consistency":round(consistency,3),"Fluency":round(fluency,3),"Redundancy":round(redundancy,3),"Comment":f"len={L}, cov={cov:.2f}, div={div:.2f}, red={red:.2f}"}

def gpt_validate(section_name: str, draft_text: str, *, categories: Optional[List[str]] = None) -> Dict[str, Any]:
    cats = categories or ["Accuracy","Relevance","Consistency","Fluency","Redundancy"]
    if not draft_text or not draft_text.strip():
        return {"scores": offline_unieval_proxy(draft_text), "total": 0.0, "meta": {"mode": "empty"}}
    if not _GPT_OK or client is None:
        scores = offline_unieval_proxy(draft_text)
        total = _clip01(0.3*scores["Accuracy"] + 0.25*scores["Relevance"] + 0.25*scores["Consistency"] + 0.2*scores["Fluency"] - 0.2*scores["Redundancy"])
        return {"scores": scores, "total": round(total,3), "meta": {"mode": "offline"}}
    system_prompt = ("당신은 한국어 행정문서 평가 전문가입니다. 반드시 JSON만 출력하세요. 점수 범위는 0~1이며 소수점 세 자리 이내.")
    rubric = ("평가기준:\n- Accuracy: 사실 및 근거 일치\n- Relevance: 주제 적합성\n- Consistency: 논리 일관성\n- Fluency: 문장 유창성 및 행정 문체\n- Redundancy: 반복 최소화(낮을수록 좋음)\n"
              "출력 스키마(JSON): {\"Accuracy\": float, \"Relevance\": float, \"Consistency\": float, \"Fluency\": float, \"Redundancy\": float, \"Comment\": str}")
    user_prompt = (f"[섹션명]: {section_name}\n\n[평가대상 초안]:\n{draft_text}\n\nJSON 외의 텍스트는 절대 출력하지 마세요.")
    try:
        res = client.chat.completions.create(
            model=GPT_MODEL, temperature=0.0, response_format={"type":"json_object"},
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":rubric},{"role":"user","content":user_prompt}], timeout=60
        )
        raw = res.choices[0].message.content
        parsed = json.loads(raw)
        out = {}
        for k in cats + ["Comment"]:
            out[k] = parsed.get(k, "" if k=="Comment" else 0.0)
        out = {k: (_clip01(v) if k!="Comment" else v) for k,v in out.items()}
        scores = {k: round(out[k],3) for k in cats}; scores["Comment"] = out["Comment"]
        total = _clip01(0.3*scores["Accuracy"] + 0.25*scores["Relevance"] + 0.25*scores["Consistency"] + 0.2*scores["Fluency"] - 0.2*scores["Redundancy"])
        return {"scores": scores, "total": round(total,3), "meta": {"model": GPT_MODEL, "timestamp": int(time.time()), "section": section_name}}
    except Exception:
        scores = offline_unieval_proxy(draft_text)
        total = _clip01(0.3*scores["Accuracy"] + 0.25*scores["Relevance"] + 0.25*scores["Consistency"] + 0.2*scores["Fluency"] - 0.2*scores["Redundancy"])
        return {"scores": scores, "total": round(total,3), "meta": {"mode": "fallback"}}

# =========================
# LangGraph-like State & ValidatorNode
# =========================
@dataclass
class PipelineState:
    query: str
    section_name: str
    doc_type: str
    constraints: List[str]
    context: str = ""
    draft_json: str = ""
    obj: Optional[GenOutput] = None
    eval_result: Optional[Dict[str, Any]] = None
    ref_stats: Optional[Dict[str, Any]] = None

class ValidatorNode:
    def __init__(self, *, min_total: float = 0.70, min_cov: float = 0.15, hard_fail_on_schema: bool = True, hard_fail_on_refs: bool = False, hard_fail_on_score: bool = False):
        self.min_total = min_total
        self.min_cov = min_cov
        self.hard_fail_on_schema = hard_fail_on_schema
        self.hard_fail_on_refs = hard_fail_on_refs
        self.hard_fail_on_score = hard_fail_on_score

    def __call__(self, st: PipelineState) -> PipelineState:
        # 1) schema
        try:
            validated_dict = validate_json_payload(st.draft_json)
            obj = GenOutput(**validated_dict)
        except Exception as e:
            logger.error("Schema validation failed at section=%s: %s", st.section_name, e)
            if self.hard_fail_on_schema: raise
            return replace(st, obj=None)

        # 2) reference coverage
        ref_stats = reference_metrics(obj.body)
        if ref_stats.get("coverage", 0.0) < self.min_cov:
            logger.warning("Low reference coverage: section=%s cov=%.3f < %.3f", st.section_name, ref_stats.get("coverage", 0.0), self.min_cov)
            if self.hard_fail_on_refs:
                raise ValueError(f"Reference coverage below threshold at section={st.section_name}: {ref_stats.get('coverage',0.0):.3f} < {self.min_cov:.3f}")

        # 3) evaluation
        eval_result = gpt_validate(st.section_name, obj.body)
        total = float(eval_result.get("total", 0.0))
        if total < self.min_total:
            logger.warning("Low UNIEVAL total score: section=%s total=%.3f < %.3f", st.section_name, total, self.min_total)
            if self.hard_fail_on_score:
                raise ValueError(f"UNIEVAL total below threshold at section={st.section_name}: {total:.3f} < {self.min_total:.3f}")

        # 4) merge meta
        meta = dict(obj.meta or {}); meta.update({"unieval": eval_result, "reference_stats": ref_stats})
        obj.meta = meta
        return replace(st, obj=obj, eval_result=eval_result, ref_stats=ref_stats)

# =========================
# Visualization: UNIEVAL
# =========================
def _prepare_scores(obj: Any, invert_redundancy: bool = True) -> Tuple[List[str], List[float], Dict[str, float], float]:
    uni = (getattr(obj, "meta", None) or {}).get("unieval", {})
    scores = uni.get("scores", {}); total = _clip01(uni.get("total", 0.0))
    metrics = ["Accuracy", "Relevance", "Consistency", "Fluency", "Redundancy"]
    values = []
    parsed = {}
    for m in metrics:
        v = _clip01(scores.get(m, 0.0))
        if m == "Redundancy" and invert_redundancy: v = 1.0 - v
        values.append(v); parsed[m] = v
    return metrics, values, parsed, total

def plot_unieval_results(obj: Any, out_dir: str, invert_redundancy: bool = True) -> Dict[str, str]:
    out = {}
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    metrics, values, scores, total = _prepare_scores(obj, invert_redundancy=invert_redundancy)

    # Bar
    plt.figure(figsize=(6,4)); plt.bar(metrics, values); plt.ylim(0,1)
    plt.title(f"UNIEVAL Metrics (Bar) • Total={total:.3f}"); plt.ylabel("Score (0~1)"); plt.tight_layout()
    bar_path = out_path / "UNIEVAL_bar.png"; plt.savefig(bar_path, dpi=200); plt.close(); out["bar"] = str(bar_path)

    # Radar
    N = len(metrics); angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    vals_closed = values + values[:1]; ang_closed = np.concatenate([angles, [angles[0]]])
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(ang_closed, vals_closed, linewidth=2); ax.fill(ang_closed, vals_closed, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles), metrics); ax.set_ylim(0,1)
    plt.title(f"UNIEVAL Metrics (Radar) • Total={total:.3f}"); plt.tight_layout()
    radar_path = out_path / "UNIEVAL_radar.png"; plt.savefig(radar_path, dpi=200); plt.close(); out["radar"] = str(radar_path)

    # JSON/CSV
    json_payload = {"metrics": metrics, "values": values, "scores": scores, "total": round(total,3), "meta": getattr(obj,"meta",{})}
    json_path = out_path / "UNIEVAL_scores.json"; json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), "utf-8"); out["json"] = str(json_path)
    csv_path = out_path / "UNIEVAL_scores.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["Metric","Score"]); [w.writerow([m, f"{v:.3f}"]) for m,v in zip(metrics, values)]; w.writerow(["Total", f"{total:.3f}"])
    out["csv"] = str(csv_path)
    logger.info("UNIEVAL outputs saved: %s", out)
    return out

# =========================
# Orchestrator
# =========================
def run_pipeline(
    query: str,
    section_name: str = "도입 배경 및 필요성",
    doc_type: str = "행정 자동화 기획서",
    constraints: Optional[List[str]] = None,
    corpus_dir: str = "./corpus",
    out_dir: str = "./outputs",
    embed_model_name: str = "intfloat/multilingual-e5-large",
    use_real_llm: bool = False,
    client_override=None
) -> GenOutput:
    """
    1) Ingest (auto-discover in corpus_dir)
    2) Build embeddings
    3) Retrieve context
    4) Generate (LLM or stub)
    5) Validate + Score
    6) Save outputs (json/csv/png) under out_dir/run_YYYYMMDD_HHMMSS
    """
    start_time = time.time()
    constraints = constraints or ["근거 기반", "중복 최소화", "정량적 수치 포함", "UNIEVAL 기준 준수"]

    # 1. Load corpus
    corpus = TextCorpus(embed_model_name=embed_model_name)
    cdir = Path(corpus_dir); cdir.mkdir(parents=True, exist_ok=True)
    for p in cdir.glob("*.pdf"): corpus.add_pdf(str(p))
    for p in cdir.glob("*.docx"): corpus.add_docx(str(p))
    for p in list(cdir.glob("*.txt")) + list(cdir.glob("*.md")): corpus.add_text(str(p))
    for p in list(cdir.glob("*.html")) + list(cdir.glob("*.htm")): corpus.add_html(str(p))
    for p in cdir.glob("*.csv"): corpus.add_csv(str(p))
    for p in cdir.glob("*.json"): corpus.add_json(str(p))
    corpus.summary()

    # 2. Build embeddings (late hybrid)
    corpus.build_embeddings()

    # 3. Retriever & context
    retriever = HybridRetriever(corpus)
    context = retriever.build_context(query, k=6)

    # 4. Draft
    if use_real_llm and client_override is not None:
        draft_obj = call_llm_generate(section_name=section_name, context_chunks=context.split("\n\n"), constraints=constraints, client=client_override)
        llm_json = json.dumps(draft_obj, ensure_ascii=False, indent=2)
    else:
        llm_json = call_llm_stub(f"문서유형: {doc_type}\n{context}")

    # 5. Validate
    val = ValidatorNode()
    st = PipelineState(query=query, section_name=section_name, doc_type=doc_type, constraints=constraints, context=context, draft_json=llm_json)
    st = val(st)

    # 6. Save outputs
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"run_{ts}"; out_path.mkdir(parents=True, exist_ok=True)
    # generated.json
    result_data = st.obj.model_dump() if hasattr(st.obj, "model_dump") else st.obj.dict()
    (out_path / "generated.json").write_text(json.dumps(result_data, ensure_ascii=False, indent=2), "utf-8")
    # context
    (out_path / "context.txt").write_text(context, "utf-8")
    # plots
    plot_unieval_results(st.obj, str(out_path))

    elapsed = time.time() - start_time
    logger.info("[완료] 총 실행 시간: %.2f초 | 출력: %s", elapsed, out_path)
    return st.obj

# =========================
# __main__ (example)
# =========================
if __name__ == "__main__":
    # Minimal example run (stub mode)
    run_pipeline(
        query="행정문서 자동화를 위한 규정 및 서식 표준화 방안",
        section_name="도입 배경 및 필요성",
        doc_type="행정 자동화 기획서",
        corpus_dir="./corpus",
        out_dir="./outputs",
        embed_model_name="intfloat/multilingual-e5-large",
        use_real_llm=False  # set True with client_override and _GPT_OK=True
    )
