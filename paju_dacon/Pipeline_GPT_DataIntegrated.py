# -*- coding: utf-8 -*-
"""
Pipeline_GPT_DataIntegrated.py
E2E 문서 생성 파이프라인 (Hybrid Retrieval + Validation + Export + RPA Stub)

"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re, json, csv, time, sys

import numpy as np
from PyPDF2 import PdfReader
from pydantic import BaseModel, ValidationError
from rank_bm25 import BM25Okapi

# ===== (옵션) FAISS 임베딩 =====
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False


# =========================
# 0) 데이터 모델 & 유틸
# =========================
class GenOutput(BaseModel):
    summary: str
    body: str
    recommendations: List[Dict[str, str]]
    action_items: List[Dict[str, str]]
    references: List[str]

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def sentence_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """문장 기반 + overlap 청킹 (한글 문서 최적화)"""
    if not text:
        return []
    # 문장 분리: 마침표/물음표/느낌표/종결 '다.' 패턴 기준
    sents = re.split(r'(?<=[\.!\?]|다\.)\s+', text)
    chunks, buf, cur_len = [], [], 0
    for sent in sents:
        sent = sent.strip()
        if not sent:
            continue
        buf.append(sent)
        cur_len += len(sent)
        if cur_len >= chunk_size:
            chunk = normalize_text(" ".join(buf))
            if len(chunk) >= 50:
                chunks.append(chunk)
            # overlap: 뒤에서 일정 길이 문장 유지
            keep = []
            kept_len = 0
            for s in reversed(buf):
                keep.append(s)
                kept_len += len(s)
                if kept_len >= overlap:
                    break
            buf = list(reversed(keep))
            cur_len = sum(len(x) for x in buf)
    if buf:
        chunk = normalize_text(" ".join(buf))
        if len(chunk) >= 50:
            chunks.append(chunk)
    return chunks


# =========================
# 1) 코퍼스
# =========================
@dataclass
class DocRecord:
    doc_id: str
    title: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

class TextCorpus:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[str] = []
        self.records: List[DocRecord] = []
        self.chunk_meta: List[Dict[str, Any]] = []

    def add_pdf(self, file_path: str, title: Optional[str] = None, chunk_size: int = 500, overlap: int = 50):
        p = Path(file_path)
        reader = PdfReader(str(p))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        text = normalize_text(text)
        doc_id = p.stem
        title = title or p.stem
        rec = DocRecord(doc_id=doc_id, title=title, text=text, meta={"type": "pdf", "path": str(p)})
        self.records.append(rec)
        self.documents.append({"path": str(p), "text": text})
        self._add_chunks_from_text(text, doc_id, title, "pdf", chunk_size, overlap)

    def add_txt(self, file_path: str, doc_type: str = "txt", title: Optional[str] = None,
                chunk_size: int = 500, overlap: int = 50, encoding: str = "utf-8"):
        p = Path(file_path)
        text = normalize_text(p.read_text(encoding=encoding))
        doc_id = p.stem
        title = title or p.stem
        rec = DocRecord(doc_id=doc_id, title=title, text=text, meta={"type": doc_type, "path": str(p)})
        self.records.append(rec)
        self.documents.append({"path": str(p), "text": text})
        self._add_chunks_from_text(text, doc_id, title, doc_type, chunk_size, overlap)

    def _add_chunks_from_text(self, text: str, doc_id: str, title: str, doc_type: str,
                              chunk_size: int, overlap: int):
        for i, ch in enumerate(sentence_chunking(text, chunk_size=chunk_size, overlap=overlap)):
            self.chunks.append(ch)
            self.chunk_meta.append({
                "doc_id": doc_id,
                "title": title,
                "type": doc_type,
                "chunk_id": i
            })

    def summary(self):
        print(f"📚 Documents: {len(self.documents)} | 🧩 Chunks: {len(self.chunks)}")


# =========================
# 2) Hybrid Retriever
# =========================
class HybridRetriever:
    def __init__(self, corpus: TextCorpus, embed_model_name: str = "intfloat/e5-large"):
        if not corpus.chunks:
            raise ValueError("❌ Corpus has no chunks. Load PDFs/TXTs and chunk before initializing retriever.")
        self.corpus = corpus
        # BM25
        tokenized = [c.split() for c in corpus.chunks]
        self.bm25 = BM25Okapi(tokenized)
        # FAISS (옵션)
        self.has_faiss = _FAISS_OK
        if self.has_faiss:
            self.emb_model = SentenceTransformer(embed_model_name)
            self.emb_mat = self.emb_model.encode(
                corpus.chunks, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=False
            )
            dim = self.emb_mat.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.emb_mat)

    def search(self, query: str, topk: int = 6, alpha: float = 0.5) -> List[Tuple[int, float]]:
        # BM25
        token_q = query.split()
        bm25_scores = self.bm25.get_scores(token_q)
        b = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-8)
        # FAISS
        if self.has_faiss:
            q_emb = self.emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            D, I = self.index.search(q_emb, topk)
            faiss_scores = np.zeros_like(b)
            for rank, idx in enumerate(I[0]):
                faiss_scores[idx] = D[0][rank]
            f = (faiss_scores - faiss_scores.min()) / (faiss_scores.ptp() + 1e-8)
        else:
            f = np.zeros_like(b)
        hybrid = alpha * b + (1 - alpha) * f
        top_idx = np.argsort(-hybrid)[:topk]
        return [(int(i), float(hybrid[i])) for i in top_idx]

    def build_context(self, query: str, k: int = 6) -> str:
        hits = self.search(query, topk=k, alpha=0.5)
        lines = []
        for idx, sc in hits:
            meta = self.corpus.chunk_meta[idx] if idx < len(self.corpus.chunk_meta) else {}
            title = meta.get("title", "NA")
            typ = meta.get("type", "NA")
            cid = meta.get("chunk_id", idx)
            lines.append(f"[{typ}/{title}#{cid}] {self.corpus.chunks[idx]}")
        return "\n\n".join(lines)


# =========================
# 3) Prompt Builder & LLM 스텁
# =========================
def build_prompt(section_name: str, doc_type: str, constraints: List[str], references: str, query: str) -> str:
    return f"""
# 작성항목: [{section_name}]
# 문서유형: [{doc_type}]
# 작성조건: {', '.join(constraints)}
# 참고자료(요약/발췌):
{references}

다음 기준에 맞춰 내용을 생성하세요:
1) 출력 형식: JSON. 키:
{{
  "summary": "한 문단 요약",
  "body": "상세 본문",
  "recommendations": [{{"title":"", "detail":"", "impact_estimate":""}}],
  "action_items": [{{"task":"", "owner":"", "due":"YYYY-MM-DD"}}],
  "references": ["출처1", "출처2"]
}}
2) 본문 내 주장 옆에 (기관, 20xx) 형태의 간단 주석을 최소 1회 이상 표기.
3) 한국어 공식 보고서 문체로 간결하고 근거 기반으로 서술.
4) 쿼리: "{query}"
""".strip()

def call_llm_stub(prompt: str) -> str:
    """외부 API 없이 규격을 만족하는 더미 JSON을 반환 (운영 시 실제 LLM으로 교체)"""
    dummy = {
        "summary": "하이브리드 검색으로 규정·서식을 통합 분석하여 행정문서 자동화를 설계한다.",
        "body": "BM25와 FAISS를 결합해 규정/서식을 정밀 탐색하고, RPA로 결재/집계/입력을 자동화한다. (파주시, 2025)",
        "recommendations": [
            {"title": "RPA 단계적 도입", "detail": "보고서 집계→결재 연동→원장 반영 순서로 단계 적용", "impact_estimate": "월 30~45시간 절감"}
        ],
        "action_items": [
            {"task": "결재메일 자동화 스크립트 배포", "owner": "정보통신과", "due": "2025-11-15"}
        ],
        "references": ["파주시 주요업무계획(2025)", "전자정부 표준프레임워크 가이드(2023)"]
    }
    return json.dumps(dummy, ensure_ascii=False, indent=2)


# =========================
# 4) Validation
# =========================
def validate_json_payload(js: str) -> GenOutput:
    try:
        obj = json.loads(js)
        return GenOutput(**obj)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"[ValidationError] 생성 JSON이 스키마와 다릅니다: {e}")

def check_reference_annotations(text: str) -> bool:
    """본문 내 (기관, 20xx) 형태 주석이 존재하는지 간단 점검"""
    return bool(re.search(r"\([\w가-힣]+,\s*20\d{2}\)", text))


# =========================
# 5) Export
# =========================
def export_json(obj: GenOutput, out_path: str):
    Path(out_path).write_text(json.dumps(obj.dict(), ensure_ascii=False, indent=2), encoding="utf-8")

def export_csv_action_items(obj: GenOutput, csv_path: str):
    header = ["task", "owner", "due"]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for it in obj.action_items:
            w.writerow({k: it.get(k, "") for k in header})


# =========================
# 6) RPA Stubs
# =========================
def rpa_send_approval_mail(obj: GenOutput, to_addr: str):
    print(f"[RPA] 결재 메일 전송 → {to_addr}")
    print(f"제목: [결재요청] {obj.summary[:50]}...")
    print(f"본문(요약): {obj.summary}")

def rpa_append_report_ledger(obj: GenOutput, ledger_csv: str):
    Path(ledger_csv).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(ledger_csv).exists()
    with open(ledger_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["date", "summary", "n_action_items", "references"])
        now = time.strftime("%Y-%m-%d")
        w.writerow([now, normalize_text(obj.summary), len(obj.action_items), ";".join(obj.references)])

def rpa_fill_form_stub(obj: GenOutput):
    print("[RPA] 폼 자동 입력(스텁) — 실제 운영 환경 API/자동화 스크립트 연동 지점")


# =========================
# 7) LangGraph-Style Nodes
# =========================
@dataclass
class PipelineState:
    query: str
    section_name: str
    doc_type: str
    constraints: List[str] = field(default_factory=list)
    references_ctx: str = ""
    draft_json: str = ""
    obj: Optional[GenOutput] = None

class DataIngestNode:
    def __init__(self, corpus_dir: str, chunk_size: int = 500, overlap: int = 50):
        self.corpus_dir = Path(corpus_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def __call__(self) -> TextCorpus:
        corpus = TextCorpus()
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"❌ Corpus directory not found: {self.corpus_dir}")

        # PDF 우선 로드
        for p in self.corpus_dir.glob("*.pdf"):
            corpus.add_pdf(str(p), chunk_size=self.chunk_size, overlap=self.overlap)
        # TXT도 지원
        for p in self.corpus_dir.glob("*.txt"):
            # 파일명에 따라 규정/서식/가이드라인 태깅 예시
            dtype = "규정" if "reg" in p.stem else ("서식" if "form" in p.stem else "가이드라인")
            corpus.add_txt(str(p), doc_type=dtype, chunk_size=self.chunk_size, overlap=self.overlap)

        corpus.summary()
        if len(corpus.chunks) == 0:
            raise RuntimeError("❌ No chunks loaded. Check PDF/TXT extraction and directory content.")
        return corpus

class ContextSearchNode:
    def __init__(self, retriever: HybridRetriever, k: int = 6):
        self.retriever = retriever
        self.k = k
    def __call__(self, st: PipelineState) -> PipelineState:
        st.references_ctx = self.retriever.build_context(st.query, k=self.k)
        return st

class DraftWriterNode:
    def __call__(self, st: PipelineState) -> PipelineState:
        prompt = build_prompt(st.section_name, st.doc_type, st.constraints, st.references_ctx, st.query)
        st.draft_json = call_llm_stub(prompt)  # 운영 시 실제 LLM으로 교체
        return st

class ValidatorNode:
    def __call__(self, st: PipelineState) -> PipelineState:
        obj = validate_json_payload(st.draft_json)
        if not check_reference_annotations(obj.body):
            print("[WARN] 본문에 (기관, 연도) 형태 주석이 부족할 수 있습니다. 프롬프트를 보강하세요.")
        st.obj = obj
        return st

class ExporterNode:
    def __init__(self, out_dir: str = "./outputs"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
    def __call__(self, st: PipelineState) -> PipelineState:
        assert st.obj is not None, "Export 단계 전에 Validator를 통과해야 합니다."
        export_json(st.obj, str(self.out_dir / "generated.json"))
        export_csv_action_items(st.obj, str(self.out_dir / "action_items.csv"))
        print(f"[EXPORT] JSON/CSV 저장 완료 → {self.out_dir}")
        return st

class RPAOpsNode:
    def __init__(self, out_dir: str = "./outputs", approval_addr: str = "approval@org.local"):
        self.out_dir = Path(out_dir)
        self.approval_addr = approval_addr
    def __call__(self, st: PipelineState) -> PipelineState:
        assert st.obj is not None
        rpa_send_approval_mail(st.obj, to_addr=self.approval_addr)
        rpa_append_report_ledger(st.obj, str(self.out_dir / "report_ledger.csv"))
        rpa_fill_form_stub(st.obj)
        return st


# =========================
# 8) Orchestrator
# =========================
def run_pipeline(
    query: str,
    section_name: str = "도입 배경 및 필요성",
    doc_type: str = "행정 자동화 기획서",
    constraints: Optional[List[str]] = None,
    corpus_dir: str = "/home/alpaco/homework/paju_dacon/corpus",
    out_dir: str = "./outputs",
    k_ctx: int = 6,
):
    constraints = constraints or ["근거 기반", "중복 최소화", "정량적 수치 포함", "UNIEVAL 기준 준수"]

    # 1) Data ingest
    ingest = DataIngestNode(corpus_dir=corpus_dir, chunk_size=500, overlap=50)
    corpus = ingest()

    # 2) Retriever
    retriever = HybridRetriever(corpus)

    # 3) Build state
    st = PipelineState(query=query, section_name=section_name, doc_type=doc_type, constraints=constraints)

    # 4) Nodes
    ctx_node = ContextSearchNode(retriever, k=k_ctx)
    draft_node = DraftWriterNode()
    val_node = ValidatorNode()
    exp_node = ExporterNode(out_dir=out_dir)
    rpa_node = RPAOpsNode(out_dir=out_dir, approval_addr="approval@org.local")

    # 5) Flow
    st = ctx_node(st)
    st = draft_node(st)
    st = val_node(st)
    st = exp_node(st)
    st = rpa_node(st)

    print("[DONE] 결과:", str(Path(out_dir) / "generated.json"))
    return st.obj


# =========================
# 9) Main
# =========================
if __name__ == "__main__":
    try:
        run_pipeline(
            query="하이브리드 검색을 통해 행정문서 자동화를 위한 규정/서식 요건을 통합",
            section_name="도입 배경 및 필요성",
            doc_type="행정 자동화 기획서",
            constraints=["근거 기반", "중복 최소화", "정량적 수치 포함", "UNIEVAL 기준 준수"],
            corpus_dir="/home/alpaco/homework/paju_dacon/corpus",
            out_dir="./outputs",
            k_ctx=6,
        )
    except Exception as e:
        print("failed:", e, file=sys.stderr)
        sys.exit(1)