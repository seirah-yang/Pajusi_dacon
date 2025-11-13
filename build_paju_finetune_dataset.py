# -*- coding: utf-8 -*-
"""
파주시 행정문서용 SKT A.X 파인튜닝 데이터셋 자동 생성 스크립트
data/
 ├─ laws_pdf/      # 파주시 조례/규칙 PDF
 ├─ press_html/    # 보도자료 HTML
 └─ csv/           # 공공데이터 CSV
 
설계 원칙
- 외부 LLM 호출 없이, 원문으로부터 지도학습 쌍을 규칙 기반으로 생성
- 한국어 행정 문체 템플릿 일관화 (system role)
- 출처/근거 메타 포함(파일명, 조문, 날짜 등) → RAG/검증 연동 용이

의존성
    pip install pdfplumber pytesseract pillow beautifulsoup4 readability-lxml lxml pandas python-dateutil kss

출력 포맷(JSONL)
    {"messages": [
        {"role":"system","content": "...파주시 공문서 어조..."},
        {"role":"user","content":   "질문 또는 지시"},
        {"role":"assistant","content":"근거 기반 행정 문장"}
     ],
     "meta": {"source":"file.pdf#p3","law_id":"paju_sdgs","date":"2024-03-01"}}
"""

from __future__ import annotations
import os, re, json, glob, hashlib, random, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import pandas as pd

# PDF
import pdfplumber
try:
    import pytesseract
    from PIL import Image
    _OCR_OK = True
except Exception:
    _OCR_OK = False

# HTML
from bs4 import BeautifulSoup
from readability import Document

# Text utils
try:
    import kss  # 한국어 문장 분리
    _KSS = True
except Exception:
    _KSS = False

from dateutil import parser as dateparser

# -------------------------------
# 템플릿 & 공통 문구
# -------------------------------
SYSTEM_PROMPT = (
    "당신은 파주시청에서 근무하는 행정보고서/민원응대 작성 전문가입니다. "
    "답변은 한국어 공문서 어조로, 간결하고 단정하게 작성하세요. "
    "가능하면 '근거' 섹션에 출처(조례명, 조문 또는 보도자료 제목, 날짜)를 명시하세요."
)

# 사용자 지시 템플릿
LAW_USER_TEMPLATES = [
    "다음 조례(일부) 내용을 시민이 이해하기 쉽게 3~5문장으로 요약하고, 핵심 목적과 적용 대상을 분리해 설명하세요.\n\n[원문]\n{chunk}",
    "다음 조례 조문을 행정 보고서 요약 형식으로 정리하세요. 항목: 배경, 목적, 적용범위, 근거.\n\n[원문]\n{chunk}",
]

PRESS_USER_TEMPLATES = [
    "다음 보도자료의 핵심 내용을 4~6문장으로 정리하고, 시민 입장에서 기대효과 2가지를 추가하세요.\n\n[보도자료]\n{chunk}",
    "다음 보도자료를 정책 브리핑 요약으로 재구성하세요. 항목: 개요, 주요 내용, 추진 일정, 문의.\n\n[보도자료]\n{chunk}",
]

CSV_USER_TEMPLATES = [
    "다음 지표 데이터(일부)를 근거로 정책 시사점을 3가지 도출하세요. 수치는 범위를 유지하며 과장하지 마세요.\n\n[데이터 요약]\n{chunk}",
    "다음 표의 수치를 바탕으로 행정동 단위의 변화 추세를 3~4문장으로 정리하고 주의점 1가지를 제시하세요.\n\n[표]\n{chunk}",
]

# 조례명 추출용 간단 패턴 (필요시 커스텀 가능)
LAW_TITLE_PAT = re.compile(r"^\s*(.*?조례).*", re.M)
ARTICLE_PAT   = re.compile(r"제\s*\d+\s*조[\s\S]*?")

# -------------------------------
# 헬퍼: 텍스트 정리/청킹
# -------------------------------

def normalize_text(s: str) -> str:
    s = (s or "").replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sent_split(s: str) -> List[str]:
    s = normalize_text(s)
    if not s:
        return []
    if _KSS:
        try:
            return [x.strip() for x in kss.split_sentences(s) if x.strip()]
        except Exception:
            pass
    # fallback: 마침표 기준 단순 분할
    parts = re.split(r"(?<=[.!?\u3002\uFF0E\uFF01\uFF1F])\s+", s)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    """문장 단위 청킹, 토큰 정보 없으므로 문자 길이 기준."""
    sents = sent_split(text)
    chunks, buf = [], []
    cur_len = 0
    for sent in sents:
        if cur_len + len(sent) + 1 <= max_chars:
            buf.append(sent)
            cur_len += len(sent) + 1
        else:
            if buf:
                chunks.append(" ".join(buf).strip())
            # overlap 일부 유지
            buf = [sent] if len(sent) < max_chars else [sent[:max_chars]]
            cur_len = sum(len(x) + 1 for x in buf)
    if buf:
        chunks.append(" ".join(buf).strip())
    # 인접 청크 중복 감소
    if overlap and len(chunks) > 1:
        trimmed = []
        for i, ch in enumerate(chunks):
            if i == 0:
                trimmed.append(ch)
            else:
                trimmed.append(ch[overlap:].strip() if len(ch) > overlap else ch)
        return trimmed
    return chunks

# -------------------------------
# 파서: PDF(조례), HTML(보도자료), CSV(지표)
# -------------------------------

def parse_pdf(path: Path) -> Tuple[str, Dict[str, str]]:
    """pdfplumber → (텍스트, 메타) / 실패 시 OCR(선택)"""
    text = ""
    with pdfplumber.open(str(path)) as pdf:
        pages = []
        for p in pdf.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t:
                pages.append(t)
        text = "\n".join(pages)
    text = normalize_text(text)

    if not text and _OCR_OK:
        # 간단 OCR (큰 PDF는 비용↑ 주의)
        import tempfile
        from pdf2image import convert_from_path
        with tempfile.TemporaryDirectory() as td:
            images = convert_from_path(str(path), dpi=200, output_folder=td)
            ocr_texts = []
            for im in images:
                ocr_texts.append(pytesseract.image_to_string(im, lang="kor+eng"))
            text = normalize_text("\n".join(ocr_texts))

    meta = {"source": path.name}
    # 조례명 후보
    m = LAW_TITLE_PAT.search(text)
    if m:
        meta["law_title"] = normalize_text(m.group(1))
    return text, meta


def parse_html(path: Path) -> Tuple[str, Dict[str, str]]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    doc = Document(html)
    content_html = doc.summary()
    soup = BeautifulSoup(content_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = normalize_text(soup.get_text(" "))
    # 제목/날짜 추정
    title = normalize_text(doc.short_title() or path.stem)
    date = None
    # 파일명에서 날짜 추출 시도
    m = re.search(r"(20\d{2}[.-]?\d{1,2}[.-]?\d{1,2})", path.name)
    if m:
        try:
            date = dateparser.parse(m.group(1)).date().isoformat()
        except Exception:
            pass
    return text, {"source": path.name, "title": title, "date": date}


def parse_csv(path: Path, max_rows: int = 120) -> Tuple[str, Dict[str, str]]:
    df = pd.read_csv(path)
    # 샘플링 + 요약 문자열 생성(민감정보 없는 지표 위주 가정)
    head = df.head(max_rows)
    summary = head.to_markdown(index=False)
    return summary, {"source": path.name, "columns": ",".join(df.columns[:10])}

# -------------------------------
# 샘플 생성기
# -------------------------------

@dataclass
class Sample:
    messages: List[Dict[str, str]]
    meta: Dict[str, str]


def build_messages_from_chunk(kind: str, chunk: str, meta: Dict[str, str]) -> Sample:
    chunk = chunk.strip()
    if not chunk:
        raise ValueError("empty chunk")
    if kind == "law":
        user_t = random.choice(LAW_USER_TEMPLATES).format(chunk=chunk)
    elif kind == "press":
        user_t = random.choice(PRESS_USER_TEMPLATES).format(chunk=chunk)
    else:
        user_t = random.choice(CSV_USER_TEMPLATES).format(chunk=chunk)

    # 규칙 기반 어시스턴트 초안: 원문 핵심문장 + 구조적 요약 뼈대
    # (LLM 미사용 환경을 고려하여 간단화)
    sents = sent_split(chunk)[:5]
    key = " ".join(sents)
    assistant = (
        f"요약: {key}\n\n"
        f"핵심 포인트:\n- 주요 내용: 원문 핵심을 요약하여 제시했습니다.\n"
        f"- 적용 범위/대상: 원문에서 명시된 범위를 준용합니다.\n"
        f"근거: {meta.get('law_title') or meta.get('title') or meta.get('source')}"
    ).strip()

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_t},
        {"role": "assistant", "content": assistant}
    ]
    # meta 축약
    m = {k: v for k, v in meta.items() if v}
    return Sample(messages=msgs, meta=m)

# -------------------------------
# 메인: 폴더 순회 → 파싱 → 청킹 → 샘플 생성
# -------------------------------

def iter_files(d: Optional[str], exts: Tuple[str, ...]) -> List[Path]:
    if not d:
        return []
    p = Path(d)
    if not p.exists():
        return []
    files = []
    for ext in exts:
        files.extend(p.rglob(f"*.{ext}"))
    return sorted(files)


def build_dataset(laws_dir: Optional[str], press_dir: Optional[str], csv_dir: Optional[str],
                  max_chars: int = 1200, overlap: int = 120,
                  max_per_source: int = 20) -> List[Sample]:
    samples: List[Sample] = []

    # 1) 조례/규칙 PDF
    for fp in iter_files(laws_dir, ("pdf",)):
        try:
            text, meta = parse_pdf(fp)
            if not text:
                continue
            chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)[:max_per_source]
            for i, ch in enumerate(chunks):
                m = dict(meta)
                m.update({"source": f"{fp.name}#chunk{i+1}"})
                samples.append(build_messages_from_chunk("law", ch, m))
        except Exception as e:
            print(f"[WARN] PDF parse failed: {fp} -> {e}")

    # 2) 보도자료 HTML
    for fp in iter_files(press_dir, ("html", "htm")):
        try:
            text, meta = parse_html(fp)
            if not text:
                continue
            chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)[:max_per_source]
            for i, ch in enumerate(chunks):
                m = dict(meta)
                m.update({"source": f"{fp.name}#chunk{i+1}"})
                samples.append(build_messages_from_chunk("press", ch, m))
        except Exception as e:
            print(f"[WARN] HTML parse failed: {fp} -> {e}")

    # 3) 공공데이터 CSV
    for fp in iter_files(csv_dir, ("csv",)):
        try:
            text, meta = parse_csv(fp)
            if not text:
                continue
            # 표는 한 덩어리로 취급 (max_per_source 적용)
            for i in range(min(3, max_per_source)):
                m = dict(meta)
                m.update({"source": f"{fp.name}#table"})
                samples.append(build_messages_from_chunk("csv", text, m))
        except Exception as e:
            print(f"[WARN] CSV parse failed: {fp} -> {e}")

    return samples


def shuffle_and_split(samples: List[Sample], train_ratio: float = 0.85, seed: int = 42) -> Tuple[List[Sample], List[Sample]]:
    random.Random(seed).shuffle(samples)
    n = int(len(samples) * train_ratio)
    return samples[:n], samples[n:]


def write_jsonl(path: str, samples: List[Sample]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            rec = {"messages": s.messages, "meta": s.meta}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laws", type=str, default=None, help="조례/규칙 PDF 폴더")
    ap.add_argument("--press", type=str, default=None, help="보도자료 HTML 폴더")
    ap.add_argument("--csv", type=str, default=None, help="공공데이터 CSV 폴더")
    ap.add_argument("--out", type=str, required=True, help="학습셋 json 경로")
    ap.add_argument("--eval", type=str, required=True, help="검증셋 json 경로")
    ap.add_argument("--train-ratio", type=float, default=0.85)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--max-per-source", type=int, default=20)
    args = ap.parse_args()

    samples = build_dataset(
        laws_dir=args.laws,
        press_dir=args.press,
        csv_dir=args.csv,
        max_chars=args.max_chars,
        overlap=args.overlap,
        max_per_source=args.max_per_source,
    )

    if not samples:
        print("[ERROR] 생성된 샘플이 없습니다. 입력 경로를 확인하세요.")
        return

    train, eval_ = shuffle_and_split(samples, train_ratio=args.train_ratio)
    write_jsonl(args.out, train)
    write_jsonl(args.eval, eval_)

    print(f"[OK] samples={len(samples)} train={len(train)} eval={len(eval_)}")
    print(f" - train: {args.out}")
    print(f" - eval : {args.eval}")


if __name__ == "__main__":
    main()
