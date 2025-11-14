# rag_service.py
# 파주시 전용 RAG MVP (JSONL 기반, top1 전용)

from typing import List, Dict
from pathlib import Path
import json
import re
from rank_bm25 import BM25Okapi

# JSONL 데이터 경로
DATA_PATHS = {
    "cleaned": Path("/home/alpaco/homework/kimcy/data/paju_cleaned.jsonl"),
    "carecenter": Path("/home/alpaco/homework/kimcy/data/paju_public_health.json")
}


def _load_paju_docs() -> List[Dict[str, str]]:
    """
    - cleaned(.jsonl): 조례/규정 문서
    - carecenter(.json): 보건소 연락처/인텐트 문서
    """
    docs: List[Dict[str, str]] = []

    for name, path in DATA_PATHS.items():
        if not path.exists():
            print(f"[WARN] Paju RAG data file not found: {path}")
            continue

        print(f"[INFO] Loading: {name} ({path})")

        # 1) JSON Lines 처리 (.jsonl)
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    doc = _normalize_doc(obj, source=name)
                    if doc:
                        docs.append(doc)

        # 2) 일반 JSON 처리 (.json)
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[WARN] JSON decode failed: {path}")
                    continue

            #  {"health_center": [ {...}, ... ]}
            if isinstance(data, dict) and "health_center" in data:
                for obj in data["health_center"]:
                    if not isinstance(obj, dict):
                        continue
                    doc = _normalize_doc(obj, source=name)
                    if doc:
                        docs.append(doc)

            # "인텐트 → contacts" 구조
            elif isinstance(data, dict) and name == "carecenter":
                for intent_key, node in data.items():
                    if not isinstance(node, dict):
                        continue

                    desc = str(node.get("description", "")).strip()
                    contacts = node.get("contacts", [])
                    if not isinstance(contacts, list):
                        continue

                    for contact in contacts:
                        if not isinstance(contact, dict):
                            continue

                        # contact dict 복사해서 search_keywords 붙여주기
                        obj = dict(contact)  # shallow copy
                        extra_kw = " ".join([intent_key, desc]).strip()
                        # 기존 search_keywords가 있으면 뒤에 붙이고, 없으면 새로 생성
                        base_kw = str(obj.get("search_keywords", "")).strip()
                        if base_kw:
                            obj["search_keywords"] = f"{base_kw} {extra_kw}"
                        else:
                            obj["search_keywords"] = extra_kw

                        doc = _normalize_doc(obj, source=name)
                        if doc:
                            docs.append(doc)

            # (b) 리스트 전체가 문서 리스트인 경우
            elif isinstance(data, list):
                for obj in data:
                    if not isinstance(obj, dict):
                        continue
                    doc = _normalize_doc(obj, source=name)
                    if doc:
                        docs.append(doc)

            # (c) 단일 dict 문서
            elif isinstance(data, dict):
                doc = _normalize_doc(data, source=name)
                if doc:
                    docs.append(doc)

    print(f"[INFO] Total Loaded Documents: {len(docs)}")
    return docs


def _normalize_doc(obj: Dict[str, str], source: str) -> Dict[str, str] | None:
    """
    - cleaned: instruction/input/output/content 기반
    - carecenter: department/team/phone/duties/search_keywords 기반
    """

    # 1) 보건소 연락처(carecenter) → 한 덩어리 텍스트로 변환
    if source == "carecenter":
        dept = str(obj.get("department", "")).strip()
        team = str(obj.get("team", "")).strip()
        position = str(obj.get("position", "")).strip()
        phone = str(obj.get("phone", "")).strip()
        duties_list = obj.get("duties", [])
        if isinstance(duties_list, list):
            duties = ", ".join(map(str, duties_list))
        else:
            duties = str(duties_list)

        search_keywords = str(obj.get("search_keywords", "")).strip()

        title_parts = [dept, team, position]
        title = " ".join(p for p in title_parts if p) or "파주시 보건소 연락처"

        # content: 검색용 텍스트 블록
        content_lines = [
            f"부서: {dept}" if dept else "",
            f"팀: {team}" if team else "",
            f"직책: {position}" if position else "",
            f"전화번호: {phone}" if phone else "",
            f"주요 업무: {duties}" if duties else "",
            f"검색 키워드: {search_keywords}" if search_keywords else "",
        ]
        content = "\n".join([ln for ln in content_lines if ln]).strip()

        if not content:
            return None

        return {
            "title": title,
            "content": content,
            "instruction": "",   # 연락처 데이터는 instruction/input 없음
            "input": "",
            "source": source,
            # 필요하면 메타 정보 더 붙이기
            "phone": phone,
            "department": dept,
            "team": team,
            "position": position,
        }

    # 2) 기존 규칙 유지
    instruction = str(obj.get("instruction", "")).strip()
    input_text = str(obj.get("input", "")).strip()
    # 규정/조례 데이터는 output 또는 content 필드에서 본문 텍스트를 가져옴
    output_text = str(obj.get("output", obj.get("content", ""))).strip()

    if not output_text:
        return None  # 내용 없는 건 스킵

    # 제목 생성 (기존 규칙 유지)
    if instruction and input_text:
        title = f"{instruction} | {input_text}"
    else:
        title = instruction or input_text or f"파주시 문서 ({source})"

    return {
        "title": title,
        "content": output_text,
        "instruction": instruction,
        "input": input_text,
        "source": source,
    }

# JSONL에서 로드한 실제 문서
PAJU_DOCS: List[Dict[str, str]] = [d for d in _load_paju_docs() if d]

# BM25 인덱스 생성
def _clean_text(text: str) -> str:
    """
    간단한 정규화:
    - 특수문자 제거
    - 소문자 변환
    - 앞뒤 공백 제거
    """
    text = re.sub(r"[^\w가-힣0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def clean_llm_output(text: str) -> str:
    """
    챗봇 최종 출력에 있는 ###, **, * 등 마크다운 제거용.
    BM25 토큰화와는 절대 섞이면 안 됨.
    """
    if not text:
        return ""

    # 마크다운 제거
    for pat in ["***", "**", "*", "###", "##", "#"]:
        text = text.replace(pat, "")

    # 기본 특수문자 정리
    text = re.sub(r"[^\w가-힣0-9\s.,?!\-–]", " ", text)

    # 여러 공백 하나로
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _tokenize(text: str) -> List[str]:
    """BM25용 토큰화 함수 (MVP용: 공백 기준)"""
    return _clean_text(text).split()


# BM25용 텍스트
BM25_TEXTS: List[str] = [
    f"{doc.get('title', '')}\n{doc.get('content', '')}" for doc in PAJU_DOCS
]

# 각 문서를 토큰 리스트로 변환
BM25_TOKENS: List[List[str]] = [
    _tokenize(text) for text in BM25_TEXTS
]

# BM25 엔진 생성
BM25_ENGINE = BM25Okapi(BM25_TOKENS)
print(f"[INFO] BM25 Ready (docs={len(PAJU_DOCS)})")


def _keyword_score(query: str, text: str) -> int:
    """
    아주 단순한 키워드 매칭 점수 (MVP용)
    공백 기준 , 단어들이 text 안에 얼마나 포함되는지 카운트
    """
    q_words = [w for w in query.lower().split() if w]  # 빈 문자열 제거
    t_lower = text.lower()
    return sum(1 for w in q_words if w in t_lower)


def _score_doc(query: str, doc: Dict[str, str]) -> int:
    """
    문서 전체(title+content)를 대상으로 기본 키워드 스코어 +
    조문명(input)이 잘 맞으면 +점수
    """
    title = doc.get("title", "")
    content = doc.get("content", "")
    input_name = doc.get("input", "")

    base_score = _keyword_score(query, f"{title}\n{content}")

    # 질의에 "제1조(목적)" 같은 조문명이 들어가 있고,
    # doc["input"]에 정확히 같은 문자열이 있으면 보너스
    if input_name:
        q = query.strip()
        if input_name in q or q in input_name:
            base_score += 5  # 조문명 매칭 보너스

    return base_score


def retrieve_context(query: str) -> str:
    """
    BM25로 질문과 가장 유사한 문서 1개를 찾아서 문서의 content만 반환(LLM 컨텍스트용)
    """
    if not PAJU_DOCS:
        return ""

    tokens = _tokenize(query)
    if not tokens:
        return ""

    scores = BM25_ENGINE.get_scores(tokens)  # list 또는 np.ndarray
    # numpy 안 쓰고도 안전하게 최대 인덱스 구하기
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    best_doc = PAJU_DOCS[best_idx]
    content = best_doc.get("content", "")
    return content


def retrieve_top1_doc(query: str) -> Dict[str, str]:
    """
    BM25로 찾은 top1 문서 전체를 반환 (title, content, instruction 등 포함)
    - 디버깅이나 부가 정보가 필요할 때 사용
    """
    if not PAJU_DOCS:
        return {}

    tokens = _tokenize(query)
    if not tokens:
        return {}

    scores = BM25_ENGINE.get_scores(tokens)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    return PAJU_DOCS[best_idx]
