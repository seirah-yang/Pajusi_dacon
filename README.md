# Pajusi_dacon
# Paju Embedding Tuner

Recursive + Late Hybrid Chunking + BM25/FAISS Hybrid Retrieval.
Grid-search style tuning for chunk size (`max_chars`) and late mixing weight (`late_alpha`).

##아래는 지금까지 업로드한 전체 코드 구조를 ‘한눈에 이해할 수 있도록’ 정리한 간단한 설명입니다.
이 프로젝트는 파주시 어르신·독거노인을 위한 음성 기반 행정·보건 안내 챗봇(KIOSK) 을 만드는 것이며, 구성요소는 STT → RAG → GPT → TTS 로 이어지는 파이프라인입니다.

🔷 전체 구조(한눈에 보기)
[마이크 음성]
   ↓
STT (Whisper / GPT STT)
   ↓
RAG 검색 (BM25 + FAISS)
   ↓
GPT 답변 생성 (SKT ax4)
   ↓
TTS (gTTS)
   ↓
[MP3 음성 출력]

## 1. rag_engine.py — 문서 검색 엔진(RAG 핵심)

### 역할

- 파주시 행정·보건 문서들을 벡터화하여 검색하는 모듈

- 두 가지 검색을 동시에 수행하는 Hybrid RAG

- BM25(전통적 텍스트 검색) + FAISS(딥러닝 임베딩 기반 검색)

### 주요 기능

- HuggingFaceEmbeddings 로 E5-Large 임베딩 불러오기

- 미리 생성된 벡터스토어(./data/vector_store)에서 문서 로드

- BM25 점수와 FAISS 점수를 합쳐 중복 제거 후 최종 RAG 결과 반환

## 2. llm_engine.py — GPT 응답 생성 모듈

### 역할

- STT + RAG 결과를 기반으로 GPT에게 답변을 요청하는 부분

- PT 시스템 프롬프트: “노인에게 설명하듯 쉬운 말로 안내”

### 동작

- RAG에서 가져온 문서 텍스트를 합쳐 GPT에 전달

### 모델: 
- SKT ax4 

- GPT로부터 자연어 답변 생성

## 3. kiosksvc.py — KIOSK 서비스 전체 파이프라인 조립

### 역할

- KIOSK가 실제로 이용하는 핵심 서비스

- STT → RAG → GPT → TTS 전체 프로세스를 순서대로 실행

- 주요 처리 흐름

  음성을 STT로 텍스트 변환 ➔ RAG 검색 ➔ GPT로 답변 생성 ➔ 텍스트 정리(clean_text) ➔ TTS로 mp3 생성 ➔ JSON 형태로 프론트엔드에 전달

## 4. stt_tts.py — 음성 인식(STT) + 음성 합성(TTS)

- STT: OpenAI Whisper 기반

### 모델 
(1) gpt: SKT a.x 4.0

- 업로드된 음성 파일을 텍스트로 변환

(2) TTS : gTTS(구글 TTS) 활용

- 매우 간단하고 빠르게 MP3 파일 생성

## 6. main.py — FastAPI 서버 기본 엔드포인트

- /voice-chat API

- 음성 파일 업로드 → mp3 응답 반환

- 실제 KIOSK는 index.html에서 이 API를 호출함

## 7. paju_public_health.json — 파주시 보건소 조직 데이터

- 모든 팀, 담당자, 전화번호, 담당업무(duties) 저장

- RAG 검색 시 근거로 사용

- 어르신 문의(“누구한테 전화해야 해요?”)에 강력히 활용

## 8. rag_service.py — 파주시 전용 RAG(BM25) 엔진

### 기능: 

- JSONL(행정문서) + JSON(보건소 조직) 통합 로딩

- 불필요한 특수문자 제거(clean_text)

- BM25 기반의 빠른 검색

- GPT에 보낼 context를 구성하는 핵심 모듈

## 9. paju_careon_fastapi.py — 최종 완성된 돌봄ON 챗봇 API

### 역할: 
이 프로젝트의 실제 완성형 버전 SKT A.X GPT(ax4) + 빠른 Whisper + gTTS 통합 HTML UI(index.html)와 연동

### 주요 구성

- STT: faster-whisper

- RAG: retrieve_context() 사용

- GPT: SKT ax4 호출

- TTS: gTTS

- /paju/voice-chat 엔드포인트에서 mp3 + 텍스트 함께 반환

## 10. index.html — KIOSK용 UI(자동 음성 녹음 + 무음 감지)

### 특징: 

- “말하기” 버튼 → STT 전송

- 자동 무음 감지로 말이 끝나면 자동 stop

- GPT 답변과 오디오 재생 표시

- 실제 키오스크 화면처럼 디자인됨

🔷 요약 — “이 프로젝트는 무엇을 하는가?”

이 프로젝트는 다음 기능을 완전 자동화한 KIOSK 음성 챗봇입니다.

1) 시민이 말하면(STT) 파주시 문서들에서 필요한 내용 검색(RAG)

2) GPT가 노인 친화형 문장으로 답변(GPT · ax4)

3) 그 답변을 음성으로 읽어주는(TTS)

4) 키오스크 화면에서 바로 들려주는 UI

즉, “말하면 파주시 행정 정보를 찾아서 읽어주는 AI 도우미(KIOSK)” 입니다. 

## Notes
- `late_alpha` controls the blend between local (chunk) embedding and global (document) embedding:
  - 0.0 → only chunk semantics
  - 1.0 → only document semantics
  - typical: 0.3~0.7
- `max_chars` controls recursive chunk size; 800~1600 works well for administrative docs.
- Retrieval uses BM25 + FAISS with RRF fusion by default.

## Switch to e5-large
Use:
```
--embed_model intfloat/e5-large
```
or keep the recommended multilingual variant for Korean-heavy corpora:
```
--embed_model intfloat/multilingual-e5-large
```

## Next
- Replace the simple precision@k proxy with labeled relevance for robust evaluation.
- Persist FAISS with IVF/HNSW for larger corpora if needed.
