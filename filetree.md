프로젝트 구조

[1] 데이터 생성 · 학습 계층
 ├── Paju_Instruction_Builder_v2.py   조례 기반 Q/A 데이터셋 생성
 ├── Paju_IndexBuilder_v1_FAISS.py                 문서 벡터 인덱스(FAISS) 구축
 └── (Fine-tuning or LangGraph Training)            SKT A.X 4.0 모델 학습

[2] 서비스 계층 (시민 대화용)
 └── paju_careon_fastapi.py                        음성 입력 → STT → GPT → TTS 챗봇 서버


# 연계구조 파이프라인
[Step 1] 조례/PDF 수집
     ↓
[Step 2] Paju_Instruction_Builder_v2_HybridIngest.py
     |
     ↳  (TXT + PDF 병합 → Q/A JSONL 생성: instruction_dataset.jsonl 개별파일이 만들어짐)
     ↓
[Step 3] Paju_IndexBuilder_v1_FAISS.py
     |
     ↳(FAISS 인덱스 구축: 여러 JSONL 합치고 노이즈 제거 ➔ Cleaned.jsonl 1개 파일로 통합)
     ↓
[Step 4] LangGraph 파이프라인 통합
     |
     ↳ (Paju_LangGraph_RAG_v1.py: 질의 -> FAISS검색 -> 답변)
     ↓
[Step 5] SKT API로 모델 재학습 ➔ fine_tuned_ax4.py ➔ Fine-tuned Model ID
     ↳ (시민 음성 질문 → STT → GPT → TTS 응답)
