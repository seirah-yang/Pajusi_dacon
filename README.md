# Pajusi_dacon
# Paju Embedding Tuner

Recursive + Late Hybrid Chunking + BM25/FAISS Hybrid Retrieval.
Grid-search style tuning for chunk size (`max_chars`) and late mixing weight (`late_alpha`).

## Install
```bash
pip install sentence-transformers faiss-cpu rank-bm25 pypdf
```

## Prepare
- Put your administrative PDFs under a folder, e.g. `./pdfs`.
- (Optional) Create an evaluation JSON file `eval.json` like:
```json
[
  {"query":"파주시 AI 행정서비스 추진 배경은?", "relevant_keywords":["추진 배경","배경","목적"]},
  {"query":"조례 개정 근거를 알려줘", "relevant_keywords":["조례","개정","근거"]}
]
```

## Run
```bash
python paju_embedding_tuner.py \
  --pdf_dir ./pdfs \
  --embed_model intfloat/multilingual-e5-large \
  --grid_max_chars 800,1200,1600 \
  --grid_late_alpha 0.3,0.5,0.7 \
  --k 5 \
  --eval_json ./eval.json \
  --out_dir ./embed_tuning_out
```

Artifacts:
- `embed_tuning_out/tuning_results.json` — aggregated metrics per setting
- `faiss_*.index` — FAISS IP index (cosine-equivalent)
- `chunks_*.json` and `chunk_meta_*.json` — saved chunks and metadata

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
