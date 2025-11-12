# ================================================================
# Paju_IndexBuilder_v1_FAISS_Stable.py
# JSONL 기반 파주시 행정문서 벡터 인덱스 구축 (FAISS + E5-Large)
# ================================================================

import os
import json
import numpy as np
import pandas as pd
import faiss
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


DATA_PATH = "/home/alpaco/homework/paju-dolbomon/paju_cleaned_sqdm.jsonl"   # 입력 JSONL
OUTPUT_DIR = "/home/alpaco/homework/paju-dolbomon/vector_index"             # 출력 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAISS_PATH = os.path.join(OUTPUT_DIR, "paju_faiss.index")
META_PATH = os.path.join(OUTPUT_DIR, "paju_meta.parquet")


print("🔹 Loading embedding model: intfloat/multilingual-e5-large ...")
model = SentenceTransformer("intfloat/multilingual-e5-large")

device = "cuda" if model.device.type == "cuda" else "cpu"
print(f" Model loaded on: {device}")


records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            d = json.loads(line)
            records.append({
                "instruction": d.get("instruction", ""),
                "output": d.get("output", ""),
                "source_file": d["sqdm"]["id"] if "sqdm" in d else "unknown"
            })
        except json.JSONDecodeError:
            continue

df = pd.DataFrame(records)
print(f" Loaded {len(df)} records from {DATA_PATH}")


embeddings = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="🔹 Embedding records", ncols=100):
    text = f"{row['instruction']} {row['output']}"
    emb = model.encode(text, normalize_embeddings=True)
    embeddings.append(emb)

embeddings = np.array(embeddings, dtype="float32")
print(f" Embeddings computed: {embeddings.shape}")


index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product = cosine similarity
index.add(embeddings)

faiss.write_index(index, FAISS_PATH)
df.to_parquet(META_PATH, index=False)

print("\n FAISS 인덱스 및 메타데이터 저장 완료!")
print(f"➡ FAISS 인덱스 파일: {FAISS_PATH}")
print(f"➡ 메타데이터 파일:   {META_PATH}")
