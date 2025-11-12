# ================================================================
# Paju_Instruction_Builder_v2_HybridIngest.py
# TXT + PDF 병합형 조문 기반 질의응답 데이터셋 생성기
# Model: intfloat/multilingual-e5-large
# ================================================================

import os
import re
import json
import pdfplumber
from tqdm import tqdm
import kss
from sentence_transformers import SentenceTransformer


# Hybrid IngestNode (TXT + PDF 병합 로드)
def load_hybrid_document(txt_path, pdf_path=None):
    text_txt, text_pdf = "", ""

    # --- TXT 파일 우선 로드 ---
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text_txt = f.read()

    # --- PDF 보완(img & 표) 로드 ---
    if pdf_path and os.path.exists(pdf_path):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_pdf += page_text + "\n"
        except Exception as e:
            print(f" PDF parsing error in {pdf_path}: {e}")

    # --- TXT 기준 병합 ---
    merged = text_txt.strip()
    if len(text_pdf) > len(text_txt):
        merged += "\n\n[PDF Supplement]\n" + text_pdf.strip()

    return merged.strip()


# NormalizeNode (정규화 및 전처리)
def normalize_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"제\s*(\d+)조", r"\n제\1조", text)  # 조문 분리
    text = re.sub(r"―|–|•", "-", text)
    return text.strip()


# Chunk+EmbedNode (조문 단위 청킹 및 E5-Large 임베딩)
def chunk_and_embed(text, model):
    chunks = []
    articles = re.split(r"(제\d+조\([^)]+\))", text)[1:]

    for i in range(0, len(articles), 2):
        if i + 1 >= len(articles):
            break
        title = articles[i].strip()
        content = articles[i + 1].strip()

        sents = kss.split_sentences(content)
        joined = " ".join(sents)
        emb = model.encode(joined, normalize_embeddings=True)

        chunks.append({
            "article": title,
            "content": joined,
            "embedding": emb.tolist()
        })
    return chunks


# ================================================================
# InstructionBuilderNode (조문 기반 질의·응답 자동 생성)
# ================================================================
def build_instruction(chunks, ordinance_title):
    dataset = []
    for ch in chunks:
        art = ch["article"]
        cont = ch["content"]
        question = f"{ordinance_title}의 {art} 내용은 무엇인가요?"
        answer = f"{ordinance_title} {art} 요약: {cont[:250]}..."
        dataset.append({
            "instruction": question,
            "input": art,
            "output": answer
        })
    return dataset


# ExportNode (JSONL 저장)
def save_jsonl(dataset, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for d in dataset:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Instruction JSONL saved → {output_path}")


# Main Pipeline
def main(input_dir, output_root):
    print(" Loading embedding model")
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    os.makedirs(output_root, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    for txt_name in tqdm(txt_files, desc="Processing documents"):
        base_name = os.path.splitext(txt_name)[0]
        txt_path = os.path.join(input_dir, txt_name)
        pdf_path = os.path.join(input_dir, base_name + ".pdf")

        merged_text = load_hybrid_document(txt_path, pdf_path)
        norm_text = normalize_text(merged_text)
        chunks = chunk_and_embed(norm_text, model)
        dataset = build_instruction(chunks, base_name)

        output_file = os.path.join(
            output_root, f"{base_name}_instruction_dataset.jsonl"
        )
        save_jsonl(dataset, output_file)


# Entry Point
if __name__ == "__main__":
    base_dir = "/home/alpaco/homework/paju-dolbomon/data"
    txt_dir = os.path.join(base_dir, "laws_txt")
    pdf_dir = os.path.join(base_dir, "laws_pdf")
    output_root = "/home/alpaco/homework/outputs"

    print(" Loading embedding model")
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    os.makedirs(output_root, exist_ok=True)
    txt_files = [f for f in os.listdir(txt_dir) if f.lower().endswith(".txt")]
    print(f"📂 Found {len(txt_files)} text files in laws_txt")

    for txt_name in tqdm(txt_files, desc="Processing documents"):
        base_name = os.path.splitext(txt_name)[0]
        txt_path = os.path.join(txt_dir, txt_name)
        pdf_path = os.path.join(pdf_dir, base_name + ".pdf")

        # PDF 존재 확인
        if not os.path.exists(pdf_path):
            print(f" PDF not found for: {base_name}")
            pdf_path = None

        merged_text = load_hybrid_document(txt_path, pdf_path)
        norm_text = normalize_text(merged_text)
        chunks = chunk_and_embed(norm_text, model)
        dataset = build_instruction(chunks, base_name)

        output_file = os.path.join(
            output_root, f"{base_name}_instruction_dataset.jsonl"
        )
        save_jsonl(dataset, output_file)
