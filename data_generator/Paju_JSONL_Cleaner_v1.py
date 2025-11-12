# ================================================================
# Paju_JSONL_Cleaner_v1.py
#  조례별 Q/A JSONL을 통합 + 파인튜닝용 정제(cleaning)
# ================================================================

import os
import json
import re
from tqdm import tqdm
import time

for i in tqdm(range(100), ncols=40):
    time.sleep(0.02)


input_dir = "/home/alpaco/homework/paju-dolbomon/outputs"  # 개별 JSONL 파일들이 있는 폴더
output_path = "/home/alpaco/homework/paju-dolbomon/paju_cleaned.jsonl" # 통합+정제 결과


all_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]
print(f" 총 {len(all_files)}개 JSONL 파일을 병합 및 정제합니다.\n")

with open(output_path, "w", encoding="utf-8") as out:
    for file in tqdm(all_files, desc=" Processing Files", ncols=100, colour="cyan"):
        input_path = os.path.join(input_dir, file)

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f" 파일 읽기 오류: {file} → {e}")
            continue

        for line in tqdm(lines, leave=False, desc=f"  ┗ {file}", ncols=90):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # JSON 형식 오류 라인 건너뜀

            # 출력 길이 필터
            if len(d.get("output", "")) < 20 or len(d["output"]) > 500:
                continue

            # 공백 정리
            d["output"] = re.sub(r"\s+", " ", d["output"]).strip()

            # 필수 항목 검사
            if not d.get("instruction") or not d.get("output"):
                continue

            out.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"\n 통합 및 정제 완료!\n➡ 저장 위치: {output_path}")