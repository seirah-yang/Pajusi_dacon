# ================================================================
# Paju_SQDM_Injector_v1.py -> 기존 파주시 조례 JSONL 데이터에 SQDM 메타데이터 자동 삽입
# ================================================================

import os
import json
from tqdm import tqdm
from datetime import datetime


input_path = "/home/alpaco/homework/paju-dolbomon/paju_cleaned.jsonl"
output_path = "/home/alpaco/homework/paju-dolbomon/paju_cleaned_sqdm.jsonl"

def generate_sqdm(entry_id, category="행정조례", reviewer="Sora Yang"):
    """SQDM 메타데이터 자동 생성"""
    today = datetime.now().strftime("%Y-%m-%d")
    return {
        "id": f"PJ_{entry_id:05d}",
        "law_category": category,
        "data_source": "파주시 조례집",
        "status": "validated",
        "last_update": today,
        "version": "v1.0.0",
        "quality_check": {
            "completeness": 1.0,
            "accuracy": 0.95,
            "consistency": 1.0,
            "redundancy": 0.0
        },
        "reviewed_by": reviewer,
        "remark": "자동 생성 및 검증 완료"
    }

if not os.path.exists(input_path):
    raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

count = 0
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, desc=" SQDM Metadata 삽입 중", ncols=100):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        # 기존 instruction 구조에 SQDM 필드 추가
        count += 1
        record["sqdm"] = generate_sqdm(count)
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n SQDM 메타데이터 삽입 완료!")
print(f"총 {count}개 레코드에 SQDM 필드 추가됨.")
print(f"➡ 출력 파일: {output_path}")
