# 단계별 점검 및 복구 방법
# 파일이 손상된 경우 (가장 흔함)
#	1.	터미널 또는 노트북 셀에서 아래 명령 실행:

cat -n /home/alpaco/homework/paju_dacon/GPT_UNIEVAL_NLI_Pipeline.py | head -20   # 파일 상단이 정상 주석(""" …) 으로 시작하는지 확인 : 깨진 문자(□, ï, ¿, ¼ 등)가 보이면 인코딩 문제

# 	2.	이런 문자가 보이면, 파일 인코딩을 UTF-8로 다시 저장
# 오른쪽 아래 인코딩 표시 클릭 → Reopen with Encoding → UTF-8
# Ctrl+S로 다시 저장 = 터미널에서 (한 번에 고치기):

iconv -f euc-kr -t utf-8 /home/alpaco/homework/paju_dacon/GPT_UNIEVAL_NLI_Pipeline.py -o fixed.py
mv fixed.py /home/alpaco/homework/paju_dacon/GPT_UNIEVAL_NLI_Pipeline.py

# 문법 오류가 있는 경우
#	1.	아래 명령으로 문법 점검:

python -m py_compile /home/alpaco/homework/paju_dacon/GPT_UNIEVAL_NLI_Pipeline.py

# 문법 오류가 있을시 출력 예시 
File ".../GPT_UNIEVAL_NLI_Pipeline.py", line 250
  def run_pipeline(    # <-- 여기 표시됨
  ^
SyntaxError: invalid syntax

# 이때, 그 줄을 에디터를 열고 수정 

# 파일 고치고 저장한 뒤 다시 로드 
import importlib, GPT_UNIEVAL_NLI_Pipeline
importlib.reload(GPT_UNIEVAL_NLI_Pipeline)
from GPT_UNIEVAL_NLI_Pipeline import run_pipeline
