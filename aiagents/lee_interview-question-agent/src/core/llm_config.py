# src/core/llm_config.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from typing import Type, Any

# 1. 현재 스크립트 파일의 절대 경로를 가져온다.
current_script_path = os.path.abspath(__file__)
# 2. 현재 스크립트 파일이 있는 디렉토리 (예: src\test)를 가져온다.
current_dir = os.path.dirname(current_script_path)

# 3. 프로젝트의 루트 디렉토리 (AGENT) 경로를 계산한다.
#    src\test 에서 두 단계 위로 올라가면 AGENT 디렉토리가 된다.
#    os.path.join을 사용하여 OS별 경로 구분자를 처리한다.
project_root_dir = os.path.join(current_dir, "..", "..")

# 4. .env 파일의 전체 경로를 만든다.
dotenv_path = os.path.join(project_root_dir, ".env")
print(dotenv_path)
# 5. load_dotenv 함수에 계산된 .env 파일 경로를 명시적으로 전달한다.
load_dotenv(dotenv_path=dotenv_path)
GOOGLE_API_KEY = os.getenv("GEMINI_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_KEY 환경 변수가 설정되지 않았습니다. .env 파일에 GEMINI_KEY를 설정해주세요.")

# Gemini LLM 모델 초기화
# 실무 팁: temperature를 0으로 설정하면 정보 추출과 같이 결정론적이고 사실에 기반한 답변을 유도한다.
# 창의적인 답변이 필요할 때는 더 높은 온도를 사용한다.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

# LLM과 Pydantic 스키마를 연결하여 구조화된 출력을 받기 위한 함수
# 이 함수는 ChatPromptTemplate과 LLM을 묶어 구조화된 출력을 반환하는 체인을 만든다.
def get_structured_parser_chain(
    output_schema: Type[Any],
    instruction: str # <-- 이 인자를 다시 추가해야 한다!
):
    """
    Pydantic 스키마에 맞춰 LLM이 구조화된 응답을 생성하도록 지시하는 Runnable 체인을 생성한다.
    'instruction'은 System Message 내용으로 고정되고, 'content' 변수로 Human Message 내용을 받는다.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(instruction), # instruction 매개변수 내용을 직접 사용
        HumanMessagePromptTemplate.from_template("{content}") # 'content' 변수를 기대
    ])
    
    return prompt | llm.with_structured_output(output_schema)