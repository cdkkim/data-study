from typing import List
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

# Pydantic의 BaseModel을 임포트한다.
from pydantic import BaseModel, Field # 이 줄을 추가한다.

# 프로젝트 내부 모듈 임포트
from src.agents.state import InterviewAgentState,InterviewQuestion
from src.utils.document_loaders import load_document_from_bytes, load_document_from_url, load_document_from_path # 모든 로더 함수 임포트
from src.utils.llm_parsers import extract_job_posting_info, extract_resume_info
from src.core.llm_config import get_structured_parser_chain, llm as co

async def retrieve_and_parse_documents_node(state: InterviewAgentState) -> InterviewAgentState:
    """
    채용 공고 및 이력서 문서를 로드하고 핵심 정보를 LLM으로 파싱한다.
    """
    print("\n--- [Node: retrieve_and_parse_documents] 문서 로드 및 파싱 시작 ---")
    current_state = state.model_copy()

    job_posting_docs: List[Document] = []
    resume_docs: List[Document] = []
    
    try:
        # 채용 공고 문서 로드
        if current_state.job_posting_input:
            if isinstance(current_state.job_posting_input, str): # URL
                # await 추가
                job_posting_docs = await load_document_from_url(current_state.job_posting_input) 
            elif isinstance(current_state.job_posting_input, bytes): # Bytes
                job_posting_docs = await load_document_from_bytes(current_state.job_posting_input, "pdf") 
            
            if not job_posting_docs:
                raise ValueError("채용 공고 문서를 로드하지 못했습니다.")
            
            # await 추가 (extract_job_posting_info도 비동기 함수)
            current_state.job_posting_info = await extract_job_posting_info(job_posting_docs)
            current_state.messages.append(AIMessage(content="채용 공고 정보 추출 완료."))
            print("채용 공고 정보 추출 완료.")
        else:
            raise ValueError("채용 공고 입력이 없습니다.")

        # 이력서 문서 로드
        if current_state.resume_input:
            if isinstance(current_state.resume_input, str): # URL 또는 로컬 파일 경로
                if current_state.resume_input.startswith("http"):
                    # await 추가
                    resume_docs = await load_document_from_url(current_state.resume_input)
                else: # 로컬 파일 경로 (이 함수는 동기 함수로 가정)
                    # load_document_from_path는 동기 함수이므로 await 불필요 (만약 내부적으로 비동기라면 await 추가)
                    # 이전 가이드에서 동기 함수로 작성했으므로 일단 await 없음
                    resume_docs = load_document_from_path(current_state.resume_input) 
            elif isinstance(current_state.resume_input, bytes): # Bytes
                # await 추가
                # 마찬가지로 파일 타입 힌트 필요
                resume_docs = await load_document_from_bytes(current_state.resume_input, "pdf") # 임시로 "pdf"로 가정
            
            if not resume_docs:
                raise ValueError("이력서 문서를 로드하지 못했습니다.")
            
            # await 추가 (extract_resume_info도 비동기 함수)
            current_state.resume_info = await extract_resume_info(resume_docs)
            current_state.messages.append(AIMessage(content="이력서 정보 추출 완료."))
            print("이력서 정보 추출 완료.")
        else:
            raise ValueError("이력서 입력이 없습니다.")

    except Exception as e:
        current_state.error_message = f"문서 로드 및 파싱 중 오류 발생: {e}"
        current_state.messages.append(AIMessage(content=f"오류: {current_state.error_message}"))
        print(f"오류: {current_state.error_message}")
    
    return current_state


async def generate_interview_questions_node(state: InterviewAgentState) -> InterviewAgentState:
    """
    파싱된 채용 공고 및 이력서 정보를 바탕으로 면접 질문을 생성한다.
    """
    print("\n--- [Node: generate_interview_questions] 면접 질문 생성 시작 ---")
    current_state = state.model_copy()

    if not current_state.job_posting_info or not current_state.resume_info:
        current_state.error_message = "질문 생성을 위한 채용 공고 또는 이력서 정보가 부족합니다."
        current_state.messages.append(AIMessage(content=f"오류: {current_state.error_message}"))
        print(f"오류: {current_state.error_message}")
        return current_state
    
    try:
        # Pydantic 모델을 사용하여 질문 생성을 위한 스키마 정의 (임시)
        # 실제로는 InterviewQuestion 모델 리스트를 생성하도록 유도할 것임
        class QuestionList(BaseModel):
            questions: List[InterviewQuestion] = Field(description="생성된 면접 질문 목록")

        instruction = """
        당신은 채용 담당자이자 경험 많은 면접관입니다.
        아래 제공된 채용 공고 정보와 지원자의 이력서 정보를 바탕으로,
        지원자의 역량을 깊이 있게 평가할 수 있는 5개 내외의 면접 질문을 생성해주세요.
        질문은 지원자의 경험과 채용 공고의 요구사항을 연결하여 구체적이고 심층적으로 질문해야 합니다.
        특히 '왜 이 기술을 써야 하는가', '어떤 상황에 적합한가'와 같은 설계 사고를 유도하는 질문을 5개 포함해주세요.
        질문 유형은 '기술', '경험', '인성', '설계 사고' 등으로 분류하고, 관련 키워드를 포함해주세요.
        응답은 반드시 Pydantic 스키마에 명시된 JSON 형식이어야 합니다.
        """
        
        human_message_content = f"""
        --- 채용 공고 정보 ---
        회사: {current_state.job_posting_info.company_name}
        직무: {current_state.job_posting_info.job_title}
        필수 기술: {', '.join(current_state.job_posting_info.required_skills)}
        주요 업무: {', '.join(current_state.job_posting_info.responsibilities)}
        요약: {current_state.job_posting_info.description_summary}

        --- 지원자 이력서 정보 ---
        이름: {current_state.resume_info.name}
        기술 스택: {', '.join(current_state.resume_info.skills)}
        경력: {current_state.resume_info.experiences}
        프로젝트: {current_state.resume_info.projects}
        요약: {current_state.resume_info.self_introduction_summary}

        위 정보를 바탕으로 면접 질문을 생성해주세요.
        """

        # get_structured_parser_chain 호출 시 instruction만 전달한다.
        chain = get_structured_parser_chain(
            output_schema=QuestionList,
            instruction=instruction # instruction 인자를 다시 전달
        )

       # invoke 호출 시에는 'content' 변수만 전달한다.
        llm_response = await chain.ainvoke({"content": human_message_content})
        
        if isinstance(llm_response, QuestionList):
            current_state.interview_questions = llm_response.questions
            
            # 여기서 5개 질문만 남기도록 한다. (LLM이 더 많이 생성할 경우)
            current_state.interview_questions = current_state.interview_questions[:5] 
            
            if current_state.interview_questions:
                current_state.messages.append(AIMessage(content=f"면접 질문 {len(current_state.interview_questions)}개 생성 완료."))
                print(f"면접 질문 {len(current_state.interview_questions)}개 생성 완료.")
            else:
                current_state.error_message = "LLM이 질문을 생성하지 못했습니다."
                current_state.messages.append(AIMessage(content=f"오류: {current_state.error_message}"))
                print(f"오류: {current_state.error_message}")
        else:
            raise TypeError("LLM 응답이 예상 Pydantic 스키마와 일치하지 않습니다.")

    except Exception as e:
        current_state.error_message = f"면접 질문 생성 중 오류 발생: {e}"
        current_state.messages.append(AIMessage(content=f"오류: {current_state.error_message}"))
        print(f"오류: {current_state.error_message}")
    
    return current_state
