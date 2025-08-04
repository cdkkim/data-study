# src/utils/llm_parsers.py
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import List

from src.core.llm_config import get_structured_parser_chain
from src.agents.state import JobPostingInfo, ResumeInfo


async def extract_job_posting_info(documents:List[Document]) -> JobPostingInfo:
    """
    채용 공고 문서에서 JobPostingInfo 스키마에 맞춰서 정보를 추출
    """
    if not documents:
        raise ValueError("채용 공고 문서가 제공되지 않았습니다.")
    
    full_content = "\n\n".join([doc.page_content for doc in documents])
    
    instruction = """
    당신은 채용 공고를 분석하여 핵심 정보를 추출하는 전문가입니다.
    제공된 채용 공고 내용에서 다음 Pydantic 스키마에 맞춰 모든 필수 정보를 정확하게 추출하고,
    응답은 반드시 json 형식으로 반환합니다. 스키마에 명시된 필드만 포함해야 합니다.
    특히, 'required_skills', 'preferred_skills', 'responsibilities', 'qualifications', 'benefits'는 리스트 형태로 명확히 구분하여 추출해야 합니다.
    'description_summary'는 전체 채용 공고의 핵심 내용을 300자 이내로 요약해야 합니다.
    """

    chain = get_structured_parser_chain(
        output_schema= JobPostingInfo,
        instruction=instruction, 
    )
    print("--- 채용 공고 정보 추출 중 (LLM 호출) ---")
    # invoke 호출 시에는 'content' 변수만 전달한다. (instruction은 이미 체인에 고정됨)
    extracted_info = await chain.ainvoke({"content": full_content})
    print("--- 채용 공고 정보 추출 완료 ---")
    return extracted_info

async def extract_resume_info(documents: List[Document]) -> ResumeInfo:
    """
    이력서 문서에서 ResumeInfo 스키마에 맞춰 정보를 추출한다.
    """
    if not documents:
        raise ValueError("이력서 문서가 제공되지 않았습니다.")

    full_content = "\n\n".join([doc.page_content for doc in documents])

    instruction = """
    당신은 이력서를 분석하여 지원자의 핵심 정보를 추출하는 전문가입니다.
    제공된 이력서 내용에서 다음 Pydantic 스키마에 맞춰 모든 필수 정보를 정확하게 추출하고, 가능한 한 많은 정보를 채워주세요.
    응답은 반드시 JSON 형식이어야 하며, 스키마에 명시된 필드만 포함해야 합니다.
    'education', 'experiences', 'skills', 'projects'는 리스트 형태로 명확히 구분하여 추출해야 합니다.
    특히 'experiences'와 'projects'는 각 항목의 'description'을 상세하게 포함해야 합니다.
    'self_introduction_summary'는 자기소개서 내용이 있다면 300자 이내로 요약하세요.
    """
    # --- 여기를 수정한다: instruction 인자를 다시 전달한다 ---
    chain = get_structured_parser_chain(
        output_schema=ResumeInfo,
        instruction=instruction, # <-- 이 라인이 누락되어 있었음
    )

    print("--- 이력서 정보 추출 중 (LLM 호출) ---")
    # invoke 호출 시에는 'content' 변수만 전달한다.
    extracted_info = await chain.ainvoke({"content": full_content})
    print("--- 이력서 정보 추출 완료 ---")
    return extracted_info