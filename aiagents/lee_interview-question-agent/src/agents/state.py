# src/agent/state.py
from typing import List, Optional, Dict, Any, Union
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# Pydantic 모델을 사용하여 채용 공고와 이력서 정보 구조화
class JobPostingInfo(BaseModel):
    """채용 공고에서 추출된 핵심 정보."""
    company_name: str = Field(description="회사 이름")
    job_title: str = Field(description="채용 직무")
    required_skills: List[str] = Field(description="필수 기술 스택")
    preferred_skills: List[str] = Field(default_factory=list, description="우대 기술 스택 (선택 사항)")
    responsibilities: List[str] = Field(description="주요 업무 내용")
    qualifications: List[str] = Field(description="자격 요건")
    benefits: Optional[List[str]] = Field(default_factory=list, description="복리후생 (선택 사항)")
    industry: Optional[str] = Field(default=None, description="회사 산업군 (선택 사항)")
    description_summary: str = Field(description="채용 공고 요약")

class ResumeInfo(BaseModel):
    """이력서에서 추출된 핵심 정보."""
    name: str = Field(description="지원자 이름")
    contact: Optional[str] = Field(default=None, description="연락처 (이메일, 전화번호 등)")
    education: List[Dict[str, str]] = Field(default_factory=list, description="학력 정보 (예: [{'degree': '학사', 'major': '컴퓨터 공학', 'school': 'OO대학교'}])")
    experiences: List[Dict[str, str]] = Field(default_factory=list, description="경력 정보 (예: [{'company': 'XX회사', 'title': '소프트웨어 엔지니어', 'duration': '2020-2023', 'description': '프로젝트 A 개발'}])")
    skills: List[str] = Field(default_factory=list, description="보유 기술 스택")
    projects: List[Dict[str, str]] = Field(default_factory=list, description="주요 프로젝트 경험 (예: [{'name': '프로젝트 B', 'description': '설명', 'technologies': ['Python', 'Django']}])")
    self_introduction_summary: Optional[str] = Field(default=None, description="자기소개서 요약 (선택 사항)")

class InterviewQuestion(BaseModel):
    """생성된 면접 질문."""
    question_id: int = Field(description="질문 ID")
    question_text: str = Field(description="면접 질문 내용")
    question_type: str = Field(description="질문 유형 (e.g., '기술', '경험', '인성', '설계 사고')")
    related_keywords: List[str] = Field(default_factory=list, description="질문과 관련된 키워드 (e.g., 'Python', '데이터베이스 최적화')")



# LangGraph 에이전트의 전체 상태 정의
class InterviewAgentState(BaseModel):
    """
    면접 에이전트의 현재 상태를 나타내는 Pydantic 모델.
    각 노드에서 읽고 쓸 수 있는 모든 정보가 포함된다.
    """
     # 입력 및 기본 정보 (FastAPI 입력과 직접 연결)
    job_posting_input: Optional[Union[str, bytes]] = Field(None, description="채용 공고 URL 또는 파일 내용 (bytes)")
    resume_input: Optional[Union[str, bytes]] = Field(None, description="이력서 파일 경로 또는 파일 내용 (bytes)")

    # 추출된 정보
    job_posting_info: Optional[JobPostingInfo] = Field(None, description="파싱된 채용 공고 정보")
    resume_info: Optional[ResumeInfo] = Field(None, description="파싱된 이력서 정보")

    # 면접 질문 정보 (단일 응답이므로, 최종 결과만 저장)
    interview_questions: List[InterviewQuestion] = Field(default_factory=list, description="생성된 면접 질문 목록")
    
    # 메시지 히스토리와 제어 플래그는 디버깅 및 간결한 에러 로깅을 위해 최소한으로 유지
    messages: List[BaseMessage] = Field(default_factory=list, description="LLM과의 대화 및 Agent 로그 기록")
    error_message: Optional[str] = Field(None, description="발생한 오류 메시지")
    
    class Config:
        arbitrary_types_allowed = True