# main.py
import io
import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
# file : http 요청시 파일(바이너리 데이터)를 받을 때 사용하는 의존성
# UploadFile : file 의존성을 통해 실제로 전달받은 업로드된 파일의 정보를 담는 객체
# HttpException : FastAPI에서 HTTP 오류 응답을 발생시키기  위한 클래스
# Depends : FastAPI 의존성 주입을 사용하기 위함 (현재는 사용하지 않으므로 제거 가능)
from typing import Optional, List
from pydantic import BaseModel,Field 

# 프로젝트 내부 모듈 임포트 (경로 수정: src.agents -> src.agent)
from src.agents.state import InterviewAgentState, JobPostingInfo, ResumeInfo, InterviewQuestion
from src.agents.graph import build_interview_agent_graph

# LangSmith 설정
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "interview-agent-fastapi-agent")


app = FastAPI(
    title = "AI 면접 Agent API", # 제목을 더 명확하게 변경
    description="이력서와 채용 공고를 기반으로 면접 질문을 생성하고 시뮬레이션하는 Agent API", # 설명 변경
    version="0.1.0"
)

# Agent 그래프 초기화
interview_agent_app = build_interview_agent_graph()



# 응답 모델 정의 (수정: 단일 질문 대신 질문 리스트를 직접 반환)
class InterviewAgentResponse(BaseModel):
    status: str
    message: str
    job_posting_info: Optional[JobPostingInfo] = None
    resume_info: Optional[ResumeInfo] = None
    # current_question 대신 interview_questions 리스트를 반환한다.
    interview_questions: Optional[List[InterviewQuestion]] = Field(default_factory=list, description="생성된 면접 질문 목록")
    # interview_finished, feedback 필드는 제거한다.
    error: Optional[str] = None
    all_messages: Optional[List[dict]] = None

@app.get("/")
async def root():
    return {"message" : "AI 면접 Agent API가 실행중 입니다. /docs에서 API 문서를 확인하세요"}

@app.post("/start_interview/", response_model=InterviewAgentResponse)
async def start_interview(
    job_posting_file: Optional[UploadFile] = File(None, description="채용 공고 PDF/HWP 파일"),
    job_posting_url: Optional[str] = Form(None, description="채용 공고 URL"),
    resume_file: Optional[UploadFile] = File(None, description="이력서 PDF/HWP 파일"),
    # resume_url: Optional[str] = Form(None, description="이력서 URL")
):
    """
    이력서와 채용 공고를 기반으로 면접 질문 5개를 생성합니다.
    """
    initial_state = InterviewAgentState()
    
    if job_posting_file:
        initial_state.job_posting_input = await job_posting_file.read()
        print(f"채용 공고 파일({job_posting_file.filename}) 수신 완료. 크기: {len(initial_state.job_posting_input)} bytes")
    elif job_posting_url:
        initial_state.job_posting_input = job_posting_url
        print(f"채용 공고 URL({job_posting_url}) 수신 완료.")
    else:
        raise HTTPException(status_code=400, detail="채용 공고 파일 또는 URL 중 하나는 반드시 제공되어야 합니다.")

    if resume_file:
        initial_state.resume_input = await resume_file.read()
        print(f"이력서 파일({resume_file.filename}) 수신 완료. 크기: {len(initial_state.resume_input)} bytes")
    elif resume_url:
        initial_state.resume_input = resume_url
        print(f"이력서 URL({resume_url}) 수신 완료.")
    else:
        raise HTTPException(status_code=400, detail="이력서 파일 또는 URL 중 하나는 반드시 제공되어야 합니다.")

    try:
        if interview_agent_app is None:
            print("오류: Agent가 아직 초기화되지 않았습니다. FastAPI startup 로그를 확인하세요.")
            raise HTTPException(status_code=503, detail="Agent가 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.")

        # --- 핵심 변경 부분: astream을 돌며 실행하고, 마지막에 get_state()로 최종 상태를 가져온다. ---
        # LangGraph 실행 설정 (매 호출마다 새로운 실행을 시작하기 위해)
        config = {"configurable": {"thread_id": "some-unique-id-for-this-run"}} 
        # thread_id는 상태를 저장하고 싶을 때 사용되는데, 현재는 stateless이므로 고유한 값 아무거나 주면 됨.
        # 실제로는 UUID.uuid4() 등으로 생성하여 사용한다.
        # LangGraph의 get_state() 메서드를 사용하려면 config가 필요하다.

        # astream을 통해 Agent를 실행하여 모든 노드를 통과시킨다.
        # 여기서 나오는 값들은 중간 델타일 수 있으며, 최종 상태가 아닐 수 있다.
        async for _ in interview_agent_app.astream(
            {"interview_agent_state": initial_state}, 
            config=config # config를 astream에도 전달한다.
        ):
            pass # 루프를 돌면서 모든 노드 실행을 기다린다.

        # 실행이 완료된 후, get_state()를 호출하여 최종 상태를 가져온다.
        # get_state()는 RunnableState 상태 객체를 반환하며, 그 안에 TypedDict가 있다.
        # 주의: get_state()는 RunnableConfig를 인자로 받는다.
        final_state_from_graph = interview_agent_app.get_state(config)
        
        # RunnableState 객체에서 .values를 통해 내부 TypedDict를 가져온다.
        # 이 TypedDict에서 'interview_agent_state' 키를 통해 Pydantic 모델 인스턴스를 추출한다.
        final_agent_state: InterviewAgentState = final_state_from_graph.values["interview_agent_state"]


        if final_agent_state.error_message:
            messages_for_response = [{"type": msg.type, "content": msg.content} for msg in final_agent_state.messages]
            return InterviewAgentResponse(
                status="error",
                message=f"Agent 실행 중 오류 발생: {final_agent_state.error_message}",
                error=final_agent_state.error_message,
                all_messages=messages_for_response
            )
        else:
            messages_for_response = [{"type": msg.type, "content": msg.content} for msg in final_agent_state.messages]
            return InterviewAgentResponse(
                status="success",
                message="면접 질문 생성 완료.",
                job_posting_info=final_agent_state.job_posting_info,
                resume_info=final_agent_state.resume_info,
                interview_questions=final_agent_state.interview_questions,
                all_messages=messages_for_response
            )
    
    except HTTPException as http_exc:
        raise http_exc
    except ValueError as ve:
        print(f"ValueError during interview start: {ve}")
        raise HTTPException(status_code=400, detail=f"데이터 유효성 오류: {ve}")
    except Exception as e:
        print(f"Unhandled Error during interview start: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요. (Debug Info: {e})")