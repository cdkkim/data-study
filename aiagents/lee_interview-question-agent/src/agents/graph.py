# src/agent/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict
# MemorySaver 임포트
from langgraph.checkpoint.memory import MemorySaver 

# 프로젝트 내부 모듈 임포트
from src.agents.state import InterviewAgentState
from src.agents.nodes import (
    retrieve_and_parse_documents_node,
    generate_interview_questions_node,
)

# LangGraph에서 사용할 에이전트의 상태 정의
# TypedDict를 사용하되, InterviewAgentState Pydantic 모델 인스턴스를 그 안에 담는다.
class AgentState(TypedDict):
    """
    LangGraph에서 사용할 에이전트의 상태 정의.
    InterviewAgentState Pydantic 모델을 메인 상태로 관리한다.

    각 노드 함수는 이 TypedDict를 인자로 받지만, 실제로는
    state["interview_agent_state"]에 접근하여 InterviewAgentState Pydantic 모델을 수정하고 반환한다.
    LangGraph는 노드 함수가 반환하는 TypedDict를 기존 상태와 병합한다.
    이때 Pydantic 모델 자체를 통째로 교체하는 것이 가장 간단하고 명확한 방법이다.
    """
    interview_agent_state: InterviewAgentState


def build_interview_agent_graph():
    """
    면접 Agent의 LangGraph 워크플로우를 구축하고 컴파일한다.
    """
    workflow = StateGraph(AgentState)

    # --- 노드 정의 및 연결: 핵심 수정 부분 ---
    # 각 노드 함수는 AgentState (TypedDict)를 입력으로 받고,
    # 변경된 AgentState (TypedDict)를 반환해야 한다.
    # 우리의 노드 함수(retrieve_and_parse_documents_node 등)는 InterviewAgentState를 입력/출력하므로,
    # 이를 LangGraph의 AgentState와 맞추기 위한 어댑터 함수를 정의하고 사용한다.

    # retrieve_and_parse_documents_node는 async 함수이므로, 래퍼도 async여야 한다.
    async def _retrieve_and_parse_documents_wrapper(state: AgentState) -> AgentState:
        # AgentState에서 InterviewAgentState를 추출하여 노드 함수에 전달
        updated_interview_agent_state = await retrieve_and_parse_documents_node(state["interview_agent_state"])
        # 노드 함수의 결과를 다시 AgentState 형태로 래핑하여 반환
        return {"interview_agent_state": updated_interview_agent_state}

    # generate_interview_questions_node는 async 함수이므로, 래퍼도 async여야 한다.
    async def _generate_interview_questions_wrapper(state: AgentState) -> AgentState:
        updated_interview_agent_state = await generate_interview_questions_node(state["interview_agent_state"])
        return {"interview_agent_state": updated_interview_agent_state}


    # 1. 노드 추가 (이제 정의된 비동기 래퍼 함수를 사용)
    workflow.add_node("retrieve_and_parse_documents", _retrieve_and_parse_documents_wrapper)
    workflow.add_node("generate_interview_questions", _generate_interview_questions_wrapper)

    # 2. 시작점 설정
    workflow.set_entry_point("retrieve_and_parse_documents")

       # 3. 조건부 엣지(Edge) 정의: 에러 발생 시 흐름 제어 (유지)
    def decide_on_error(state: AgentState) -> str:
        current_agent_state: InterviewAgentState = state["interview_agent_state"]
        if current_agent_state.error_message:
            print(f"--- [DECIDE] 에러 발생: {current_agent_state.error_message}. END로 전환 ---")
            return "end_with_error"
        else:
            print("--- [DECIDE] 에러 없음. 다음 단계로 진행 ---")
            return "continue"


    # 문서 파싱 후 에러 발생 여부에 따라 분기
    workflow.add_conditional_edges(
        "retrieve_and_parse_documents",
        decide_on_error,
        {
            "continue": "generate_interview_questions", # 에러 없으면 질문 생성
            "end_with_error": END # 에러 발생 시 즉시 종료
        }
    )

    # 질문 생성 후 에러 발생 여부에 따라 분기 (질문 생성 후 바로 END로 연결)
    workflow.add_conditional_edges(
        "generate_interview_questions",
        decide_on_error, # 질문 생성 자체에서 에러가 났으면 종료
        {
            "continue": END, # 에러 없으면 바로 종료 (면접 시뮬레이션 종료)
            "end_with_error": END # 에러 발생 시 즉시 종료
        }
    )
    
    # manage_interview_flow 및 evaluate_answer_and_provide_feedback 관련 엣지들은 모두 제거한다.

    # 그래프 컴파일 (체크포인터 설정)
    # 메모리 기반 체크포인터 설정
    # 이 부분은 그래프를 컴파일하기 전에 정의되어야 한다.
    memory_saver = MemorySaver()
    app = workflow.compile(checkpointer=memory_saver) # compile 메서드에 checkpointer 인자 추가
    print("--- LangGraph Agent 그래프 컴파일 완료 ---")
    return app
