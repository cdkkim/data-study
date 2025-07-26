import os
import pdb
from typing import Optional, Literal, List

import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel


# ----------------------
# State
# ----------------------
class InterviewAgentState(BaseModel):
    resume_file_path: str
    resume: Optional[str] = None
    job_posting_url: str
    job_description: Optional[str] = None
    language: Optional[Literal['ko', 'en']] = None
    interview_type: Optional[Literal['behavioral', 'technical', 'general']] = None
    questions: Optional[List[str]] = None
    question_index: int = 0
    current_question: Optional[str] = None
    answers: List[str] = []
    messages: List[BaseMessage] = []


# ----------------------
# Tools
# ----------------------
@tool(description="Parse uploaded resume file into plain text (.pdf or .txt only).")
def parse_resume_file(state: InterviewAgentState) -> InterviewAgentState:
    print('parse_resume_file')
    pdb.set_trace()
    ext = os.path.splitext(state.resume_file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(state.resume_file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".txt":
        with open(state.resume_file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    state.resume = text
    return state


@tool(description="Fetch job description text from a given URL.")
def fetch_job_description(state: InterviewAgentState) -> InterviewAgentState:
    print('fetch_job_description')
    pdb.set_trace()
    response = requests.get(state.job_posting_url)
    state.job_description = response.text[:2000]
    return state


@tool(description="Analyze the resume and job description to determine language and interview type.")
def analyze_resume_and_job(state: InterviewAgentState) -> InterviewAgentState:
    print('analyze_resume_and_job')
    pdb.set_trace()
    jd = state.job_description or ""
    resume = state.resume or ""
    state.interview_type = "technical" if "Python" in jd or "backend" in jd.lower() else "general"
    state.language = "ko" if "한글" in resume or "한글" in jd else "en"
    return state


@tool(description="Generate 5 mock interview questions.")
def generate_mock_questions(state: InterviewAgentState) -> InterviewAgentState:
    print('generate_mock_questions')
    pdb.set_trace()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = f"""
You are a mock interviewer. Based on the following resume and job description, generate 5 {state.interview_type} interview questions in {state.language.upper()}.

Resume:
{state.resume}

Job Description:
{state.job_description}
"""
    new_message = HumanMessage(content=prompt)
    response = llm.invoke([new_message])
    questions = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    state.questions = questions
    state.messages.extend([new_message, AIMessage(content=response.content)])
    state.question_index = 0
    return state


@tool(description="Show the next interview question based on the current index.")
def present_question(state: InterviewAgentState) -> InterviewAgentState:
    print('present_question')
    if state.questions and state.question_index < len(state.questions):
        state.current_question = state.questions[state.question_index]
        state.question_index += 1
    else:
        state.current_question = None
    return state


def get_graph():
    graph = StateGraph(InterviewAgentState)

    graph.add_node("parse_resume_file", ToolNode(tools=[parse_resume_file]))
    graph.add_node("fetch_job_description", ToolNode(tools=[fetch_job_description]))
    graph.add_node("analyze_resume_and_job", ToolNode(tools=[analyze_resume_and_job]))
    graph.add_node("generate_mock_questions", ToolNode(tools=[generate_mock_questions]))
    graph.add_node("present_question", ToolNode(tools=[present_question]))

    graph.set_entry_point("parse_resume_file")
    graph.add_edge("parse_resume_file", "fetch_job_description")
    graph.add_edge("fetch_job_description", "analyze_resume_and_job")
    graph.add_edge("analyze_resume_and_job", "generate_mock_questions")
    graph.add_edge("generate_mock_questions", "present_question")

    def should_continue(state: InterviewAgentState) -> str:
        return "continue" if state.questions and state.question_index < len(state.questions) else "done"

    graph.add_conditional_edges(
        "present_question",
        path=should_continue,
        path_map={"continue": "present_question", "done": END}
    )

    return graph.compile()


# ----------------------
# CLI Interface
# ----------------------
def run_cli(resume_path: str, job_url: str):
    state = InterviewAgentState(
        resume_file_path=resume_path,
        job_posting_url=job_url,
        answers=[],
        messages=[AIMessage(
            content='You are a mock interviewer. Please answer the following questions. Use the format: "Q1: Your question here" and "A1: Your answer here". Do not include any additional text.')]
    )
    graph = get_graph()
    state = graph.invoke(state)
    while state.current_question:
        print(f"\nQuestion {state.question_index}: {state.current_question}")
        answer = input("Your answer: ")
        state.answers.append(answer)
        state = graph.invoke(state)
    print("\n✅ Interview complete! Here's a summary:\n")
    for i, question in enumerate(state.questions):
        print(f"Q{i + 1}: {question}")
        print(f"A{i + 1}: {state.answers[i] if i < len(state.answers) else '[No answer]'}\n")


# ----------------------
# Streamlit UI
# ----------------------
st.title("🎤 AI Mock Interview Simulator")

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

uploaded = st.file_uploader("Upload your resume (.pdf or .txt)", type=["pdf", "txt"])
job_url = st.text_input("Job Posting URL")
graph = get_graph()

if st.button("Start Interview") and uploaded and job_url:
    temp_path = os.path.join("temp_resume." + uploaded.name.split(".")[-1])
    with open(temp_path, "wb") as f:
        f.write(uploaded.read())

    initial_state = InterviewAgentState(
        resume_file_path=temp_path,
        job_posting_url=job_url,
        answers=[],
        messages=[
            AIMessage(content='You are a mock interviewer. Please answer the following questions. Use the format: "Q1: Your question here" and "A1: Your answer here". Do not include any additional text.')
        ]
    )
    print('Start Interview')
    pdb.set_trace()
    state = graph.invoke(initial_state)
    st.session_state.agent_state = state
    st.success("Interview started!")

if st.session_state.agent_state:
    state = st.session_state.agent_state
    if state.current_question:
        st.subheader(f"Q{state.question_index}: {state.current_question}")
        answer = st.text_area("Your Answer")
        if st.button("Next Question"):
            if answer.strip():
                state.answers.append(answer.strip())
            state = graph.invoke(state)
            st.session_state.agent_state = state
            st.experimental_rerun()
    else:
        st.success("✅ Interview completed!")
        st.markdown("### Your Answers:")
        for i, q in enumerate(state.questions or []):
            st.markdown(f"**Q{i + 1}:** {q}")
            st.markdown(f"> {state.answers[i] if i < len(state.answers) else '_No answer_'}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the AI Mock Interview CLI")
    parser.add_argument("--resume", required=True, help="Path to resume file (.pdf or .txt)")
    parser.add_argument("--job", required=True, help="Job posting URL")
    args = parser.parse_args()
    run_cli(args.resume, args.job)
