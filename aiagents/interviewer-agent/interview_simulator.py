import asyncio
import json
import os
import pdb
from typing import Optional, Literal, List

import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from openai.helpers import LocalAudioPlayer
from pydantic import BaseModel
from typing_extensions import Annotated
import base64
from openai import AsyncOpenAI
# from pydub import AudioSegment
import tempfile

from graph import convert_graph_as_agent

try:
    import subprocess
    import tempfile

    HALLUCINATION_CHECK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not setup hallucination check: {e}")
    HALLUCINATION_CHECK_AVAILABLE = False


hallucination_agent = convert_graph_as_agent()

# ----------------------
# State
# ----------------------
class InterviewAgentState(BaseModel):
    resume_file_path: str
    resume: Optional[str]
    job_posting_url: str
    job_description: Optional[str]
    language: Optional[Literal['ko', 'en']] = 'ko'
    interview_type: Optional[Literal['behavioral', 'technical', 'general']]
    questions: List[str] = []
    verified_questions: List[str] = []
    question_index: int = 1
    current_question: Optional[str]
    answers: List[str] = []
    messages: Annotated[List[BaseMessage], add_messages]


# ----------------------
# Tools
# ----------------------
def parse_resume_file(state: InterviewAgentState) -> dict:
    print('parse_resume_file')
    ext = os.path.splitext(state.resume_file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(state.resume_file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".txt":
        with open(state.resume_file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    return {"resume": text}


def fetch_job_description(state: InterviewAgentState) -> dict:
    print('fetch_job_description')
    response = requests.get(state.job_posting_url)
    return {"job_description": response.text[:2000]}


def analyze_resume_and_job(state: InterviewAgentState) -> dict:
    print('analyze_resume_and_job')
    jd = state.job_description or ""
    resume = state.resume or ""
    interview_type = "technical" if "Python" in jd or "backend" in jd.lower() else "general"
    language = "ko" if "한글" in resume or "한글" in jd else "en"
    return {"interview_type": interview_type, "language": language}


def generate_mock_questions(state: InterviewAgentState) -> dict:
    print('generate_mock_questions')
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = f"""
You are a mock interviewer. Based on the following resume and job description, generate 5 {state.interview_type} interview questions in Korean.

Resume:
{state.resume}

Job Description:
{state.job_description}
"""
    new_message = HumanMessage(content=prompt)
    response = llm.invoke([new_message])
    questions = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    messages = state.messages + [new_message, AIMessage(content=response.content)]
    return {"questions": questions, "question_index": 0, "messages": messages}


def verify_questions(state: InterviewAgentState) -> dict:
    print('verify_questions with agent')
    verified_questions = []

    for question in state.questions[1:]:
        if not question.strip():
            continue

        result = hallucination_agent.run_sync(question)
        print('======================')
        print(f"Question: {question}")
        print(f"Result: {result}")
        print('======================')

        try:
            result_data = json.loads(result.parts[0].content)
            hallucination_score = result_data.get('score', 0.0)
            if hallucination_score < 0.7:
                verified_questions.append(question)
            else:
                print(f"Filtered out: {question} (score: {hallucination_score})")
        except Exception as e:
            print(f"Failed to verify question: {question} with error: {e}")
            verified_questions.append(question)

    return {"verified_questions": verified_questions}


def present_question(state: InterviewAgentState) -> InterviewAgentState:
    print('present_question')
    questions = state.verified_questions if state.verified_questions else state.questions
    question_index = state.question_index
    if questions and question_index < len(questions):
        current_question = questions[question_index]
        question_index += 1
    else:
        current_question = None
    state.current_question = current_question
    state.question_index = question_index
    return state


def generate_tts_audio(question: str, voice="alloy") -> str:
    """
    Generate TTS audio for the given question using OpenAI and return path to audio file (mp3).
    voice: alloy, nova, shimmer, echo
    """
    client = AsyncOpenAI()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def create_audio():
        async with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=question,
                instructions="Speak in a professional and warm tone.",
                response_format="pcm",
        ) as response:
            await LocalAudioPlayer().play(response)
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".pcm") as f_pcm:
            #     async for chunk in response.iter_bytes():
            #         f_pcm.write(chunk)
            #     pcm_path = f_pcm.name
            #
            # # Convert PCM to MP3 using pydub
            # audio = AudioSegment.from_file(pcm_path, format="s16le", frame_rate=24000, channels=1)
            # mp3_path = pcm_path.replace(".pcm", ".mp3")
            # audio.export(mp3_path, format="mp3")
            # return mp3_path

    return loop.run_until_complete(create_audio())


@st.cache_data
def generate_tts_audio_cached(question: str):
    return generate_tts_audio(question)


# ----------------------
# Graph builder
# ----------------------
def get_graph():
    graph = StateGraph(InterviewAgentState)
    graph.add_node("parse_resume_file", parse_resume_file)
    graph.add_node("fetch_job_description", fetch_job_description)
    graph.add_node("analyze_resume_and_job", analyze_resume_and_job)
    graph.add_node("generate_mock_questions", generate_mock_questions)
    # graph.add_node("verify_questions", verify_questions)
    graph.add_node("present_question", present_question)

    graph.add_edge(START, "parse_resume_file")
    graph.add_edge("parse_resume_file", "fetch_job_description")
    graph.add_edge("fetch_job_description", "analyze_resume_and_job")
    graph.add_edge("analyze_resume_and_job", "generate_mock_questions")
    # graph.add_edge("generate_mock_questions", "verify_questions")
    # graph.add_edge("verify_questions", "present_question")
    graph.add_edge("generate_mock_questions", "present_question")
    graph.add_edge("present_question", END)

    return graph.compile()


# ----------------------
# CLI Interface
# ----------------------
def run_cli(resume_path: str, job_url: str):
    initial_state = InterviewAgentState(
        resume_file_path=resume_path,
        job_posting_url=job_url,
        resume=None,
        job_description=None,
        language=None,
        interview_type=None,
        questions=[],
        verified_questions=[],
        question_index=0,
        current_question=None,
        answers=[],
        messages=[]
    )
    graph = get_graph()
    state = graph.invoke(initial_state)

    while state.get('current_question'):
        print(f"\nQuestion {state.get('question_index')}: {state.get('current_question')}")
        answer = input("Your answer: ")
        state.get('answers').append(answer)
        update = present_question(InterviewAgentState(**state))
        state.update(update)
    print("\n✅ Interview complete! Here's a summary:\n")

    questions_to_show = state.get("verified_questions", []) or state.get("questions", [])
    for i, question in enumerate(questions_to_show):
        print(f"Q{i + 1}: {question}")
        print(f"A{i + 1}: {state.get('answers')[i] if i < len(state.get('answers')) else '[No answer]'}\n")


# ----------------------
# Streamlit UI
# ----------------------
st.title("🎤 AI Mock Interview Simulator")

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

uploaded = st.file_uploader("Upload your resume (.pdf or .txt)", type=["pdf", "txt"])
job_url = st.text_input("Job Posting URL")

if st.button("Start Interview") and uploaded and job_url:
    temp_path = os.path.join("temp_resume." + uploaded.name.split(".")[-1])
    with open(temp_path, "wb") as f:
        f.write(uploaded.read())

    initial_state = {
        "resume_file_path": temp_path,
        "job_posting_url": job_url,
        "resume": None,
        "job_description": None,
        "language": None,
        "interview_type": None,
        "questions": [],
        "verified_questions": [],
        "question_index": 0,
        "current_question": None,
        "answers": [],
        "messages": []
    }
    print('Start Interview')
    graph = get_graph()
    state = graph.invoke(initial_state)
    st.session_state.agent_state = state
    st.success("Interview started!")

if st.session_state.agent_state:
    state = st.session_state.agent_state
    if state["current_question"]:
        question = state["current_question"]
        st.subheader(f"Q{state['question_index']}: {question}")

        if st.button("▶ Play"):
            with st.spinner("Generating voice..."):
                audio_path = generate_tts_audio(question)
                # audio_file = open(audio_path, "rb")
                # audio_bytes = audio_file.read()
                # st.audio(audio_bytes, format="audio/mp3")

        answer = st.text_area("Your Answer")
        if st.button("Next Question"):
            if answer.strip():
                state["answers"].append(answer.strip())
            update = present_question(InterviewAgentState(**state))
            state.update(update)
            st.session_state.agent_state = state
            st.rerun()
    else:
        st.success("✅ Interview completed!")
        st.markdown("### Your Answers:")
        questions_to_show = state.get("verified_questions", []) or state.get("questions", [])
        for i, q in enumerate(questions_to_show):
            st.markdown(f"**Q{i + 1}:** {q}")
            st.markdown(f"> {state['answers'][i] if i < len(state['answers']) else '_No answer_'}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the AI Mock Interview CLI")
    parser.add_argument("--resume", required=True, help="Path to resume file (.pdf or .txt)")
    parser.add_argument("--job", required=True, help="Job posting URL")
    args = parser.parse_args()
    run_cli(args.resume, args.job)
