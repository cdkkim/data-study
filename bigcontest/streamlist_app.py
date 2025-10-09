import streamlit as st
import google.generativeai as genai
import json
import os

# ─────────────────────────────
# 1. Gemini API 설정
# ─────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.warning("⚠️ 환경변수 GOOGLE_API_KEY가 설정되어 있지 않습니다. Streamlit secrets에 키를 추가하세요.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# ─────────────────────────────
# 2. personas.json 로드
# ─────────────────────────────
def load_personas(file_path="personas.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            personas = json.load(f)
        return personas
    except FileNotFoundError:
        st.error(f"❌ personas.json 파일을 찾을 수 없습니다. 먼저 `prompt_generator.py`를 실행하세요.")
        return []
    except json.JSONDecodeError:
        st.error("❌ personas.json 파일 형식이 잘못되었습니다.")
        return []

personas = load_personas()

# ─────────────────────────────
# 3. Streamlit UI 구성
# ─────────────────────────────
st.set_page_config(page_title="AI 마케팅 전략 컨설턴트 (Gemini)", layout="wide")
st.title("💡 AI 마케팅 전략 컨설턴트")

if not personas:
    st.stop()

# JSON 데이터에서 고유값 추출
industries = sorted({p["업종"] for p in personas})
franchise_types = sorted({p["프랜차이즈여부"] for p in personas})
store_ages = sorted({p["점포연령"] for p in personas})
age_groups = sorted({p["고객연령대"] for p in personas})
behaviors = sorted({p["고객행동"] for p in personas})

col1, col2, col3 = st.columns(3)
industry = col1.selectbox("업종 선택", industries)
franchise = col2.selectbox("점포 형태", franchise_types)
store_age = col3.selectbox("점포 연령", store_ages)

col4, col5 = st.columns(2)
age_group = col4.selectbox("고객 연령대", age_groups)
behavior = col5.selectbox("고객 행동 특성", behaviors)

generate_button = st.button("🔍 전략 생성하기")

# ─────────────────────────────
# 4. Gemini API 호출 함수
# ─────────────────────────────
# https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash
def call_gemini(prompt, model_name="gemini-2.5-flash", temperature=0.6, max_output_tokens=65535):
    try:
        model = genai.GenerativeModel(model_name)
        generation_config = {"temperature": temperature, "max_output_tokens": max_output_tokens}
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        return f"❌ Gemini 호출 오류: {e}"

# ─────────────────────────────
# 5. 선택된 persona → prompt 불러오기
# ─────────────────────────────
def find_prompt(personas, industry, franchise, store_age, age_group, behavior):
    for p in personas:
        if (
            p["업종"] == industry
            and p["프랜차이즈여부"] == franchise
            and p["점포연령"] == store_age
            and p["고객연령대"] == age_group
            and p["고객행동"] == behavior
        ):
            return p["prompt"]
    return None

# ─────────────────────────────
# 6. 실행 및 결과 출력
# ─────────────────────────────
if generate_button:
    selected_prompt = find_prompt(personas, industry, franchise, store_age, age_group, behavior)
    if not selected_prompt:
        st.error("⚠️ 해당 조합의 프롬프트를 찾을 수 없습니다. prompt_generator.py를 확인하세요.")
        st.stop()

    st.subheader("🧩 선택된 페르소나")
    st.markdown(f"""
    - 업종: **{industry}**
    - 형태: **{franchise}**
    - 점포 연령: **{store_age}**
    - 고객 연령대: **{age_group}**
    - 고객 행동: **{behavior}**
    """)

    with st.expander("📜 프롬프트 보기"):
        st.code(selected_prompt, language="markdown")

    st.info("Gemini API 호출 중입니다... ⏳")
    with st.spinner("AI 전략 생성 중..."):
        result = call_gemini(selected_prompt)

    st.markdown("### 📈 생성된 마케팅 전략 결과")
    st.markdown(result)

    st.download_button(
        label="⬇️ 결과 다운로드 (Markdown)",
        data=result,
        file_name=f"marketing_plan_{industry}_{store_age}.md",
        mime="text/markdown",
    )
