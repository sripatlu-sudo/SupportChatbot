import os
import streamlit as st
from langchain_classic.vectorstores import FAISS
from langchain_classic.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS   # <-- DuckDuckGo web search

# -----------------------------------
# PAGE SETUP — AI TECHNO FEST THEME
# -----------------------------------
st.set_page_config(page_title="AI Techno Fest Chatbot", layout="wide")

st.markdown(
    """
    <style>

    /* -----------------------------------------
       🌐 AI TECHNO FEST — FUTURISTIC THEME
       Neon Lights • Dark Mode • Cyber Grid Glow
    ----------------------------------------- */

    body, .stApp {
        background: radial-gradient(circle at 20% 20%, #0b0f17, #02040a 80%);
        font-family: "Orbitron", "Segoe UI", sans-serif;
        color: #E2E8F0;
    }

    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    /* --- TITLE --- */
    .spectrum-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-top: -10px;
        color: #00E5FF;
        letter-spacing: 2px;
        text-shadow:
            0 0 8px #00E5FF,
            0 0 20px rgba(0,229,255,0.7);
    }

    .spectrum-subtitle {
        text-align: center;
        margin-bottom: 25px;
        font-size: 18px;
        color: #10F0FF;
        text-shadow: 0 0 6px rgba(16,240,255,0.7);
    }

    /* --- CHAT BUBBLES --- */
    .chat-bubble-user {
        padding: 14px;
        background: rgba(0, 150, 255, 0.08);
        border-left: 5px solid #00A8FF;
        border-radius: 12px;
        margin-bottom: 10px;
        color: #7DD3FC;
        backdrop-filter: blur(6px);
        box-shadow:
            0 0 10px rgba(0,168,255,0.3),
            inset 0 0 10px rgba(0,168,255,0.1);
    }

    .chat-bubble-bot {
        padding: 14px;
        background: rgba(0,255,180,0.07);
        border-left: 5px solid #00FFC6;
        border-radius: 12px;
        margin-bottom: 20px;
        color: #99F6E4;
        backdrop-filter: blur(6px);
        box-shadow:
            0 0 10px rgba(0,255,198,0.3),
            inset 0 0 10px rgba(0,255,198,0.1);
    }

    /* --- INPUT & BUTTONS --- */
    .stButton>button {
        background-color: #00F2FF !important;
        color: #001A24 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.3rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 0 12px #00F2FF;
    }
    .stButton>button:hover {
        background-color: #00C2CC !important;
        box-shadow: 0 0 18px #00F2FF;
        transform: scale(1.03);
    }

    .stCheckbox label {
        color: #9AE6FF !important;
        font-weight: 600;
        text-shadow: 0 0 6px rgba(0,255,255,0.5);
    }

    .disclaimer-box {
        background: rgba(255, 204, 0, 0.07);
        border-left: 4px solid #FFBB00;
        padding: 12px;
        border-radius: 8px;
        color: #FFEAA7;
        backdrop-filter: blur(4px);
    }

    /* --- FOOTER --- */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(0, 15, 25, 0.8);
        padding: 12px 0;
        text-align: center;
        box-shadow: 0 -4px 15px rgba(0,255,255,0.3);
        border-top: 1px solid rgba(0,255,255,0.3);
        backdrop-filter: blur(8px);
    }

    .footer a {
        font-size: 16px;
        color: #00E5FF;
        text-decoration: none;
        font-weight: 700;
        text-shadow: 0 0 6px #00E5FF;
    }

    </style>

    <div class="spectrum-title">🎧 AI TECHNO FEST CHATBOT</div>
    <div class="spectrum-subtitle">Neon Intelligence • Cyberpunk Support • Live Web AI</div>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# DISCLAIMER WIDGET
# -------------------------
with st.expander("⚠️ Disclaimer"):
    st.markdown(
        """
        <div class="disclaimer-box">
        This chatbot uses AI + web search for assistance.  
        Responses may be incomplete or inaccurate.  
        Do not rely on this system for legal, medical, or financial decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------
# RAG + GPT INITIALIZATION
# -----------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def init_chain():
    vectordb = FAISS.load_local(
        "faiss_index",
        HuggingFaceEmbeddings(model_name="thenlper/gte-small"),
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

chain = init_chain()

if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------------
# DUCKDUCKGO SEARCH FUNCTION
# -----------------------------------
def duckduckgo_search(query, max_results=5):
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


# -----------------------------------
# CHAT INPUT + WEB SEARCH TOGGLE
# -----------------------------------
st.write("")
col1, col2 = st.columns([0.82, 0.18])

with col1:
    question = st.chat_input("💬 Ask me anything...")

with col2:
    search_web = st.checkbox("🔍 Search web")


# -----------------------------------
# PROCESS USER QUESTION
# -----------------------------------
if question:
    with st.spinner("⚡ Running AI engines..."):

        if search_web:
            # ----- DUCKDUCKGO SEARCH -----
            results = duckduckgo_search(question, max_results=5)

            st.markdown("### 🌐 DuckDuckGo Results (Live)")
            for r in results:
                st.write(f"**{r.get('title','')}**")
                st.write(r.get("body", ""))
                st.write("---")

            # Generate LLM Answer
            combined = "\n".join(
                [f"{r.get('title','')}: {r.get('body','')}" for r in results]
            )

            prompt = f"""
            You are an AI assistant answering using LIVE web results.

            WEB SEARCH RESULTS:
            {combined}

            USER QUESTION:
            {question}

            Provide the best possible answer.
            """

            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.2,
                openai_api_key=OPENAI_API_KEY,
            )

            response_text = llm.invoke(prompt).content

        else:
            # ----- VECTOR DB RAG -----
            result = chain({
                "question": question,
                "chat_history": st.session_state.history
            })
            response_text = result["answer"]

    st.session_state.history.append(
        (f"{question}  {'(Web Search)' if search_web else '(RAG)'}",
         response_text)
    )


# -----------------------------------
# CHAT HISTORY UI — NEON STYLE
# -----------------------------------
for user_msg, bot_msg in reversed(st.session_state.history):
    st.markdown(
        f"<div class='chat-bubble-user'><b>You:</b> {user_msg}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='chat-bubble-bot'><b>AI:</b> {bot_msg}</div>",
        unsafe_allow_html=True
    )


# -----------------------------------
# FOOTER — CYBERPUNK STYLE
# -----------------------------------
st.markdown(
    """
    <div class="footer">
        <div class="main-block">
            🔥 AI Techno Fest • Powered by GPT-4 + DuckDuckGo •
            <a href="https://www.ai-chatbot.com" target="_blank">ai-chatbot.com</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
