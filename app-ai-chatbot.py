import os
import streamlit as st
from langchain_classic.vectorstores import FAISS
from langchain_classic.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# -------------------------------
# PAGE CONFIG — BLACK & WHITE THEME
# -------------------------------
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# -------------------------------
# BLACK & WHITE THEME STYLING
# -------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    .stApp {
        background-color: #121212;
        color: #f5f5f5;
        font-family: 'Roboto', sans-serif;
    }

    header {visibility: hidden;}

    /* Title & Subtitle */
    .bw-title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 12px #ffffff44, 0 0 24px #ffffff22;
        margin-top: -40px;
        margin-bottom: 5px;
    }

    .bw-subtitle {
        text-align: center;
        font-size: 18px;
        color: #b0b0b0;
        margin-bottom: 25px;
    }

    /* Chat bubbles with halo effect */
    .chat-bubble-user {
        padding: 14px;
        background-color: #1e1e1e;
        border-left: 5px solid #ffffff;
        border-radius: 12px;
        margin-bottom: 10px;
        color: #f5f5f5;
        box-shadow: 0 0 10px #ffffff44;
    }

    .chat-bubble-bot {
        padding: 14px;
        background-color: #2c2c2c;
        border-left: 5px solid #b0b0b0;
        border-radius: 12px;
        margin-bottom: 20px;
        color: #e0e0e0;
        box-shadow: 0 0 10px #b0b0b044;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #1a1a1a;
        border-left: 4px solid #777777;
        padding: 12px;
        margin-bottom: 15px;
        border-radius: 6px;
        color: #cccccc;
    }

    /* Input and checkbox halo */
    .stTextInput>div>div>input {
        background-color: #1e1e1e !important;
        color: #f5f5f5 !important;
        border: 1px solid #555555 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        box-shadow: 0 0 12px #ffffff22;
    }

    .stButton>button {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #555555 !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 0 12px #ffffff22;
        font-weight: 500 !important;
    }

    .stButton>button:hover {
        box-shadow: 0 0 20px #ffffff44 !important;
        border-color: #ffffff !important;
    }

    .stCheckbox label, .stCheckbox div[data-baseweb] {
        color: #f5f5f5 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 12px 0;
        border-top: 1px solid #444444;
        margin-top: 25px;
        font-size: 14px;
        color: #b0b0b0;
    }

    .footer img {
        height: 20px;
        vertical-align: middle;
        margin-right: 8px;
        filter: brightness(0) invert(1); /* white logo */
    }
    </style>

    <div class="bw-title">🤖 AI Chatbot</div>
    <div class="bw-subtitle">Sleek Black & White AI Experience</div>
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
        This AI chatbot is a demonstration tool. Responses may not always be accurate.  
        Avoid using this tool for legal, financial, or sensitive decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# OPENAI / RAG SETUP
# -------------------------
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

# -------------------------
# CHAT INPUT
# -------------------------
st.write("")  # spacing
input_col1, input_col2 = st.columns([0.82, 0.18])

with input_col1:
    question = st.chat_input("💬 Ask your AI...")

with input_col2:
    search_web = st.checkbox("🔍 Search web")

# -------------------------
# HANDLE QUESTION
# -------------------------
if question:
    with st.spinner("⚡ Processing your query..."):
        if search_web:
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.2,
                openai_api_key=OPENAI_API_KEY,
            )
            response_text = llm.invoke(question).content
        else:
            response = chain(
                {
                    "question": question,
                    "chat_history": st.session_state.history,
                }
            )
            response_text = response["answer"]

    st.session_state.history.append(
        (f"{question}  {'(Web Search)' if search_web else '(RAG)'}", response_text)
    )

# -------------------------
# DISPLAY CHAT
# -------------------------
for user, bot in reversed(st.session_state.history):
    st.markdown(
        f"<div class='chat-bubble-user'><b>You:</b> {user}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='chat-bubble-bot'><b>AI:</b> {bot}</div>",
        unsafe_allow_html=True
    )

# -------------------------
# FOOTER WITH SPECTRUM LINK
# -------------------------
st.markdown(
    """
    <div class="footer">
        <a href="https://www.spectrum.com" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Spectrum_logo_2017.svg" alt="Spectrum Logo">
            spectrum.com
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
