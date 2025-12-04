import os
import pathlib, streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAIVectorStore
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# -----------------------------------
# PAGE SETUP — AI TECHNO FEST THEME
# -----------------------------------
st.set_page_config(page_title=" Customer Support Chatbot ", layout="wide")

st.markdown(
    """
    <style>

    /* Import futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 20% 20%, #11001f, #000000 70%);
        color: #d6eaff;
        font-family: 'Orbitron', sans-serif;
    }

    /* Hide Streamlit default header */
    header {visibility: hidden;}

    .fest-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        color: #00eaff;
        text-shadow: 0 0 18px #00eaff;
        margin-top: -40px;
        margin-bottom: 5px;
        font-family: 'Orbitron', sans-serif;
    }

    .fest-subtitle {
        text-align: center;
        font-size: 18px;
        color: #a2e8ff;
        margin-bottom: 25px;
    }

    .chat-bubble-user {
        padding: 14px;
        background: linear-gradient(90deg, #2d0054, #52008a);
        border-left: 5px solid #b44bff;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #f4eaff;
        box-shadow: 0 0 8px #52008a;
    }

    .chat-bubble-bot {
        padding: 14px;
        background: linear-gradient(90deg, #003b49, #005f73);
        border-left: 5px solid #00eaff;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #e7faff;
        box-shadow: 0 0 8px #00eaff;
    }

    .disclaimer-box {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #ffdd00;
        padding: 12px;
        margin-bottom: 15px;
        border-radius: 6px;
    }

    </style>

    <div class="fest-title">⚡ Customer Support Chatbot</div>
    <div class="fest-subtitle">AI Techno Fest Edition • Powering Answers with AI</div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# DISCLAIMER WIDGET
# -------------------------
with st.expander("⚠️ Disclaimer"):
    st.markdown("""
    <div class="disclaimer-box">
    This chatbot is an AI assistant built for demonstration at the AI Techno Fest.  
    Responses may be inaccurate or incomplete.  
    Do not use this tool for legal, financial or sensitive personal decisions.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# SWITCHED: FAISS → OPENAI VECTOR STORE
# ------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def init_chain():

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Load or create OpenAI Vector Store
    # Assumes you already uploaded documents to OpenAI Vector Store separately
    vector_store = OpenAIVectorStore(
        embedding=embeddings,
        collection_name="UserDocs"  # your vector DB name
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(
        model="gpt-4.1",
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
# CHAT INPUT
# -----------------------------------
question = st.chat_input("💬 How can we help you?")

if question:
    with st.spinner("⚡ Processing with AI Techno Core..."):
        response = chain({
            "question": question,
            "chat_history": st.session_state.history,
        })
    st.session_state.history.append((question, response["answer"]))

# ------------------------------
# Chat display with neon bubbles
# ------------------------------
for user, bot in reversed(st.session_state.history):
    st.markdown(
        f"<div class='chat-bubble-user'><b>You:</b> {user}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='chat-bubble-bot'><b>AI:</b> {bot}</div>",
        unsafe_allow_html=True
    )
