import pathlib, streamlit as st
from langchain_classic.vectorstores import FAISS
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.llms import Ollama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from openai import OpenAI
import base64
import requests

# -------------------------
# CONFIG
# -------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or "YOUR_OPENAI_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Spectrum Support Chatbot", layout="wide")

# -------------------------
# AI TECHNO FEST THEME
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

.stApp {
    background: radial-gradient(circle at top, #0b0f1a, #000000 60%);
    color: #e0e0e0;
    font-family: 'Orbitron', sans-serif;
}

h1 {
    color: #00eaff;
    text-shadow: 0px 0px 12px #00eaff;
    text-align: center;
}

.chat-message {
    padding: 12px;
    margin: 8px 0;
    border-radius: 10px;
    color:#f0f7ff;
}

.user-msg { border-left: 4px solid #00eaff; background: linear-gradient(90deg, #2d0054, #52008a); }
.bot-msg { border-left: 4px solid #ff00ff; background: linear-gradient(90deg, #003b49, #005f73); }

.voice-btn {
    padding:10px 20px;border-radius:8px;background:#00eaff;border:none;color:black;font-weight:bold;cursor:pointer;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚡ Spectrum Support Chatbot — AI Techno Fest</h1>", unsafe_allow_html=True)

# -------------------------
# RAG CHAIN INITIALIZATION
# -------------------------
@st.cache_resource
def init_chain():
    vectordb = FAISS.load_local(
        "faiss_index",
        HuggingFaceEmbeddings(model_name="thenlper/gte-small"),
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    llm = Ollama(model="gemma3:4b", temperature=0.1)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
    )

chain = init_chain()

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# VOICE INPUT COMPONENT
# -------------------------
st.subheader("🎤 Speak instead of typing")
voice_file = st.file_uploader("Upload a short voice note (wav, mp3, m4a)", type=["wav","mp3","m4a"])

if voice_file:
    st.info("Transcribing audio...")
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-tts",
        file=voice_file
    )
    st.session_state.voice_text = transcript.text
    st.success("✅ Transcription complete!")

# -------------------------
# CHAT INPUT (text or voice)
# -------------------------
default_text = st.session_state.get("voice_text", "")
question = st.chat_input("💬 Ask Spectrum Techno Assistant...", value=default_text)

if question:
    with st.spinner("⚡ Thinking..."):
        response = chain({
            "question": question,
            "chat_history": st.session_state.history,
        })
    st.session_state.history.append((question, response["answer"]))
    if "voice_text" in st.session_state:
        del st.session_state.voice_text  # reset after use

# -------------------------
# CHAT HISTORY DISPLAY
# -------------------------
for user, bot in reversed(st.session_state.history):
    st.markdown(f"<div class='chat-message user-msg'><b>You:</b> {user}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-message bot-msg'><b>AI:</b> {bot}</div>", unsafe_allow_html=True)
