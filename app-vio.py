# app.py
import os
import io
import time
import uuid
import tempfile
import base64
from datetime import datetime

import streamlit as st
from duckduckgo_search import DDGS
from gtts import gTTS
import PyPDF2

# LangChain / FAISS imports
from langchain_classic.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# OpenAI for Whisper transcription (via openai lib)
import openai

# ---------------------------
# Config / API Keys
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("Set OPENAI_API_KEY in environment to enable embeddings, LLM and Whisper features.")
openai.api_key = OPENAI_API_KEY

# FAISS index location
FAISS_INDEX_DIR = "faiss_index"

# ---------------------------
# Page config & Theme
# ---------------------------
st.set_page_config(page_title="AI Techno Fest Chatbot", layout="wide")

st.markdown(
    """
    <style>
    /* AI TECHNO FEST — Neon dark theme */
    body, .stApp {
        background: radial-gradient(circle at 20% 20%, #0b0f17, #02040a 80%);
        font-family: "Orbitron", "Segoe UI", sans-serif;
        color: #E2E8F0;
    }
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .spectrum-title { text-align:center; font-size:38px; color:#00E5FF; text-shadow: 0 0 8px #00E5FF; margin-top:-8px; }
    .spectrum-subtitle { text-align:center; color:#10F0FF; margin-bottom:18px; }
    .chat-bubble-user { padding:12px; background: rgba(0,150,255,0.07); border-left:4px solid #00A8FF; border-radius:10px; color:#7DD3FC; margin-bottom:8px; }
    .chat-bubble-bot { padding:12px; background: rgba(0,255,180,0.06); border-left:4px solid #00FFC6; border-radius:10px; color:#99F6E4; margin-bottom:12px; }
    .disclaimer-box { background: rgba(255,204,0,0.06); border-left:4px solid #FFBB00; padding:10px; border-radius:8px; color:#FFEAA7; }
    .footer { position: fixed; left:0; bottom:0; width:100%; padding:10px 0; text-align:center; background: rgba(0,15,25,0.7); box-shadow: 0 -4px 12px rgba(0,255,255,0.08); }
    .sidebar .stButton>button { background:#00F2FF; color:#001A24; }
    </style>

    <div class="spectrum-title">🎧 AI TECHNO FEST CHATBOT</div>
    <div class="spectrum-subtitle">Neon Intelligence • Live Web + RAG • Voice In/Out</div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helper: DuckDuckGo search
# ---------------------------
def duckduckgo_search(query, max_results=5):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results

# ---------------------------
# Initialize embeddings, LLM, FAISS index & chain
# ---------------------------
@st.experimental_singleton
def init_components():
    # OpenAI embeddings (requires OPENAI_API_KEY env)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Load or create FAISS vectorstore
    if os.path.exists(FAISS_INDEX_DIR):
        try:
            vectordb = FAISS.load_local(FAISS_INDEX_DIR, embeddings)
        except Exception:
            # if load fails, create empty
            vectordb = FAISS.from_texts([], embeddings)
    else:
        vectordb = FAISS.from_texts([], embeddings)

    # Retriever & LLM
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.1, openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return {"embeddings": embeddings, "vectordb": vectordb, "chain": chain, "llm": llm, "memory": memory}

components = init_components()
vectordb = components["vectordb"]
chain = components["chain"]
embeddings = components["embeddings"]
llm = components["llm"]

# ---------------------------
# Session state defaults
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: [{"who":"user"/"bot","text":..,"time":.."}]
if "last_audio_path" not in st.session_state:
    st.session_state.last_audio_path = None

# ---------------------------
# Sidebar: History + Settings
# ---------------------------
with st.sidebar:
    st.header("💾 Chat History")
    if st.session_state.history:
        for msg in reversed(st.session_state.history):
            ts = msg.get("time","")
            who = msg.get("who","")
            txt = msg.get("text","")
            avatar = "🤖" if who=="bot" else "🗣️"
            st.markdown(f"{avatar} **{who.title()}**  •  _{ts}_")
            st.write(txt)
            st.markdown("---")
    else:
        st.info("No messages yet — start the conversation!")

    st.markdown("## ⚙️ Settings")
    enable_web = st.checkbox("Enable DuckDuckGo Web Search", value=False)
    enable_tts = st.checkbox("Enable Text-to-Speech (gTTS)", value=True)
    enable_transcribe = st.checkbox("Enable Audio Transcription (Whisper)", value=True)
    st.markdown("**Model / Embeddings**")
    # Model selection simplified:
    model_choice = st.selectbox("LLM model", ["gpt-4", "gpt-4o"], index=0)
    # Update LLM model in session if changed
    if model_choice != llm.model_name:
        components["llm"] = ChatOpenAI(model_name=model_choice, temperature=0.1, openai_api_key=OPENAI_API_KEY)
        chain.llm = components["llm"]

    st.markdown("---")
    st.markdown("Upload documents (PDF / TXT / MD) to add to FAISS index")
    uploaded = st.file_uploader("Upload file to index", type=["pdf","txt","md"], accept_multiple_files=True)
    if uploaded:
        progress = st.progress(0)
        all_texts = []
        for i, f in enumerate(uploaded):
            name = f.name
            content = None
            if name.lower().endswith(".pdf"):
                try:
                    reader = PyPDF2.PdfReader(f)
                    pages = []
                    for p in reader.pages:
                        pages.append(p.extract_text() or "")
                    content = "\n".join(pages)
                except Exception as e:
                    st.error(f"Failed to read PDF {name}: {e}")
            else:
                content = f.read().decode("utf-8")
            if content:
                all_texts.append(content)
            progress.progress(int((i+1)/len(uploaded)*100))

        if all_texts:
            # chunk & index
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            docs = []
            for txt in all_texts:
                chunks = splitter.split_text(txt)
                docs.extend(chunks)
            # convert to simple texts and index into FAISS
            vectordb.add_texts(docs)
            vectordb.save_local(FAISS_INDEX_DIR)
            st.success("Indexed uploaded documents into FAISS.")
    st.markdown("---")
    st.markdown("Notes:\n- Whisper transcription uses OpenAI audio API (requires key).\n- TTS uses gTTS (Google Text-to-Speech).")

# ---------------------------
# Main: Tabs for Chat / KB / Settings
# ---------------------------
tabs = st.tabs(["💬 Chat", "📚 Knowledge Base", "⚙️ Tools"])

with tabs[0]:
    # Chat UI
    st.markdown("### Live Chat")
    input_col, opt_col = st.columns([0.78, 0.22])
    with input_col:
        user_input = st.chat_input("💬 How can we assist you? (type or use audio upload below)")
    with opt_col:
        use_web_search = st.checkbox("🔍 Search web (DuckDuckGo)", value=enable_web)

    # Audio upload for STT (upload file or recorded file)
    st.markdown("#### 🎤 Audio → Text")
    st.info("Upload an audio file (wav, mp3, m4a). The audio will be transcribed via OpenAI Whisper (if enabled).")
    audio_file = st.file_uploader("Upload audio for transcription", type=["wav","mp3","m4a"], key="audioupload")
    if audio_file and enable_transcribe:
        with st.spinner("Transcribing audio with Whisper..."):
            try:
                # Save to temp file
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1])
                tmp.write(audio_file.read())
                tmp.flush()
                tmp.close()
                # OpenAI whisper transcription
                audio_for_whisper = open(tmp.name, "rb")
                resp = openai.Audio.transcribe("whisper-1", audio_for_whisper)
                transcript = resp.get("text", "")
                st.success("Transcription complete — added to input box.")
                # Set user_input to transcript by writing into session and showing
                # streamlit doesn't let us programmatically set chat_input, so display as text_area for confirmation
                st.text_area("Transcript (copy to chat input to send)", value=transcript, height=120)
                st.session_state._last_transcript = transcript
            except Exception as e:
                st.error(f"Transcription failed: {e}")

    # If user entered text (either typed or from transcript), process
    if user_input or ("_last_transcript" in st.session_state and st.session_state._last_transcript and st.button("Send transcript as question")):
        # prefer typed user_input; fallback to transcript when sending transcript button
        if not user_input and "_last_transcript" in st.session_state:
            question = st.session_state._last_transcript
        else:
            question = user_input

        # show user message immediately
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        st.session_state.history.append({"who":"user","text":question,"time":ts})

        # Compose context from vector DB if available
        with st.spinner("⚡ Running RAG / web search..."):
            context_text = ""
            # Vector DB retrieval
            try:
                retriever = vectordb.as_retriever(search_kwargs={"k":6})
                docs = retriever.get_relevant_documents(question)
                if docs:
                    context_text += "Context from documents:\n"
                    for d in docs:
                        text_snip = (d.page_content[:600] + "...") if len(d.page_content)>600 else d.page_content
                        context_text += f"- {text_snip}\n"
            except Exception as e:
                st.warning(f"Vector retrieval failed: {e}")

            # Optional DuckDuckGo live web search
            web_results = []
            if use_web_search:
                try:
                    web_results = duckduckgo_search(question, max_results=5)
                    context_text += "\nWeb results:\n"
                    for r in web_results:
                        context_text += f"- {r.get('title','')}: {r.get('body','')}\n"
                except Exception as e:
                    st.warning(f"DuckDuckGo search failed: {e}")

            # Build prompt for LLM (concise)
            prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.

CONTEXT:
{context_text}

USER QUESTION:
{question}

Provide a concise, helpful answer. If you used external web results, mention that they were used.
"""
            # call LLM
            try:
                response = chain({"question": question, "chat_history": st.session_state.history})
                answer = response.get("answer", "")
            except Exception as e:
                # fallback direct LLM
                try:
                    components["llm"] = ChatOpenAI(model_name=model_choice, temperature=0.15, openai_api_key=OPENAI_API_KEY)
                    answer = components["llm"].generate([{"role":"user","content":prompt}]).generations[0][0].text
                except Exception as e2:
                    answer = f"Error calling model: {e2}"

        # Save bot reply
        ts2 = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        st.session_state.history.append({"who":"bot","text":answer,"time":ts2})

        # Optionally TTS
        if enable_tts:
            try:
                tts = gTTS(text=answer, lang='en')
                fn = f"/tmp/tts_{str(uuid.uuid4())}.mp3"
                tts.save(fn)
                st.session_state.last_audio_path = fn
            except Exception as e:
                st.warning(f"TTS failed: {e}")

    # Display chat (most recent first)
    st.markdown("### Chat")
    for msg in reversed(st.session_state.history[-40:]):
        who = msg["who"]
        txt = msg["text"]
        ts = msg["time"]
        if who == "user":
            st.markdown(f"<div class='chat-bubble-user'><b>You</b> · _{ts}_<br/>{txt}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-bot'><b>AI</b> · _{ts}_<br/>{txt}</div>", unsafe_allow_html=True)

    # Player for last TTS audio
    if st.session_state.last_audio_path and os.path.exists(st.session_state.last_audio_path):
        st.audio(st.session_state.last_audio_path)

with tabs[1]:
    st.header("📚 Knowledge Base")
    st.write("Current FAISS index summary:")
    try:
        # show number of vectors (approx)
        info = vectordb.index_to_docstore_id
        count = len(info) if info else "unknown"
        st.write(f"Indexed docs (approx): {count}")
    except Exception:
        st.write("No index info available.")

    st.markdown("Upload PDFs or text files in the sidebar to add documents to the index.")

    if st.button("Rebuild index from scratch (clear & re-index uploaded)"):
        # Dangerous: clear vectordb
        vectordb = FAISS.from_texts([], embeddings)
        vectordb.save_local(FAISS_INDEX_DIR)
        components["vectordb"] = vectordb
        st.success("Cleared FAISS index. Upload files in the sidebar to re-index.")

with tabs[2]:
    st.header("⚙️ Tools & Developer Options")
    st.markdown("- Model: " + components["llm"].model_name)
    st.markdown(f"- Embeddings: OpenAIEmbeddings")
    st.markdown("- DuckDuckGo search (toggle in Chat tab / sidebar)")
    st.markdown("### Audio Options")
    st.write("You can upload an audio file in the Chat tab to transcribe via OpenAI Whisper.")
    st.markdown("**Optional**: If you want real-time browser microphone capture, consider adding `streamlit-webrtc` and using on-the-fly audio capture + server transcription. Example (commented) code is available upon request.")

# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    <div class="footer">
        🔥 AI Techno Fest • Powered by GPT-4 + FAISS • <a href="https://www.ai-chatbot.com" target="_blank">ai-chatbot.com</a>
    </div>
    """,
    unsafe_allow_html=True
)
