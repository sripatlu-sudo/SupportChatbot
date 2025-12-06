import os
import pathlib, streamlit as st
#from langchain_classic.vectorstores import FAISS
from langchain_classic.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# -----------------------------------
# PAGE SETUP — BLUE HOLOGRAM MATRIX THEME
# -----------------------------------
st.set_page_config(page_title="Hologram AI Chatbot", layout="wide")

st.markdown(
    """
    <style>

    /* -----------------------------------------
       COLLEGE ADMISSIONS BROCHURE THEME
       Clean • Elegant • Academic
       ----------------------------------------- */

    body, .stApp {
        background-image: url("https://images.unsplash.com/photo-1503676260728-1c00da094a0b");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: "Georgia", "Merriweather", serif;
        color: #1D1A1A;
    }

    /* Overlay for readability */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(255, 255, 255, 0.60);
        z-index: -1;
    }

    /* Main Title */
    .college-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-top: -10px;
        color: #00274C; /* University Navy Blue */
        font-family: "Merriweather", serif;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.15);
    }

    .college-subtitle {
        text-align: center;
        color: #4A4A4A;
        font-size: 18px;
        margin-bottom: 30px;
        font-family: "Georgia";
    }

    /* Chat Bubbles — Brochure Style */
    .chat-bubble-user {
        padding: 16px;
        background: #F7F9FC;
        border-left: 5px solid #00274C;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        font-size: 16px;
    }

    .chat-bubble-bot {
        padding: 16px;
        background: #FFFFFF;
        border-left: 5px solid #FFCB05; /* University Gold */
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-size: 16px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: #FFF4D9;
        border-left: 4px solid #FFCB05;
        padding: 12px;
        border-radius: 6px;
        color: #5A4A3B;
        font-size: 15px;
    }

    /* Inputs */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #A2A2A2;
        background: rgba(255,255,255,0.75);
        font-family: "Georgia";
    }

    .stCheckbox label {
        color: #00274C !important;
        font-weight: 600;
    }

    /* Buttons — Classic Academic Look */
    .stButton>button {
        background-color: #00274C !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.3rem !important;
        border: none !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: "Merriweather", serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
    }

    .stButton>button:hover {
        background-color: #013a73 !important;
    }

    /* History Sidebar */
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        color: #00274C;
        font-family: "Merriweather", serif;
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #00274C;
        padding: 12px 0;
        text-align: center;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.15);
    }

    .footer a {
        font-size: 16px;
        color: #FFCB05;
        font-family: "Georgia";
        font-weight: 600;
        text-decoration: none;
    }

    </style>

    <div class="college-title">🎓 College Picker Chatbot</div>
    <div class="college-subtitle">Your Admissions Information Assistant</div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# DISCLAIMER WIDGET
# -------------------------
with st.expander("⚠️ Disclaimer"):
    st.markdown(
        """
        <div class="disclaimer-box">
        This AI chatbot provides automated information.  
        It may generate incomplete or inaccurate responses.  
        Please verify important details independently.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------
# VECTOR DB & LLM SETUP
# -----------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def chat_response(query):
    client = OpenAI(api_key=OPENAI_API_KEY)
    #query = "What's Deep Research?"
    response = client.responses.create(
    input= query,
    model="gpt-4o-mini",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ['vs_69347f971e348191b597c0bb6b20de9e'],
    }]
    )
    return response



if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------
# SIDEBAR — HOLOGRAM HISTORY PANEL
# -----------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>💠 Chat History</div>", unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.caption("No conversations yet.")

    for idx, (q, a) in enumerate(st.session_state.history):
        if st.button(f"💬 {q[:36]}...", key=f"hist_{idx}"):
            st.session_state.selected_history = (q, a)

    st.markdown("---")

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        #st.experimental_rerun()

if "selected_history" in st.session_state:
    q, a = st.session_state.selected_history
    st.info(f"**Past Query:** {q}\n\n**Response:** {a}")
    del st.session_state.selected_history

# -----------------------------------
# CHAT INPUT
# -----------------------------------
input_col1, input_col2 = st.columns([0.82, 0.18])

with input_col1:
    question = st.chat_input("💬 How can I help you?")

with input_col2:
    search_web = st.checkbox("🌐 Web Search")

if question:
    with st.spinner("💠 Thinking..."):
        if search_web:
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.2,
                openai_api_key=OPENAI_API_KEY,
            )
            response_text = llm.invoke(question).content
        else:
            response = chat_response(question)
            print(response.output[1].content[0].text)
            #response_text = response["answer"]

    st.session_state.history.append(
        (f"{question} {'(Web Search)' if search_web else '(RAG)'}", response.output[1].content[0].text)
    )

# -----------------------------------
# MAIN CHAT DISPLAY
# -----------------------------------
for user, bot in reversed(st.session_state.history):
    st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {user}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble-bot'><b>AI:</b> {bot}</div>", unsafe_allow_html=True)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown(
    """
    <div class="footer">
        <div class="main-block call-box">
        📞 Need more help? Call: <b><a href="tel:(833) 224-6603">833-677-8890</a></b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

#-------------------------------------------#
#----Proposed enhancements & next steps-----#
# Implement prompt engg. concepts
# Switch to from local vertor db to vectorize.io 
# Add more documents to db
# Theme enhancements
# Use A-Q for code optimization / testing ideas / documentation
# Compile evaluation dataset / perform evaluation & fine tuning (better models?)
# Another version for internal / HLD documents with image look-up capability - openai/CLIP
# Other ideas....Add audio based search
# Prepare slide deck
