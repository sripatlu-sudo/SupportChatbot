import os
import pathlib, streamlit as st
#from langchain_classic.vectorstores import FAISS
from langchain_classic.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# -----------------------------------
# PAGE SETUP — ELITE COLLEGE THEME
# -----------------------------------
st.set_page_config(page_title="🎓 Elite College Advisor", layout="wide")

st.markdown(
    """
    <style>

    /* -----------------------------------------
       ELITE COLLEGE ADMISSIONS THEME
       Sophisticated • Prestigious • Academic Excellence
       ----------------------------------------- */

    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

    body, .stApp {
        background: linear-gradient(135deg, 
            #192A56 0%, 
            #8B4513 50%, 
            #192A56 100%);
        font-family: "Crimson Text", "Times New Roman", serif;
        color: #F8F6F0;
    }

    /* Elegant overlay with subtle texture */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(255,215,0,0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255,255,255,0.05) 0%, transparent 50%);
        z-index: -1;
    }

    /* Prestigious Title */
    .college-title {
        text-align: center;
        font-size: 48px;
        font-weight: 700;
        margin-top: -10px;
        color: #FFD700; /* Elegant Gold */
        font-family: "Playfair Display", serif;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        background: linear-gradient(45deg, #FFD700, #FFA500, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .college-subtitle {
        text-align: center;
        color: #E6E6FA;
        font-size: 20px;
        margin-bottom: 35px;
        font-family: "Crimson Text", serif;
        font-style: italic;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }

    /* Elegant Chat Bubbles */
    .chat-bubble-user {
        padding: 20px;
        background: linear-gradient(135deg, rgba(25, 42, 86, 0.9), rgba(25, 42, 86, 0.7));
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-left: 4px solid #FFD700;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        font-size: 16px;
        color: #F8F6F0;
        backdrop-filter: blur(10px);
    }

    .chat-bubble-bot {
        padding: 20px;
        background: linear-gradient(135deg, rgba(139, 69, 19, 0.9), rgba(139, 69, 19, 0.7));
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-left: 4px solid #CD853F;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        font-size: 16px;
        color: #F8F6F0;
        backdrop-filter: blur(10px);
    }

    /* Elegant Disclaimer */
    .disclaimer-box {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.15), rgba(255, 215, 0, 0.05));
        border: 1px solid rgba(255, 215, 0, 0.4);
        border-left: 4px solid #FFD700;
        padding: 16px;
        border-radius: 10px;
        color: #F8F6F0;
        font-size: 15px;
        backdrop-filter: blur(5px);
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    /* Inputs */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid rgba(255, 215, 0, 0.3);
        background: rgba(255,255,255,0.1);
        font-family: "Crimson Text";
        color: #F8F6F0;
    }

    .stCheckbox label {
        color: #FFFFFF !important;
        font-weight: 600;
    }

    /* Premium Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #8B4513, #A0522D) !important;
        color: #FFD700 !important;
        border: 1px solid rgba(255, 215, 0, 0.3) !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: "Playfair Display", serif !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #A0522D, #CD853F) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4) !important;
    }

    /* Elegant Sidebar */
    .sidebar-title {
        font-size: 24px;
        font-weight: 700;
        color: #FFD700;
        font-family: "Playfair Display", serif;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }

    /* Sophisticated Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(135deg, rgba(25, 42, 86, 0.95), rgba(139, 69, 19, 0.95));
        padding: 15px 0;
        text-align: center;
        box-shadow: 0 -4px 15px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255, 215, 0, 0.3);
    }

    .footer a {
        font-size: 17px;
        color: #FFD700;
        font-family: "Playfair Display", serif;
        font-weight: 600;
        text-decoration: none;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }

    </style>

    <div class="college-title">🏛️ Elite College Advisor</div>
    <div class="college-subtitle">Your Gateway to Academic Excellence</div>
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

# Prompt template for context management
COLLEGE_ADVISOR_PROMPT = """
You are an expert college admissions advisor with deep knowledge of higher education. 
Your role is to provide accurate, helpful guidance on college selection, admissions processes, and academic planning.

Context Guidelines:
- Answer ONLY based on the provided document context
- If information is not in the context, say "I don't have specific information about that in my knowledge base"
- Focus on college admissions, academic programs, and student success
- Provide actionable advice when possible
- Be encouraging and supportive in your tone

User Question: {query}

Please provide a comprehensive response based on the available information.
"""

@st.cache_resource
def chat_response(query):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Format query with prompt template
    formatted_query = COLLEGE_ADVISOR_PROMPT.format(query=query)
    
    response = client.responses.create(
        input=formatted_query,
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
# SIDEBAR — ELEGANT HISTORY PANEL
# -----------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📚 Chat History</div>", unsafe_allow_html=True)

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
    with st.spinner("🎓 Thinking..."):
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