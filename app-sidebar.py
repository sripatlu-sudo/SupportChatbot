import os
import pathlib, streamlit as st
from langchain_classic.vectorstores import FAISS
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

    /* -------------------------------
       AMAZON E-COMMERCE THEME
       Clean, functional, bold
       ------------------------------- */

    body, .stApp {
        background: #FFFFFF;
        font-family: "Amazon Ember", "Segoe UI", sans-serif;
        color: #111111;
    }

    /* Main Title */
    .amazon-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        margin-top: -10px;
        color: #146EB4;
        padding-bottom: 5px;
    }

    .amazon-subtitle {
        text-align: center;
        color: #555555;
        font-size: 17px;
        margin-bottom: 28px;
        font-weight: 500;
    }

    /* Chat Bubbles */
    .chat-bubble-user {
        padding: 14px;
        background: #F3F3F3;
        border-left: 5px solid #146EB4;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        color: #111111;
    }

    .chat-bubble-bot {
        padding: 14px;
        background: #FFFFFF;
        border-left: 5px solid #FF9900;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        color: #2a2a2a;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background: #FFF7E6;
        border-left: 4px solid #FF9900;
        padding: 12px;
        border-radius: 6px;
        color: #5A4A3B;
    }

    /* Buttons — Amazon Style */
    .stButton>button {
        background-color: #FF9900 !important;
        color: #111111 !important;
        border-radius: 8px !important;
        padding: 0.7rem 1.3rem !important;
        border: none !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12) !important;
    }

    .stButton>button:hover {
        background-color: #F08804 !important;
    }

    /* Inputs */
    .stTextInput>div>div>input {
        border-radius: 6px;
        border: 1px solid #C8C8C8;
    }

    .stCheckbox label {
        color: #000000 !important;
        font-weight: 500;
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #146EB4;
        padding: 12px 0;
        text-align: center;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.1);
    }

    .footer a {
        font-size: 16px;
        color: #FFFFFF;
        text-decoration: none;
        font-weight: 600;
    }

    </style>

    <div class="amazon-title">⚡Customer Support Chatbot</div>
    <div class="amazon-subtitle">Your Trusted AI Assistant</div>
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

    client = OpenAI(api_key=OPENAI_API_KEY)
    query = "What's Deep Research?"
    response = client.responses.create(
    input= query,
    model="gpt-4o-mini",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ['vs_69318ea1dd0c8191bd8bf5f7131fa9dc'],
    }]
    )
    return response
    #return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

chain = init_chain()

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
            response = chain({
                "question": question,
                "chat_history": st.session_state.history,
            })
            response_text = response["answer"]

    st.session_state.history.append(
        (f"{question} {'(Web Search)' if search_web else '(RAG)'}", response_text)
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
