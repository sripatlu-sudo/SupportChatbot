import os
import pathlib, streamlit as st
from langchain_classic.vectorstores import FAISS
from langchain_classic.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# -----------------------------------
# PAGE SETUP — ELEGANT CORPORATE THEME
# -----------------------------------
st.set_page_config(page_title="Customer Support Chatbot", layout="wide")

st.markdown(
    """
    <style>

    /* --- Spectrum Brand Theme --- */

    body, .stApp {
        background: #F5F7FA;
        font-family: "Segoe UI", sans-serif;
        color: #1B1B1B;
    }

    /* Titles */
    .spectrum-title {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        margin-top: -20px;
        color: #0046BE;
        text-shadow: 0px 0px 6px rgba(0, 70, 190, 0.15);
    }

    .spectrum-subtitle {
        text-align: center;
        margin-bottom: 20px;
        font-size: 18px;
        color: #0094FF;
    }

    /* Chat bubbles */
    .chat-bubble-user {
        padding: 14px;
        background: #E8F0FE;
        border-left: 5px solid #0046BE;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #003A99;
        box-shadow: 0 0 5px rgba(0, 70, 190, 0.2);
    }

    .chat-bubble-bot {
        padding: 14px;
        background: #F0FBFF;
        border-left: 5px solid #0094FF;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #003A66;
        box-shadow: 0 0 6px rgba(0, 148, 255, 0.2);
    }

    .disclaimer-box {
        background: #F1F5F9;
        border-left: 4px solid #FFBB00;
        padding: 12px;
        border-radius: 6px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #0046BE !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        border: none !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }

    .stButton>button:hover {
        background-color: #003A99 !important;
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #ffffff;
        padding: 12px 0;
        text-align: center;
        box-shadow: 0 -4px 8px rgba(0,0,0,0.1);
    }

    .footer img {
        width: 110px;
        vertical-align: middle;
        margin-right: 8px;
    }

    .footer a {
        font-size: 16px;
        color: #0046BE;
        text-decoration: none;
        font-weight: 600;
    }

    </style>

    <div class="spectrum-title">📱 Customer Support Chatbot</div>
    <div class="spectrum-subtitle">Your Mobile • AI Support</div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# CALL SUPPORT WIDGET
# -------------------------
# Desktop fallback button (mobile users will use tel: link)
#if st.button("📞 Call Customer Support"):
#    st.markdown(
#        "<script>window.location.href='tel:8336778890'</script>",
#        unsafe_allow_html=True
#    )

# -------------------------
# DISCLAIMER WIDGET
# -------------------------
with st.expander("⚠️ Disclaimer"):
    st.markdown(
        """
        <div class="disclaimer-box">
        This chatbot uses AI to assist customers.  
        Responses may be incomplete or inaccurate.  
        Do not rely on this system for legal, medical, or financial decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------
# ORIGINAL LOGIC — UNCHANGED
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
# CHAT INPUT WITH WEB TOGGLE
# -----------------------------------
st.write("")  # spacing

input_col1, input_col2 = st.columns([0.82, 0.18])

with input_col1:
    question = st.chat_input("💬 How can we assist you?")

with input_col2:
    search_web = st.checkbox("🔍 Search web")

# Handle question
if question:
    with st.spinner("Processing..."):

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

# ------------------------------
# CHAT DISPLAY — CORPORATE STYLE
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




# ----------------------
# FOOTER — Spectrum Logo
# ----------------------
st.markdown(
    """
    <div class="footer">
        <div class="main-block call-box">
        📞 Need more help? Call: <b><a href="tel:(833) 224-6603">833-677-8890</a></b>
        <img src="" />
        <a href="https://www.ai-chatbot.com" target="_blank">ai-chatbot.com</a>
        </div>
     
    </div>
    """,
    unsafe_allow_html=True
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
# Other ideas....
# Prepare slide deck