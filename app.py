import streamlit as st
from langchain_classic.embeddings import SentenceTransformerEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.prompts import PromptTemplate
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import io


st.set_page_config(page_title="🤖 Customer Support Chatbot", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
.stApp{background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);color:#2c3e50;font-family:'Inter',sans-serif;padding-bottom:150px}
header{visibility:hidden}
.corp-title{text-align:center;font-size:36px;font-weight:600;color:#1a365d;margin-top:-30px;margin-bottom:8px;letter-spacing:-0.5px}
.corp-subtitle{text-align:center;font-size:16px;color:#4a5568;margin-bottom:30px;font-weight:400}

.css-1d391kg, .css-6qob1r, .css-18e3th9 {width: 300px !important;}
.css-1lcbmhc, .css-1rs6os, .css-17eq0hr {margin-left: 0px !important;}
.sidebar .sidebar-content {display: block !important;}
.footer-fixed{position:fixed;bottom:0;left:0;width:100%;z-index:999;background:#1a365d;color:white;padding:15px;text-align:center;border-radius:0;margin:0;box-shadow:0 -2px 10px rgba(0,0,0,0.2)}
</style>
<div class="corp-title">🤖 Customer Support Chatbot</div>
<div class="corp-subtitle">Enterprise Solutions | Powered by AI</div>
""", unsafe_allow_html=True)

@st.cache_resource
def init_chain():
    embeddings = SentenceTransformerEmbeddings(model_name="thenlper/gte-small")
    vectordb = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    prompt = PromptTemplate(
        template="""You are Spectrum mobile customer support assistant. 
        - Answer questions ONLY based on the provided context, and relavent to Sepctrum mobile products and services. 
        - Provide detailed responses, in bullet point format when possible, and be courteous. 
        - If the answer is not in the context, say "I don't have that information. Please contact  (833) 224-6603 for further assistance.

    Context: {context}

    Question: {question}

    Answer:""",
        input_variables=["context", "question"]
    )
    
    return ConversationalRetrievalChain.from_llm(
        Ollama(model="gemma3:4b", temperature=0.1),
        vectordb.as_retriever(search_kwargs={"k": 8}),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "selected_msg" not in st.session_state:
    st.session_state.selected_msg = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None


# Initialize chain
chain = init_chain()

with st.expander("⚠️ Disclaimer"):
    st.warning("This chatbot is an AI assistant built for demonstration. Responses may be inaccurate. Do not use for legal, financial or sensitive decisions.")

if question := st.chat_input("💬 How may we assist you today?"):
    if question != st.session_state.last_question:
        st.session_state.last_question = question
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.spinner("Processing your request..."):
            response = chain({"question": question, "chat_history": [(m["content"], "") for m in st.session_state.messages if m["role"] == "user"]})
        
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        if len(st.session_state.messages) > 20:
            st.session_state.messages = st.session_state.messages[-20:]
        st.rerun()


def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Customer Support Chat History", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Chat messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            story.append(Paragraph(f"<b>Question {i//2+1}:</b>", styles['Heading3']))
            story.append(Paragraph(msg["content"], styles['Normal']))
            story.append(Spacer(1, 10))
        else:
            story.append(Paragraph("<b>Answer:</b>", styles['Heading3']))
            story.append(Paragraph(msg["content"], styles['Normal']))
            story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Sidebar for chat history
with st.sidebar:
    st.header("Chat History")
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.selected_msg = None
        st.rerun()
    
    if len(st.session_state.messages) == 0:
        st.write("No conversations yet")
    else:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                if st.button(f"Q{i//2+1}: {msg['content'][:50]}...", key=f"hist_{i}"):
                    st.session_state.selected_msg = i
                    st.rerun()

# Main chat area
if st.session_state.selected_msg is not None:
    # Show selected message and its response
    selected_idx = st.session_state.selected_msg
    with st.chat_message("user"):
        st.write(st.session_state.messages[selected_idx]["content"])
    if selected_idx + 1 < len(st.session_state.messages):
        with st.chat_message("assistant"):
            st.write(st.session_state.messages[selected_idx + 1]["content"])
    if st.button("Back to full chat"):
        st.session_state.selected_msg = None
        st.rerun()
else:
    # Show full chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Add like/dislike buttons for assistant messages
            if msg["role"] == "assistant":
                col1, col2, col3 = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"like_{i}"):
                        st.session_state.feedback[i] = "like"
                with col2:
                    if st.button("👎", key=f"dislike_{i}"):
                        st.session_state.feedback[i] = "dislike"


# Download PDF button in main area
if st.session_state.messages:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📄 Download Chat as PDF", use_container_width=True):
            pdf_buffer = generate_pdf()
            st.download_button(
                label="💾 Click to Save PDF",
                data=pdf_buffer,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# Footer
st.markdown("<div class='footer-fixed'><strong>Need further assistance?</strong> Call us at <strong>(833) 224-6603</strong></div>", unsafe_allow_html=True)