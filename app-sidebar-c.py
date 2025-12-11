import streamlit as st
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI

st.set_page_config(
    page_title="College Buddy Chatbot",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🏛️ College Buddy Chatbot")
st.caption("Your college AI assistant")

# Disclaimer
with st.expander("⚠️ Disclaimer"):
    st.warning("""
    This AI chatbot provides automated information.  
    It may generate incomplete or inaccurate responses.  
    Please verify important details independently.
    """)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_data(ttl=3600)
def chat_response(query):
    try:
        client = get_openai_client()
        formatted_query = COLLEGE_ADVISOR_PROMPT.format(query=query)
        
        response = client.responses.create(
            input=formatted_query,
            model="gpt-4o-mini",
            tools=[{
                "type": "file_search",
                "vector_store_ids": ['vs_69347f971e348191b597c0bb6b20de9e'],
            }]
        )
        return response.output[1].content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_data(ttl=1800)
def web_search_response(query):
    try:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=OPENAI_API_KEY,
        )
        return llm.invoke(query).content
    except Exception as e:
        return f"Error: {str(e)}"

def upload_to_vector_store(file_content, filename):
    try:
        client = get_openai_client()
        
        file_obj = client.files.create(
            file=file_content,
            purpose="assistants"
        )
        
        client.vector_stores.files.create(
            vector_store_id="vs_69347f971e348191b597c0bb6b20de9e",
            file_id=file_obj.id
        )
        
        return f"Successfully uploaded {filename} to knowledge base!"
    except Exception as e:
        return f"Error uploading file: {str(e)}"

# Sidebar
with st.sidebar:
    st.subheader("📄 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Upload college documents",
        type=['pdf', 'txt', 'docx'],
        help="Upload college guides, requirements, or other helpful documents"
    )
    
    if uploaded_file and st.button("📤 Add to Knowledge Base", type="primary"):
        with st.spinner("Uploading to knowledge base..."):
            result = upload_to_vector_store(uploaded_file, uploaded_file.name)
        st.success(result) if "Successfully" in result else st.error(result)
    
    st.divider()
    st.subheader("📚 Chat History")
    
    if not st.session_state.history:
        st.caption("No conversations yet.")
    else:
        with st.container(height=300):
            for idx, (q, a) in enumerate(st.session_state.history[-10:]):
                if st.button(f"💬 {q[:36]}...", key=f"hist_{idx}", use_container_width=True):
                    st.info(f"**Past Query:** {q}\n\n**Response:** {a}")
    
    st.divider()
    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Chat controls
search_web = st.toggle("🌐 Web Search")
question = st.chat_input("💬 How can I help you?")

# Process question
if question and not st.session_state.processing:
    st.session_state.processing = True
    
    with st.spinner("🎓 Thinking..."):
        if search_web:
            response_text = web_search_response(question)
        else:
            response_text = chat_response(question)
    
    query_label = f"{question} {'(Web Search)' if search_web else '(RAG)'}"
    st.session_state.history.append((query_label, response_text))
    
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]
    
    st.session_state.processing = False
    st.rerun()

# Display chat
if st.session_state.history:
    for user, bot in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(user)
        with st.chat_message("assistant"):
            st.write(bot)

# Footer
st.divider()
st.info("📞 Need more help? Call: **1-800-COLLEGE**")