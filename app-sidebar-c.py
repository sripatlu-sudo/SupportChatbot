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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

body, .stApp {
    background: #262730;
    font-family: "Crimson Text", "Times New Roman", serif;
    color: #F8F6F0;
}

.college-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    margin-top: -10px;
    color: #FFD700;
    font-family: "Playfair Display", serif;
    letter-spacing: 2px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
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

.stCheckbox label {
    color: #FFFFFF !important;
    font-weight: 600;
}

.stCheckbox > label > div {
    color: #FFFFFF !important;
}

.stCheckbox span {
    color: #FFFFFF !important;
}

[data-testid="stCheckbox"] label {
    color: #FFFFFF !important;
}

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

.sidebar-title {
    font-size: 24px;
    font-weight: 700;
    color: rgba(25, 42, 86, 0.9);
    font-family: "Playfair Display", serif;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}

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

[data-testid="stAlert"] {
    background-color: lightblue !important;
}
</style>

<div class="college-title">🏛️ College Buddy Chatbot</div>
<div class="college-subtitle">Just relax....your AI assistant is here to help</div>
""", unsafe_allow_html=True)

with st.expander("⚠️ Disclaimer"):
    st.markdown("""
    <div class="disclaimer-box">
    This AI chatbot provides automated information.  
    It may generate incomplete or inaccurate responses.  
    Please verify important details independently.
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "selected_history" not in st.session_state:
    st.session_state.selected_history = None

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
- Remove any 

User Question: {query}

Please provide a comprehensive response based on the available information.
"""

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def chat_response(query):
    try:
        client = get_openai_client()
        formatted_query = COLLEGE_ADVISOR_PROMPT.format(query=query)
        
        response = client.responses.create(
            input=formatted_query,
            model="gpt-4o",
            tools=[{
                "type": "file_search",
                "vector_store_ids": ['vs_69347f971e348191b597c0bb6b20de9e'],
            }]
        )
        return response.output[1].content[0].text
    except Exception as e:
        return "I'm having trouble accessing the information you requested. Please try rephrasing your question! 😊"

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def web_search_response(query):
    try:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=OPENAI_API_KEY,
        )
        return llm.invoke(query).content
    except Exception as e:
        return "I'm having some difficulty with web search at the moment. Please try asking your question without web search, or try again later! 🌐"

def upload_to_vector_store(file_content, filename):
    try:
        client = get_openai_client()
        
        # Create file
        file_obj = client.files.create(
            file=file_content,
            purpose="assistants"
        )
        
        # Add to vector store
        client.beta.vector_stores.files.create(
            vector_store_id="vs_69347f971e348191b597c0bb6b20de9e",
            file_id=file_obj.id
        )
        
        return f"Successfully uploaded {filename} to knowledge base!"
    except Exception as e:
        return "Sorry, I couldn't upload your file right now. Please try again or contact support! 📄"

# Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📄 Upload Documents</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload college documents",
        type=['pdf', 'txt', 'docx'],
        help="Upload college guides, requirements, or other helpful documents"
    )
    
    if uploaded_file and st.button("📤 Add to Knowledge Base"):
        result = upload_to_vector_store(uploaded_file, uploaded_file.name)
        st.success(result) if "Successfully" in result else st.error(result)
    
    st.markdown("---")
    st.markdown("<div class='sidebar-title'>📚 Chat History</div>", unsafe_allow_html=True)
    
    if len(st.session_state.history) == 0:
        st.caption("No conversations yet.")
    else:
        for idx, (q, a) in enumerate(st.session_state.history[-10:]):  # Show last 10
            if st.button(f"💬 {q[:36]}...", key=f"hist_{idx}"):
                st.session_state.selected_history = (q, a)
    
    st.markdown("---")
    if st.button("🗑 Clear History"):
        st.session_state.history = []

# Chat input
col1, col2 = st.columns([0.82, 0.18])

with col1:
    question = st.chat_input("💬 How can I help you?")

with col2:
    search_web = st.checkbox("🌐 Web Search")

# Process question
if question and not st.session_state.processing:
    st.session_state.processing = True
    
    with st.spinner("🎓 Thinking..."):
        if search_web:
            response_text = web_search_response(question)
        else:
            response_text = chat_response(question)
    
    # Add to history
    query_label = f"{question} {'(Web Search)' if search_web else '(RAG)'}"
    st.session_state.history.append((query_label, response_text))
    
    # Limit history size
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]
    
    st.session_state.processing = False

# Display selected history details
if st.session_state.selected_history:
    q, a = st.session_state.selected_history
    st.info(f"**Past Query:** {q}\n\n**Response:** {a}")
    if st.button("❌ Close Details"):
        st.session_state.selected_history = None

# Display chat
for user, bot in reversed(st.session_state.history):
    st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {user}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble-bot'><b>AI:</b> {bot}</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div class="main-block call-box">
    📞 Need more help? Call: <b><a href="tel:(800) 265-5343">1-800-COLLEGE</a></b>
    </div>
</div>
""", unsafe_allow_html=True)