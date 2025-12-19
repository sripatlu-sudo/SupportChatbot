import streamlit as st
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.vectorstores import FAISS
from langchain_classic.prompts import PromptTemplate
import boto3
import json
import os
from datetime import datetime

st.set_page_config(page_title="💬 Chat", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp{background:#ffffff;max-width:400px;margin:0 auto;border-radius:15px;box-shadow:0 10px 30px rgba(0,0,0,0.2);font-family:'Arial',sans-serif}
header{visibility:hidden}
.main .block-container{padding:1rem;max-width:400px}
.chat-header{background:linear-gradient(45deg,#1976d2,#7b1fa2);color:white;padding:15px;border-radius:15px 15px 0 0;text-align:center;font-weight:600;margin:-1rem -1rem 1rem -1rem}
.chat-container{height:300px;overflow-y:auto;border:1px solid #e0e0e0;border-radius:10px;padding:10px;margin-bottom:10px}
.welcome-badge{display:inline-block;background:linear-gradient(45deg,#4caf50,#81c784);color:#ffffff;padding:6px 16px;border-radius:20px;font-size:14px;font-weight:500;margin:10px auto;box-shadow:0 3px 10px rgba(76,175,80,0.3)}
stChatInput>div{border-radius:25px}
.stChatInput input::placeholder{color:#ff9800!important}
.stSpinner>div{display:flex;justify-content:center;align-items:center;height:100px}
</style>
<div class="chat-header">💬 Spectrum Support Chatbot</div>
<div style="text-align:center"><span class="welcome-badge">🌟 Always Happy to Assist</span></div>
""", unsafe_allow_html=True)

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def init_chain():
    config = load_config()
    boto3.setup_default_session(
        aws_access_key_id=config["aws_access_key_id"],
        aws_secret_access_key=config["aws_secret_access_key"],
        region_name=config["aws_region"]
    )
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name=config["aws_region"])
    vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    prompt = PromptTemplate(
        template="""You are Spectrum mobile customer support assistant. 
        - ALWAYS respond in English language only, regardless of the language used in the question.
        - Answer questions ONLY based on the provided context, and relevant to Spectrum mobile products and services. 
        - Provide detailed responses, in bullet point format when possible, and be courteous. 
        - If the answer is not in the context, say "I don't have that information. Please contact (833) 224-6603 for further assistance."
        - Never respond in any language other than English.

    Context: {context}

    Question: {question}

    Answer in English:""",
        input_variables=["context", "question"]
    )
    
    return ConversationalRetrievalChain.from_llm(
        ChatBedrock(model_id=config["model_id"], region_name=config["aws_region"]),
        vectordb.as_retriever(search_kwargs={"k": 3}),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        combine_docs_chain_kwargs={"prompt": prompt}
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

def save_feedback(question, response, feedback):
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "response": response,
        "feedback": feedback
    }
    
    feedback_file = "feedback.json"
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    data.append(feedback_data)
    
    with open(feedback_file, 'w') as f:
        json.dump(data, f, indent=2)

chain = init_chain()

# Chat container
with st.container():
    for i, msg in enumerate(st.session_state.messages[-6:]):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            if msg["role"] == "assistant" and i > 0:
                col1, col2, col3 = st.columns([2, 2, 8])
                with col1:
                    if st.button("👍", key=f"like_{i}"):
                        question = st.session_state.messages[i-1]["content"]
                        save_feedback(question, msg["content"], "like")
                        st.session_state.feedback[i] = "like"
                with col2:
                    if st.button("👎", key=f"dislike_{i}"):
                        question = st.session_state.messages[i-1]["content"]
                        save_feedback(question, msg["content"], "dislike")
                        st.session_state.feedback[i] = "dislike"

if question := st.chat_input("💬 Hello, how can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": question})
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.spinner("Thinking..."):
            response = chain({"question": question, "chat_history": []})
    
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    st.rerun()