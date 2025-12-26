import streamlit as st
import json
import os
from datetime import datetime
from langchain_aws import ChatBedrock
from langchain_classic.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import boto3
import tempfile

st.set_page_config(page_title="🎯 Promotions AI Agent", layout="wide")

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#ffffff;font-family:'Arial',sans-serif}
.agent-title{text-align:center;font-size:48px;font-weight:700;color:#ffffff;margin-bottom:20px;text-shadow:0 2px 4px rgba(0,0,0,0.3)}
.reasoning-box{background:rgba(255,255,255,0.1);padding:20px;border-radius:10px;margin:20px 0;border-left:4px solid #ffd700}
</style>
<div class="agent-title">🎯 Promotions AI Agent</div>
""", unsafe_allow_html=True)

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def load_promotions_repository():
    with open('promotions_repository.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def init_llm():
    config = load_config()
    boto3.setup_default_session(
        aws_access_key_id=config["aws_access_key_id"],
        aws_secret_access_key=config["aws_secret_access_key"],
        region_name=config["aws_region"]
    )
    return ChatBedrock(
        model_id=config["model_id"],
        region_name=config["aws_region"],
        model_kwargs={"temperature": 0.1}
    )

def extract_pdf_content(uploaded_file):
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load and extract PDF content
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        content = "\n".join([page.page_content for page in pages])
        return content
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

def analyze_promotions_file(file_content, repository):
    prompt = PromptTemplate(
        template="""You are an expert promotions strategist AI agent. Analyze the uploaded promotions data and create an optimal configuration.

THINKING PROCESS:
1. First, analyze the uploaded promotions data structure and content
2. Identify key patterns, categories, target audiences, and business objectives
3. Compare with available promotion templates in the repository
4. Consider seasonal factors, market trends, and customer behavior
5. Propose the most effective promotion configuration

UPLOADED PROMOTIONS DATA:
{promotions_data}

AVAILABLE PROMOTION REPOSITORY:
{repository}

INSTRUCTIONS:
- Analyze the data thoroughly using step-by-step reasoning
- Propose a specific promotion configuration in JSON format
- Provide detailed reasoning for each decision
- Consider ROI, customer engagement, and business impact
- Format your response as:
  1. ANALYSIS & REASONING (detailed thought process)
  2. PROPOSED CONFIGURATION (valid JSON)

Response:""",
        input_variables=["promotions_data", "repository"]
    )
    
    llm = init_llm()
    response = llm.invoke(prompt.format(
        promotions_data=file_content,
        repository=json.dumps(repository, indent=2)
    ))
    
    return response.content

def extract_json_from_response(response_text):
    try:
        # Find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
    except:
        pass
    return None

def main():
    repository = load_promotions_repository()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Promotions File")
        uploaded_file = st.file_uploader(
            "Upload your promotions document (PDF)", 
            type=['pdf']
        )
        
        if uploaded_file:
            file_content = extract_pdf_content(uploaded_file)
            st.success(f"✅ PDF processed: {uploaded_file.name}")
            
            with st.expander("📄 View Extracted Content"):
                st.text_area("PDF Content", file_content, height=200)
    
    with col2:
        st.subheader("🗂️ Promotions Repository")
        with st.expander("📚 View Available Templates"):
            st.json(repository)
    
    if uploaded_file and st.button("🤖 Analyze & Generate Configuration", use_container_width=True):
        with st.spinner("🧠 AI Agent is thinking and analyzing..."):
            analysis_result = analyze_promotions_file(file_content, repository)
        
        # Split reasoning and configuration
        parts = analysis_result.split("PROPOSED CONFIGURATION")
        
        if len(parts) >= 2:
            reasoning = parts[0].replace("ANALYSIS & REASONING", "").strip()
            config_text = parts[1].strip()
            
            st.subheader("🧠 AI Reasoning & Analysis")
            st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)
            
            st.subheader("⚙️ Proposed Configuration")
            
            # Extract and display JSON
            proposed_config = extract_json_from_response(config_text)
            if proposed_config:
                st.json(proposed_config)
                
                # Download button
                config_json = json.dumps(proposed_config, indent=2)
                st.download_button(
                    label="💾 Download Configuration",
                    data=config_json,
                    file_name=f"promotion_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.code(config_text, language='json')
        else:
            st.subheader("🤖 AI Analysis Result")
            st.write(analysis_result)
    
    # Sample data section
    st.subheader("📋 Sample Promotions Data")
    sample_data = {
        "current_promotions": [
            {"product_category": "electronics", "current_discount": 10, "sales_performance": "low"},
            {"product_category": "clothing", "current_discount": 15, "sales_performance": "medium"},
            {"product_category": "home", "current_discount": 5, "sales_performance": "high"}
        ],
        "business_goals": ["increase_revenue", "clear_inventory", "attract_new_customers"],
        "target_season": "holiday",
        "budget_constraints": {"max_discount_percentage": 30, "duration_limit_days": 21}
    }
    
    if st.button("📝 Use Sample Data"):
        st.json(sample_data)

if __name__ == "__main__":
    main()