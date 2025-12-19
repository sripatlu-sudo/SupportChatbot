import os
import json
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.document_loaders import DirectoryLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader
import os, pathlib, textwrap, glob

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Set AWS credentials
    boto3.setup_default_session(
        aws_access_key_id=config["aws_access_key_id"],
        aws_secret_access_key=config["aws_secret_access_key"],
        region_name=config["aws_region"]
    )
    
    # Initialize embeddings
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name=config["aws_region"]
    )
    
    pdf_paths = glob.glob("data/phone*.pdf")

    raw_docs = []
    for path in pdf_paths:
        raw_docs.extend(PyPDFLoader(path).load())


    # Load documents from ./data folder
    #loader = DirectoryLoader("./data", glob="**/*phone*.pdf")
    #documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(raw_docs)
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to local directory
    vectorstore.save_local("faiss_index")
    
    print(f"Created FAISS index with {len(chunks)} chunks from {len(raw_docs)} documents")

if __name__ == "__main__":
    main()