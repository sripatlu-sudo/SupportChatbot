import os
import json
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def get_all_links(base_url):
    """Crawl the base URL and extract all relevant support links"""
    try:
        response = requests.get(base_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = set()
        links.add(base_url)  # Include the main page
        
        # Find all links on the page
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Only include Spectrum support mobile links
            if 'spectrum.net/support' in full_url and 'mobile' in full_url:
                links.add(full_url)
                print(f"Link: {full_url}")
        
        return list(links)
    except Exception as e:
        print(f"Error crawling {base_url}: {e}")
        return [base_url]

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
    
    # Get all URLs to crawl
    base_url = "https://www.spectrum.net/support/mobile/transferring-your-spectrum-mobile-service-another-iphone"
    urls = get_all_links(base_url)
    
    print(f"Found {len(urls)} URLs to process")
    
    # Load web documents
    loader = WebBaseLoader(urls)
    raw_docs = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(raw_docs)
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to local directory
    vectorstore.save_local("faiss_webindex")
    
    print(f"Created FAISS web index with {len(chunks)} chunks from {len(raw_docs)} documents")

if __name__ == "__main__":
    main()