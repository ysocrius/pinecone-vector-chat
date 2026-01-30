import os
import sys
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def check_environment():
    """Check required environment variables for Pinecone"""
    required = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    for var in required:
        if not os.getenv(var):
            print(f"Error: {var} not found in .env file!")
            return False
    return True

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def ingest_documents(docs_folder="docs", file_path=None, chunk_size=1000, chunk_overlap=200, clear_existing=False):
    """
    Ingest PDF and TXT documents into Pinecone.
    If file_path is provided (string or list), only those files are processed.
    Otherwise, scans the docs_folder.
    """
    start_time = time.time()
    
    if not check_environment():
        return False
    
    embeddings = OpenAIEmbeddings()
    
    # Optional: Clear existing vectors for a fresh sync
    if clear_existing:
        try:
            index_name = os.getenv("PINECONE_INDEX_NAME")
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(index_name)
            print(f"🧹 Clearing existing vectors from index: {index_name}...")
            index.delete(delete_all=True)
            time.sleep(5) # Give Pinecone more time to propagate deletion
        except Exception as e:
            print(f"⚠️ Warning: Could not clear index: {e}")

    files_to_process = []
    if file_path:
        # Normalize file_path to a list of absolute paths
        paths_to_check = file_path if isinstance(file_path, list) else [file_path]
        for path in paths_to_check:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                files_to_process.append(abs_path)
            else:
                print(f"Error: File {abs_path} not found.")
    else:
        if not os.path.exists(docs_folder):
            os.makedirs(docs_folder)
            return False
        for f in os.listdir(docs_folder):
            if f.endswith(('.pdf', '.txt')):
                files_to_process.append(os.path.abspath(os.path.join(docs_folder, f)))

    if not files_to_process:
        print(f"No files to process.")
        return False
    
    print(f"🚀 Processing {len(files_to_process)} file(s) (Size: {chunk_size}, Overlap: {chunk_overlap})...")
    
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    for current_file_path in files_to_process:
        file_name = os.path.basename(current_file_path)
        
        if file_name.endswith('.pdf'):
            text = extract_text_from_pdf(current_file_path)
            if text:
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": file_name, "type": "pdf"}
                    ))
        elif file_name.endswith('.txt'):
            try:
                # TextLoader uses the file path for its own metadata, we override it
                loader = TextLoader(current_file_path, encoding='utf-8')
                docs = loader.load()
                # Explicitly normalize metadata to just the filename
                for d in docs:
                    d.metadata = {"source": file_name, "type": "txt"}
                
                split_docs = text_splitter.split_documents(docs)
                documents.extend(split_docs)
            except Exception as e:
                print(f"    ✗ Failed to process {file_name}: {e}")
    
    if not documents:
        return False
    
    print(f"🧠 Syncing {len(documents)} chunks to Pinecone index: {os.getenv('PINECONE_INDEX_NAME')}...")
    
    try:
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=os.getenv("PINECONE_INDEX_NAME")
        )
        latency = time.time() - start_time
        print(f"✅ Ingestion complete in {latency:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Pinecone Error: {e}")
        return False

if __name__ == "__main__":
    ingest_documents()