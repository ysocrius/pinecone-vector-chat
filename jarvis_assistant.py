import os
import time
import uuid
import threading
from typing import List, Optional
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from werkzeug.utils import secure_filename

# LangChain & Vector Store
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Local Metrics (HuggingFace)
from sentence_transformers import SentenceTransformer, util

# Internal modules
from ingest import ingest_documents

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'docs'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pydantic Models for Validation
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)

class IngestRequest(BaseModel):
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)

# Global State for Local Metrics
# Using a local model for similarity scoring as requested (like previous repos)
try:
    print("⏳ Loading local similarity model (all-MiniLM-L6-v2)...")
    local_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Local model loaded.")
except Exception as e:
    print(f"⚠️ Could not load local model: {e}")
    local_model = None

# Rate Limiting State (Simplified)
rate_limit_store = {}

def rate_limit_check():
    """Simple rate limiting: 10 requests per minute per IP"""
    ip = request.remote_addr
    now = time.time()
    if ip not in rate_limit_store:
        rate_limit_store[ip] = []
    
    # Filter for last 60 seconds
    rate_limit_store[ip] = [t for t in rate_limit_store[ip] if now - t < 60]
    
    if len(rate_limit_store[ip]) >= 10:
        return False
    
    rate_limit_store[ip].append(now)
    return True

def get_rag_chain():
    """Initialize Pinecone RAG Chain with enhanced error logging & auto-creation"""
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not all([openai_key, pinecone_key, index_name]):
            print(f"❌ Missing Config: OpenAI: {'Set' if openai_key else 'Missing'}, Pinecone: {'Set' if pinecone_key else 'Missing'}, Index: {index_name}")
            return None, None

        embeddings = OpenAIEmbeddings()
        
        # Check if index exists and is accessible
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=pinecone_key)
        
        active_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name in active_indexes:
            # Check existing index dimension
            desc = pc.describe_index(index_name)
            if desc.dimension != 1536:
                print(f"⚠️ Warning: Index '{index_name}' has dimension {desc.dimension}, but OpenAI requires 1536.")
                print(f"📡 Attempting to delete and recreate index with correct dimensions...")
                pc.delete_index(index_name)
                time.sleep(5)
                active_indexes.remove(index_name)

        if index_name not in active_indexes:
            print(f"📡 Creating index '{index_name}' (Dimension: 1536)...")
            try:
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                print(f"✅ Index created. Waiting for DNS/Initialization (30s)...")
                time.sleep(30) # New indexes take time to propagate
            except Exception as creation_err:
                print(f"❌ Failed to create index: {creation_err}")
                return None, None

        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Verify if index has data
        test_docs = retriever.invoke("test")
        if not test_docs:
            print(f"⚠️ Warning: Index '{index_name}' is currently empty. Please ingest documents via UI.")
        
        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        
        If the answer is not in the context, say "I don't have enough information to answer that based on my current knowledge base."
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        return chain, retriever
    except Exception as e:
        import traceback
        print(f"❌ RAG Initialization Error: {str(e)}")
        print(traceback.format_exc())
        return None, None

def calculate_local_similarity(query: str, context: str):
    """Calculate similarity score using local HuggingFace model"""
    if not local_model or not context:
        return 0.0
    
    query_emb = local_model.encode(query, convert_to_tensor=True)
    context_emb = local_model.encode(context, convert_to_tensor=True)
    score = util.pytorch_cos_sim(query_emb, context_emb)
    return float(score[0][0])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "running",
        "pinecone_index": os.getenv("PINECONE_INDEX_NAME"),
        "openai_model": "gpt-4o-mini",
        "local_metrics": "active" if local_model else "inactive"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    if not rate_limit_check():
        return jsonify({"error": "Rate limit exceeded. Try again in a minute."}), 429
        
    start_time = time.time()
    try:
        data = request.json
        req = ChatRequest(**data)
        
        chain, retriever = get_rag_chain()
        if not chain:
            return jsonify({"error": "System not ready. Please ingest documents first."}), 503

        # Get relevant docs for metrics calculation
        docs = retriever.invoke(req.message)
        context_text = "\n".join([d.page_content for d in docs])
        
        # Calculate localized similarity score
        similarity_score = calculate_local_similarity(req.message, context_text)
        
        # Generate Answer
        response = chain.invoke(req.message)
        
        latency = time.time() - start_time
        
        return jsonify({
            "status": "success",
            "message": response,
            "sources": list(set([d.metadata.get('source', 'unknown') for d in docs])),
            "metrics": {
                "latency_seconds": round(latency, 2),
                "top_similarity_score": round(similarity_score, 4)
            }
        })
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle UI-based multi-file uploads and trigger ingestion"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    saved_paths = []
    filenames = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            saved_paths.append(save_path)
            filenames.append(filename)
    
    if not saved_paths:
        return jsonify({"error": "No valid file types supported"}), 400
        
    chunk_size = int(request.form.get('chunk_size', 1000))
    chunk_overlap = int(request.form.get('chunk_overlap', 200))
    
    # Trigger background ingestion for ALL uploaded files
    thread = threading.Thread(target=ingest_documents, kwargs={
        "file_path": saved_paths,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    })
    thread.start()
    
    return jsonify({
        "status": "success",
        "message": f"{len(filenames)} files ({', '.join(filenames)}) uploaded. Ingestion started."
    })

@app.route('/api/ingest-path', methods=['POST'])
def ingest_by_path():
    """Handle ingestion via specified local file paths (can be a string or list)"""
    data = request.json
    file_paths = data.get('file_path')
    
    if not file_paths:
        return jsonify({"error": "No file paths provided"}), 400
    
    # Normalize to list
    if isinstance(file_paths, str):
        paths_to_ingest = [p.strip() for p in file_paths.split(',') if p.strip()]
    elif isinstance(file_paths, list):
        paths_to_ingest = [p.strip() for p in file_paths if isinstance(p, str) and p.strip()]
    else:
        return jsonify({"error": "Invalid format for file_path. Expected string or list."}), 400

    if not paths_to_ingest:
        return jsonify({"error": "No valid file paths found"}), 400

    # Validate existence
    invalid_paths = [p for p in paths_to_ingest if not os.path.exists(p)]
    if invalid_paths:
        return jsonify({"error": f"Some paths do not exist: {', '.join(invalid_paths)}"}), 404

    chunk_size = int(data.get('chunk_size', 1000))
    chunk_overlap = int(data.get('chunk_overlap', 200))
    clear_existing = data.get('clear_existing', False)
    
    # Run ingestion in the background
    thread = threading.Thread(target=ingest_documents, kwargs={
        "file_path": paths_to_ingest,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "clear_existing": clear_existing
    })
    thread.start()
    
    return jsonify({
        "status": "success",
        "message": f"Ingestion started for {len(paths_to_ingest)} path(s)."
    })

@app.route('/api/example-questions', methods=['GET'])
def example_questions():
    return jsonify({
        "questions": [
            "What is this document about?",
            "Can you summarize the key findings?",
            "What are the specific requirements mentioned?"
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)