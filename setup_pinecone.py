import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def setup_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME") # banao-index
    
    if not api_key or not index_name:
        print("Missing credentials in .env")
        return

    pc = Pinecone(api_key=api_key)
    
    # Delete if exists
    active_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name in active_indexes:
        print(f"Deleting existing index: {index_name}")
        pc.delete_index(index_name)
        time.sleep(5)
    
    # Create with 1536 dimensions
    print(f"Creating new index: {index_name} with 1536 dimensions")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print("Wait for initialization...")
    time.sleep(10)
    print("Pinecone setup complete.")

if __name__ == "__main__":
    setup_index()
