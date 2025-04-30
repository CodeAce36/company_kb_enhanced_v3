"""Enhanced database builder with logging and error handling."""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.data_loader import load_documents, split_documents
from src.config import Config
import logging
import time
import os

def build_database():
    """Build the vector database with progress tracking and error handling."""
    # Set up logging
    logging.basicConfig(
        filename='logs/build_db.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    start_time = time.time()
    
    try:
        logging.info("[Build DB] Starting document loading and embedding...")
        print("[Build DB] Starting document loading and embedding...")
        
        # Ensure data directory exists
        if not os.path.exists(Config.DATA_DIRECTORY):
            msg = f"[Build DB] Error: Data directory '{Config.DATA_DIRECTORY}' not found"
            logging.error(msg)
            print(msg)
            return False
        
        # Load documents with metadata
        docs = load_documents(Config.DATA_DIRECTORY)
        logging.info(f"[Build DB] Loaded {len(docs)} documents")
        
        # Split into chunks
        chunks = split_documents(docs)
        logging.info(f"[Build DB] Split into {len(chunks)} chunks")
        print(f"[Build DB] Total chunks: {len(chunks)}")
        
        # Create embeddings model
        embed_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        
        # Create and persist the database
        db = Chroma.from_documents(
            chunks, 
            embedding=embed_model, 
            persist_directory=Config.VECTOR_DB_PATH
        )
        db.persist()
        
        elapsed_time = time.time() - start_time
        logging.info(f"[Build DB] Done. Vector DB saved to ./{Config.VECTOR_DB_PATH} in {elapsed_time:.2f} seconds")
        print(f"[Build DB] Done. Vector DB saved to ./{Config.VECTOR_DB_PATH} in {elapsed_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        logging.error(f"[Build DB] Error building database: {str(e)}")
        print(f"[Build DB] Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Ensure logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
    build_database()