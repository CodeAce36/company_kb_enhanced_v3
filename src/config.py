"""Configuration settings for the knowledge base system."""

class Config:
    # Model settings
    LLM_MODEL_PATH = "models/Mistral-7B-Instruct-v0.1.Q4_K_M.gguf"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TOKENS = 8192
    
    # Vector DB settings
    VECTOR_DB_PATH = "db"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    RETRIEVAL_K = 5
    MIN_RELEVANCE_SCORE = 0.7
    
    # Security settings
    CONFIDENTIALITY_LEVEL = "strict"  # 'strict', 'moderate', or 'low'
    LOGGING_ENABLED = True
    DATA_DIRECTORY = "data"
    
    # Prompt settings
    TEMPERATURE = 0.1  # Lower temperature for more factual responses