"""Enhanced document loader with metadata extraction."""

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import Config
import re
import os

def extract_metadata_from_content(content, filepath):
    """Extract classification and other metadata from document content."""
    metadata = {
        "source": filepath,
        "filename": os.path.basename(filepath),
    }
    
    # Default classification is public
    classification = "public"
    
    # Check for classification markers in the content
    if re.search(r'confidential|sensitive|private|internal[\s\-]only', content, re.IGNORECASE):
        classification = "confidential"
    elif re.search(r'internal[\s\-]only', content, re.IGNORECASE):
        classification = "internal-only"
        
    metadata["classification"] = classification
    
    # Extract document title (first heading)
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1)
    
    return metadata

def load_documents(directory=Config.DATA_DIRECTORY):
    """Load documents with enhanced metadata extraction."""
    from pathlib import Path
    docs = []
    
    for path in Path(directory).rglob("*.md"):
        loader = UnstructuredMarkdownLoader(str(path))
        loaded_docs = loader.load()
        
        # For each document, extract and enhance metadata
        for doc in loaded_docs:
            enhanced_metadata = extract_metadata_from_content(doc.page_content, str(path))
            doc.metadata.update(enhanced_metadata)
            docs.append(doc)
            
    print(f"[Data Loader] Loaded {len(docs)} documents")
    return docs

def split_documents(docs):
    """Split documents into chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, 
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"[Data Loader] Split into {len(chunks)} chunks")
    return chunks