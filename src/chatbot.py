"""Enhanced chatbot with security and response formatting."""

from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.config import Config
from src.access_control import AccessControl
import re
import json

# Define enhanced prompt with better formatting instructions
COMPANY_KB_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template='''
You are a helpful assistant for INTERNAL COMPANY USE ONLY.

IMPORTANT RULES:
1. ONLY answer using the context provided below
2. If the answer isn't in the context, say: "I don't have enough information to answer that question. Please contact the IT department for assistance."
3. Keep answers concise but complete
4. Format responses properly with headings, bullet points, or lists when appropriate
5. Never make up information
6. Never reveal confidential information to unauthorized users
7. Add a disclaimer at the end: "This information is for internal company use only"

Context:
{context}

Question:
{question}

Answer (in properly formatted markdown):
'''
)

class CompanyKnowledgeBot:
    def __init__(self, user_context=None):
        """Initialize the chatbot with user context."""
        self.user_context = user_context or {"clearance_level": "standard"}
        self.access_control = AccessControl(user_context)
        
        # Load the LLM
        self.llm = LlamaCpp(
            model_path=Config.LLM_MODEL_PATH,
            n_ctx=4096,
            n_batch=64,
            temperature=Config.TEMPERATURE,
            verbose=False
        )

        # Embedding and vector DB setup
        self.embed_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vectordb = Chroma(persist_directory=Config.VECTOR_DB_PATH, embedding_function=self.embed_model)
        
        # Use retriever with filtered results
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
        
        # Initialize QA chain
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,  # Enable source tracking
            chain_type_kwargs={"prompt": COMPANY_KB_PROMPT}
        )
    
    def answer(self, query):
        """Process the query with security checks and formatting."""
        # Security check for restricted queries
        allowed, restricted_topics = self.access_control.check_query_permission(query)
        if not allowed:
            return (f"I'm sorry, but I can't provide information on {', '.join(restricted_topics)}. "
                   "This appears to be a restricted topic. Please contact your supervisor or the "
                   "security department if you need access to this information.")
        
        # FIXED: Create custom retriever function that applies filtering
        # instead of trying to modify the existing retriever
        def get_filtered_documents(query):
            # Get documents using the original retriever
            docs = self.retriever.get_relevant_documents(query)
            # Filter docs based on user's clearance level
            filtered_docs = [doc for doc in docs if self.access_control.filter_document_by_metadata(doc)]
            return filtered_docs
        
        try:
            # Run query with custom document retrieval
            # First get filtered documents
            relevant_docs = get_filtered_documents(query)
            
            if not relevant_docs:
                return "I couldn't find any relevant information in our knowledge base that matches your question and your access level. Please try rephrasing your question or contact support for assistance."
            
            # Create a context string from the filtered documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Use the LLM with our prompt template
            prompt = COMPANY_KB_PROMPT.format(context=context, question=query)
            response = self.llm.invoke(prompt)
            
            # Log the successful query
            sources = [getattr(doc, "metadata", {}).get("source", "unknown") for doc in relevant_docs]
            self.access_control.log_access("knowledge_query", query, "read", True, details={"sources": sources})
            
            # Apply post-processing to ensure confidentiality
            answer = self.post_process_response(response)
            
            return answer
            
        except Exception as e:
            # Log the error
            self.access_control.log_access("knowledge_query", query, "read", False, details={"error": str(e)})
            return f"I encountered an error while processing your query. Please try again or contact support. Error: {str(e)}"
    
    def post_process_response(self, response):
        """Apply post-processing to ensure confidentiality and proper formatting."""
        # Ensure confidentiality disclaimer is present
        if "This information is for internal company use only" not in response:
            response += "\n\n*This information is for internal company use only*"
        
        # Additional checks based on confidentiality level
        if Config.CONFIDENTIALITY_LEVEL == "strict":
            # Check for any patterns that might indicate leaked confidential info
            patterns = [
                r'\b(?:password|pw|passwd)\s*[:=]\s*\S+',  # Passwords
                r'\b(?:api[\s-]*key|token|secret)\s*[:=]\s*\S+',  # API keys
                r'\b(?:\d{3}-\d{2}-\d{4})',  # SSN
                r'\b(?:confidential|classified)\b'  # Explicit confidentiality markers
            ]
            
            for pattern in patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    # Redact potentially sensitive information
                    response = re.sub(pattern, "[REDACTED]", response, flags=re.IGNORECASE)
        
        return response

# For backward compatibility
def chatbot_answer(query, user_context=None):
    """Legacy function for compatibility with existing code."""
    bot = CompanyKnowledgeBot(user_context)
    return bot.answer(query)