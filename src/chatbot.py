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

# Define enhanced prompt with better handling of irrelevant contexts
COMPANY_KB_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template='''
You are a helpful assistant for INTERNAL COMPANY USE ONLY.

IMPORTANT RULES:
1. ONLY answer using the context provided below
2. If the answer isn't in the context OR the context seems irrelevant to the question, say: "I don't have enough information to answer that question. This topic may not be covered in our company knowledge base. Please contact the IT department for assistance."
3. Keep answers concise but complete
4. Ensure responses are NEVER truncated - always complete your thoughts
5. Format responses properly with headings, bullet points, or lists when appropriate
6. Never make up information
7. Never reveal confidential information to unauthorized users
8. Add a disclaimer at the end: "This information is for internal company use only"

Context:
{context}

Question:
{question}

Answer (in properly formatted markdown, ensuring COMPLETE responses):
'''
)

class CompanyKnowledgeBot:
    def __init__(self, user_context=None):
        """Initialize the chatbot with user context."""
        self.user_context = user_context or {"clearance_level": "standard"}
        self.access_control = AccessControl(user_context)
        self.debug = True
        self.conversation_history = []
        
        # Load the LLM
        self.llm = LlamaCpp(
            model_path=Config.LLM_MODEL_PATH,
            n_ctx=8192,
            max_tokens=Config.MAX_TOKENS,
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
    
    def get_filtered_documents(self, query):
        """Get documents with relevance filtering and security filtering."""
        # Use similarity_search_with_score instead of just the retriever
        if self.debug:
            print(f"[DEBUG] Searching for: '{query}'")
        
        docs_with_scores = self.vectordb.similarity_search_with_score(
            query, 
            k=Config.RETRIEVAL_K
        )
        
        if self.debug:
            print(f"[DEBUG] Top scores: {[round(score, 3) for _, score in docs_with_scores[:3]]}")
        
        # Apply dual filtering: relevance score and security level
        filtered_docs = []
        for doc, score in docs_with_scores:
            # First check relevance, then security
            if score <= Config.MIN_RELEVANCE_SCORE:
                # Only check security if relevance is good
                passes_security = self.access_control.filter_document_by_metadata(doc)
                if passes_security:
                    filtered_docs.append(doc)
                    if self.debug:
                        source = getattr(doc, "metadata", {}).get("source", "unknown")
                        print(f"[DEBUG] Included doc from {source} with score {round(score, 3)}")
                else:
                    if self.debug:
                        source = getattr(doc, "metadata", {}).get("source", "unknown")
                        print(f"[DEBUG] Excluded doc from {source} with score {round(score, 3)} due to access denied")
            else:
                if self.debug:
                    source = getattr(doc, "metadata", {}).get("source", "unknown")
                    print(f"[DEBUG] Excluded doc from {source} with score {round(score, 3)} due to low relevance")
        
        return filtered_docs
    
    def answer(self, query):
        """Process the query with security checks and formatting."""
        # Store the query in conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Check if this might be a follow-up question
        original_query = query
        enhanced_query = query
        
        if len(self.conversation_history) > 1:
            is_followup = self.check_if_followup(query)
            if is_followup and self.debug:
                print(f"[DEBUG] Detected follow-up question: '{query}'")
                
            if is_followup:
                # Modify query to include context from previous exchanges
                enhanced_query = self.enhance_query_with_context(query)
                if self.debug:
                    print(f"[DEBUG] Enhanced query: '{enhanced_query}'")
        
        # Security check for restricted queries - always use original query for security
        allowed, restricted_topics = self.access_control.check_query_permission(original_query)
        if not allowed:
            response = (f"I'm sorry, but I can't provide information on {', '.join(restricted_topics)}. "
                    "This appears to be a restricted topic. Please contact your supervisor or the "
                    "security department if you need access to this information.")
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        try:
            # Run query with custom document retrieval
            # First get filtered documents
            relevant_docs = self.get_filtered_documents(enhanced_query)
            
            # Check if we have any relevant documents after filtering
            if not relevant_docs:
                response = "I couldn't find any relevant information in our knowledge base that matches your question and your access level. This topic may not be covered in our company documentation. Please contact the IT department for assistance."
                return response
            
            # Create a context string from the filtered documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Use the LLM with our prompt template
            prompt = COMPANY_KB_PROMPT.format(context=context, question=query)
            response = self.llm.invoke(prompt)
            response = self.ensure_complete_response(response)
            
            # Log the successful query
            sources = [getattr(doc, "metadata", {}).get("source", "unknown") for doc in relevant_docs]
            self.access_control.log_access("knowledge_query", query, "read", True, details={"sources": sources})
            
            # Apply post-processing to ensure confidentiality
            answer = self.post_process_response(response)
            
            # Add this line right before returning
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except Exception as e:
            # Log the error
            self.access_control.log_access("knowledge_query", query, "read", False, details={"error": str(e)})
            
            # Create error response
            error_msg = f"I encountered an error while processing your query. Please try again or contact support. Error: {str(e)}"
            
            # Add error response to conversation history
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            
            return error_msg
    
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
        # Ensure the response is complete after all processing
        response = self.ensure_complete_response(response)
    
        return response
    
    def ensure_complete_response(self, response):
        """Ensure response is complete by checking for cut-off sentences."""
        # Check if response ends with a complete sentence
        if not response.endswith(('.', '!', '?', '"', "'", ')', ']', '}', '*')):
            # If not, try to complete the last sentence or remove it
            last_sentence_end = max(
                response.rfind('.'), response.rfind('!'), 
                response.rfind('?'), response.rfind('*')
            )
            if last_sentence_end > 0:
                response = response[:last_sentence_end+1]
        
        # Ensure disclaimer is present
        if "This information is for internal company use only" not in response:
            response += "\n\n*This information is for internal company use only*"
        
        return response
    
    def check_if_followup(self, query):
        """Check if query seems to be a follow-up question."""
        # Simple heuristics for follow-up detection
        followup_indicators = [
            "it", "that", "this", "the", "these", "those",
            "summarize", "explain more", "tell me more",
            "can you", "what about", "how about"
        ]
        
        query_lower = query.lower()
        return (len(query_lower.split()) < 10 and  # Short query
                any(indicator in query_lower for indicator in followup_indicators))

    def enhance_query_with_context(self, query):
        """Add context from previous exchanges to the query."""
        # Get the last exchange
        previous_query = None
        
        for item in reversed(self.conversation_history[:-1]):
            if item["role"] == "user":
                previous_query = item["content"]
                break
        
        if previous_query:
            # Instead of creating a meta-query, combine the previous query with the follow-up
            enhanced_query = f"{previous_query} {query}"
            return enhanced_query
        
        return query

# For backward compatibility
def chatbot_answer(query, user_context=None):
    """Legacy function for compatibility with existing code."""
    bot = CompanyKnowledgeBot(user_context)
    return bot.answer(query)