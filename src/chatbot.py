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
1. ONLY answer using the context provided below - NEVER make up information
2. If the answer isn't in the context OR the context seems irrelevant to the question, say: "I don't have enough information to answer that question. This topic may not be covered in our company knowledge base. Please contact the IT department for assistance."
3. For FACTUAL information such as names, dates, or founding details, directly quote the exact text from the context
4. Do not elaborate beyond what is explicitly stated in the context
5. Keep answers concise but complete
6. Ensure responses are NEVER truncated - always complete your thoughts
7. Format responses properly with headings, bullet points, or lists when appropriate
8. Never make up information even if it seems plausible
9. Never reveal confidential information to unauthorized users
10. Add a disclaimer at the end: "This information is for internal company use only"

Context:
{context}

Question:
{question}

Answer (in properly formatted markdown, ensuring COMPLETE responses based SOLELY on the context above):
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
    
    def get_enhanced_documents(self, query):
        """Get documents with enhanced retrieval for key terms."""
        if self.debug:
            print(f"[DEBUG] Searching for: '{query}'")
        
        # Initial similarity search
        docs_with_scores = self.vectordb.similarity_search_with_score(
            query, 
            k=Config.RETRIEVAL_K * 2  # Retrieve more docs initially
        )
        
        if self.debug:
            print(f"[DEBUG] Top scores: {[round(score, 3) for _, score in docs_with_scores[:3]]}")
        
        # Check if query is about key company facts
        key_fact_terms = [
            'founder', 'founded', 'established', 'started', 
            'ceo', 'president', 'owner', 'leadership',
            'headquarters', 'location', 'address'
        ]
        
        is_key_fact_query = any(term in query.lower() for term in key_fact_terms)
        
        # For key fact queries, prioritize documents with exact term matches
        if is_key_fact_query:
            # Create patterns for each key term in the query
            query_terms = query.lower().split()
            exact_terms = [term for term in query_terms if term in key_fact_terms]
            
            if exact_terms:
                # Rerank documents based on exact matches
                reranked_docs = []
                other_docs = []
                
                for doc, score in docs_with_scores:
                    doc_content = doc.page_content.lower()
                    
                    # Check for exact term matches
                    matches = sum(1 for term in exact_terms if term in doc_content)
                    
                    if matches > 0:
                        # Boost score based on number of matches (lower is better)
                        adjusted_score = score * (0.7 ** matches)
                        reranked_docs.append((doc, adjusted_score))
                    else:
                        other_docs.append((doc, score))
                
                # Sort reranked docs by adjusted score
                reranked_docs.sort(key=lambda x: x[1])
                
                # Combine reranked docs with others
                docs_with_scores = reranked_docs + other_docs
        
        # Apply security and relevance filtering
        filtered_docs = []
        for doc, score in docs_with_scores:
            # Check if document passes security filter
            passes_security = self.access_control.filter_document_by_metadata(doc)
            
            # Apply relevance threshold
            passes_relevance = score <= Config.MIN_RELEVANCE_SCORE
            
            if passes_security and passes_relevance:
                filtered_docs.append(doc)
                if self.debug:
                    source = getattr(doc, "metadata", {}).get("source", "unknown")
                    print(f"[DEBUG] Included doc from {source} with score {round(score, 3)}")
            else:
                if self.debug:
                    source = getattr(doc, "metadata", {}).get("source", "unknown")
                    reason = "access denied" if not passes_security else "low relevance"
                    print(f"[DEBUG] Excluded doc from {source} with score {round(score, 3)} due to {reason}")
        
        return filtered_docs
    
    def answer(self, query):
        """Process the query with enhanced accuracy for key facts."""
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
            # Run query with enhanced document retrieval
            relevant_docs = self.get_enhanced_documents(enhanced_query)
            
            # Check if we have any relevant documents after filtering
            if not relevant_docs:
                response = "I couldn't find any relevant information in our knowledge base that matches your question and your access level. This topic may not be covered in our company documentation. Please contact the IT department for assistance."
                self.conversation_history.append({"role": "assistant", "content": response})
                return response
            
            # Create a context string from the filtered documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Use the LLM with our enhanced prompt template
            prompt = COMPANY_KB_PROMPT.format(context=context, question=query)
            response = self.llm.invoke(prompt)
            
            # Log the successful query
            sources = [getattr(doc, "metadata", {}).get("source", "unknown") for doc in relevant_docs]
            self.access_control.log_access("knowledge_query", query, "read", True, details={"sources": sources})
            
            # Apply standard post-processing
            response = self.post_process_response(response)
            
            # Enhance with direct quotes for key information
            response = self.enhance_with_direct_quotes(response, query, relevant_docs)
            
            # Ensure complete response
            response = self.ensure_complete_response(response)
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
                
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
    
    def enhance_with_direct_quotes(self, response, query, relevant_docs):
        """Enhance response with direct quotes for key factual information."""
        # Define patterns for important factual information
        key_fact_patterns = [
            r'founder', r'founded by', r'started by', r'created by',
            r'ceo', r'president', r'owner',
            r'established', r'founded in', r'started in',
            r'headquarters', r'located'
        ]
        
        # Check if this is a query about key facts
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in key_fact_patterns):
            # Look for relevant sentences in the source documents
            direct_quotes = []
            
            for doc in relevant_docs:
                # Split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
                
                # Look for sentences containing the key information
                for sentence in sentences:
                    if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in key_fact_patterns):
                        # Clean up the sentence
                        clean_sentence = sentence.strip()
                        
                        # Add to direct quotes if not already included
                        if clean_sentence and clean_sentence not in direct_quotes:
                            direct_quotes.append(clean_sentence)
            
            # Add direct quotes to the response if found
            if direct_quotes:
                quote_section = "\n\n**Direct quotes from company knowledge base:**\n\n"
                for i, quote in enumerate(direct_quotes[:3]):  # Limit to top 3 quotes
                    quote_section += f"> {quote}\n\n"
                
                # Add the quotes before the disclaimer
                if "This information is for internal company use only" in response:
                    response = response.replace(
                        "This information is for internal company use only", 
                        f"{quote_section}*This information is for internal company use only*"
                    )
                else:
                    response += quote_section
        
        return response

# For backward compatibility
def chatbot_answer(query, user_context=None):
    """Legacy function for compatibility with existing code."""
    bot = CompanyKnowledgeBot(user_context)
    return bot.answer(query)