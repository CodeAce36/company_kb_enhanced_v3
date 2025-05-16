"""Enhanced chatbot tester with session management and user context."""

from src.chatbot import CompanyKnowledgeBot
import json
import datetime
import os

def create_user_context(username="guest"):
    """Create a test user context."""
    # All users now have full access
    return {
        "user_id": username,
        "name": "User",
        "department": "All",
        "clearance_level": "confidential"
    }

def run_chatbot_tester():
    """Run an interactive session with the chatbot."""
    # Ensure logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    print("\n" + "="*50)
    print(" COMPANY KNOWLEDGE BOT - TEST INTERFACE ")
    print("="*50 + "\n")
    
    # Create user context with full access
    user_context = create_user_context()
    print(f"\nLogged in as: {user_context['name']}")
    print(f"Access level: Full access")
    
    # Create a chatbot instance with the user context
    chatbot = CompanyKnowledgeBot(user_context)
    
    # Start chat session
    session_history = []
    session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    print("\n[Company Knowledge Bot] Ready to answer your questions.\n")
    print("Type 'exit' to quit, 'history' to see conversation history.\n")
    
    while True:
        question = input("Question: ")
        
        if question.lower() == "exit":
            break
        elif question.lower() == "history":
            print("\n--- Conversation History ---")
            for idx, exchange in enumerate(session_history, 1):
                print(f"\n[Q{idx}] {exchange['question']}")
                print(f"\n[A{idx}] {exchange['answer']}\n")
            print("----------------------------\n")
            continue
            
        # Get answer from chatbot
        answer = chatbot.answer(question)
        
        # Record in session history
        session_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Display answer
        print("\n[Answer]:\n")
        print(answer)
        print("\n" + "-"*50)
    
    # Option to save session
    if session_history and input("\nSave this session? (y/n): ").lower() == 'y':
        # Ensure sessions directory exists
        if not os.path.exists('logs/sessions'):
            os.makedirs('logs/sessions')
        
        filename = f"logs/sessions/session_{session_id}.json"
        with open(filename, 'w') as f:
            json.dump({
                "user": user_context,
                "session_id": session_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "history": session_history
            }, f, indent=2)
        print(f"Session saved to {filename}")
    
    print("\nThank you for using the Company Knowledge Bot.")

if __name__ == "__main__":
    run_chatbot_tester()