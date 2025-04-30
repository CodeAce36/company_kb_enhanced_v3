"""Enhanced chatbot tester with session management and user context."""

from src.chatbot import CompanyKnowledgeBot
import json
import datetime
import os

def create_user_context(username="guest"):
    """Create a test user context."""
    # In a real application, this would come from authentication
    test_users = {
        "guest": {
            "user_id": "guest",
            "name": "Guest User",
            "department": "External",
            "clearance_level": "public"
        },
        "employee": {
            "user_id": "emp123",
            "name": "Regular Employee",
            "department": "Sales",
            "clearance_level": "standard"
        },
        "manager": {
            "user_id": "mgr456",
            "name": "Department Manager",
            "department": "IT",
            "clearance_level": "confidential"
        }
    }
    
    return test_users.get(username, test_users["guest"])

def run_chatbot_tester():
    """Run an interactive session with the chatbot."""
    # Ensure logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    print("\n" + "="*50)
    print(" COMPANY KNOWLEDGE BOT - TEST INTERFACE ")
    print("="*50 + "\n")
    
    # Ask for test user type
    print("Available test users:")
    print("1. guest - Public access only")
    print("2. employee - Standard internal access")
    print("3. manager - Confidential access")
    
    while True:
        choice = input("\nSelect user type (1-3, default=2): ").strip()
        if not choice:
            choice = "2"
            
        if choice == "1":
            username = "guest"
            break
        elif choice == "2":
            username = "employee"
            break
        elif choice == "3":
            username = "manager"
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
    
    user_context = create_user_context(username)
    print(f"\nLogged in as: {user_context['name']} ({user_context['department']})")
    print(f"Access level: {user_context['clearance_level']}")
    
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