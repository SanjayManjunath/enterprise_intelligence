import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# --- PATH LOGIC: Looking for .env in the Project Root ---
base_dir = os.path.dirname(__file__)
env_path = os.path.abspath(os.path.join(base_dir, "../../.env"))

if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    print(f"⚠️ Warning: .env not found at {env_path}. Ensure environment variables are set manually.")

def get_llm(model_provider="groq"):
    """
    Centralized Factory for LLM initialization.
    Principal DS Standard: Forces temperature=0 for deterministic SQL and Logic reasoning.
    """
    
    # 1. Groq Configuration (Primary Reasoning Engine)
    if model_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ ERROR: GROQ_API_KEY not found in environment.")
            
        return ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=api_key,
            max_tokens=None, 
            timeout=None,
            max_retries=2
        )
    
    # 2. Google Configuration (Backup/Multimodal Engine)
    elif model_provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ ERROR: GOOGLE_API_KEY not found in environment.")
            
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
    
    # 3. OpenAI Configuration (Alternative Reasoning Engine)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ ERROR: OPENAI_API_KEY not found in environment.")
            
        # CORRECTED ROUTING:
        # If 'openai_smart' is passed, we use the full gpt-4o.
        # Otherwise, we default to the budget-friendly gpt-4o-mini.
        model_name = "gpt-4o" if model_provider == "openai_smart" else "gpt-4o-mini"
        
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
            max_retries=2, 
            model_kwargs={"seed": 42}
        )