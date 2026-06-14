import logging
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import current_app

def get_llm():
    """
    Factory function to get the configured LLM client.
    """
    provider = current_app.config.get("LLM_PROVIDER", "openai").lower()
    
    if provider == "gemini":
        api_key = current_app.config.get("GOOGLE_API_KEY")
        model_name = current_app.config.get("GEMINI_MODEL", "gemini-2.0-flash")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )
        
    elif provider == "openai":
        api_key = current_app.config.get("OPENAI_API_KEY")
        model_name = current_app.config.get("OPENAI_MODEL", "gpt-4.1-nano")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
            
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
        )
        
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
