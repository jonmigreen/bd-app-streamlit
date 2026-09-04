"""Configuration management for OpenAI API and Streamlit app."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
    
    # Model Configuration
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")

    # Reasoning effort for GPT-5.x models. Reasoning tokens bill as output
    # tokens, so higher effort costs more; "low" suits retrieval-grounded
    # summarization. Note that these models reject `temperature` unless
    # effort is "none".
    OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")

    VALID_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")

    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        if not cls.OPENAI_VECTOR_STORE_ID:
            raise ValueError("OPENAI_VECTOR_STORE_ID is not set in environment variables")
        if cls.OPENAI_REASONING_EFFORT not in cls.VALID_REASONING_EFFORTS:
            raise ValueError(
                f"OPENAI_REASONING_EFFORT must be one of "
                f"{', '.join(cls.VALID_REASONING_EFFORTS)} "
                f"(got '{cls.OPENAI_REASONING_EFFORT}')"
            )
        return True

