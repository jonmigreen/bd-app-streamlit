"""Main Streamlit application for OpenAI chat with vector store RAG."""
import sys
import os

import streamlit as st
from openai_client import OpenAIClient
from config import Config

# Verify we're using the correct OpenAI package (not system Python)
try:
    import openai
    openai_path = os.path.dirname(openai.__file__)
    openai_version = openai.__version__
    
    # Check if we're using system Python's packages
    if '/Library/Frameworks/Python.framework' in openai_path:
        st.error(f"⚠️ **Wrong Python Environment Detected!**\n\n"
                 f"Streamlit is using the system Python (OpenAI {openai_version}) instead of your virtual environment.\n\n"
                 f"**Solution:**\n"
                 f"1. Stop this app (Ctrl+C)\n"
                 f"2. Activate your virtual environment: `source venv/bin/activate`\n"
                 f"3. Verify: `which python` should show `.../venv/bin/python`\n"
                 f"4. Run Streamlit again: `streamlit run app.py`\n\n"
                 f"Current OpenAI location: `{openai_path}`")
        st.stop()
    
    # Verify we have the correct OpenAI version (1.x, not 2.x)
    if openai_version.startswith('2.'):
        st.error(f"⚠️ **Incompatible OpenAI Version!**\n\n"
                 f"OpenAI SDK version {openai_version} is installed, but this app requires version 1.x.\n\n"
                 f"**Solution:**\n"
                 f"1. Activate your virtual environment: `source venv/bin/activate`\n"
                 f"2. Reinstall: `pip install 'openai>=1.12.0,<2.0.0'`\n"
                 f"3. Restart Streamlit")
        st.stop()
except Exception as e:
    st.error(f"Error checking OpenAI installation: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="OpenAI Chat with RAG",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "openai_client" not in st.session_state:
    try:
        st.session_state.openai_client = OpenAIClient()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.stop()

# App title and description
st.title("💬 OpenAI Chat with Vector Store RAG")
st.markdown("Ask questions and get answers enhanced with context from your vector store.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.info(f"Model: {Config.OPENAI_MODEL}")
    st.info(f"Vector Store ID: {Config.OPENAI_VECTOR_STORE_ID[:20]}..." if Config.OPENAI_VECTOR_STORE_ID else "Not configured")
    
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Querying vector store and generating response..."):
            try:
                # Get conversation history (excluding the current user message for the API call)
                conversation_history = st.session_state.messages[:-1]
                
                # Get RAG-enhanced response
                response = st.session_state.openai_client.get_rag_response(
                    user_query=prompt,
                    conversation_history=conversation_history
                )
                
                # Display response
                message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except ValueError as e:
                # Configuration errors
                error_message = f"⚠️ Configuration Error: {str(e)}\n\nPlease check your .env file and ensure all required variables are set."
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                
            except Exception as e:
                # API or other errors
                error_message = f"❌ Error: {str(e)}\n\nPlease try again or check your API configuration."
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

