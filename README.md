# OpenAI Chat with Vector Store RAG - Streamlit App

A conversational chat interface built with Streamlit that integrates OpenAI's vector store for Retrieval-Augmented Generation (RAG) and OpenAI's chat completion API.

## Features

- 💬 Continuous conversational chat interface
- 🔍 RAG-enhanced responses using OpenAI vector store
- 📝 Session-based chat history (resets on app restart)
- ⚙️ Configurable model and temperature settings
- 🎨 Clean, modern UI with Streamlit

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_VECTOR_STORE_ID=your_vector_store_id_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.7
```

**Required:**
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_VECTOR_STORE_ID`: Your OpenAI vector store ID

**Optional:**
- `OPENAI_MODEL`: Model to use (default: `gpt-4-turbo-preview`)
- `OPENAI_TEMPERATURE`: Temperature for responses (default: `0.7`)

### 3. Run the Application

**Option 1: Use the helper script (Recommended)**
```bash
./run.sh
```

**Option 2: Manual activation**
```bash
# Activate the virtual environment
source venv/bin/activate

# Run Streamlit using Python module (ensures correct environment)
python -m streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

**Troubleshooting:** If you see an error about "unexpected keyword argument 'proxies'", it means Streamlit is using the system Python instead of your venv. Use `python -m streamlit run app.py` instead of just `streamlit run app.py` to ensure the correct Python environment is used.

## Usage

1. Start the Streamlit app using the command above
2. Type your question in the chat input at the bottom
3. The app will:
   - Query your vector store for relevant context
   - Generate a response using OpenAI's chat API with the retrieved context
   - Display the response in the chat interface
4. Continue the conversation - previous messages are maintained in the session
5. Use the "Clear Chat History" button in the sidebar to reset the conversation

## Project Structure

```
├── app.py                 # Main Streamlit application
├── openai_client.py       # OpenAI API integration (vector store + chat)
├── config.py             # Configuration management
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── .env.example          # Environment template
└── README.md             # This file
```

## How It Works

1. **User Query**: User types a question in the chat interface
2. **Vector Store Query**: The app queries your OpenAI vector store for relevant context
3. **Context Retrieval**: Relevant document chunks are retrieved from the vector store
4. **Chat Completion**: The user query and retrieved context are sent to OpenAI's chat completion API
5. **Response Display**: The generated response is displayed in the chat interface

## Troubleshooting

### Configuration Errors
- Ensure your `.env` file exists and contains all required variables
- Check that your API key and vector store ID are correct

### Vector Store Errors
- Verify your vector store ID is correct
- Ensure your vector store contains indexed files
- Check that your vector store is accessible with your API key

### API Errors
- Verify your OpenAI API key is valid and has sufficient credits
- Check for rate limiting - wait a moment and try again
- Ensure you have access to the model specified in your configuration

## Notes

- Chat history is session-only and will reset when you restart the app
- The vector store query uses OpenAI's Assistants API with file_search tool
- Each query creates a temporary assistant for RAG retrieval (this can be optimized for production use)

