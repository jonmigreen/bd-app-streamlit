# OpenAI Chat with Vector Store RAG - Streamlit App

A conversational chat interface built with Streamlit that integrates OpenAI's vector store for Retrieval-Augmented Generation (RAG) and OpenAI's chat completion API.

## Features

- 💬 Continuous conversational chat interface with RAG-enhanced responses
- 🔍 Research Mode for direct document exploration
- 📚 Sources panel showing exact snippets and relevance scores
- 🐛 Debug Mode for technical details and evaluation
- 📝 Session-based chat history (resets on app restart)
- ⚙️ Configurable model and temperature settings
- 🎨 Clean, modern UI with two-column layout
- 📊 Query logging for debugging and quality evaluation

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

### Basic Chat Mode (Default)

1. Start the Streamlit app using the command above
2. Type your question in the chat input at the bottom
3. The app will:
   - Query your vector store for relevant context using Assistants API
   - Generate a response using OpenAI's chat API with the retrieved context
   - Display the response in the left column
   - Show sources in the right column with exact snippets and relevance scores
4. Continue the conversation - previous messages are maintained in the session
5. Use the "Clear Chat History" button in the sidebar to reset the conversation

### Research Mode

1. Enable "Research Mode" checkbox in the sidebar
2. Enter a search query in the search box
3. Optionally set filters (Client, Year, Topic) to narrow results
4. Click "Search" to explore documents directly
5. Select snippets using checkboxes and click "Summarize Selected" to generate summaries

### Debug Mode

1. Enable "Debug Mode" checkbox in the sidebar
2. Technical details (file IDs, relevance scores, API metadata) will be shown
3. Useful for understanding why certain documents were retrieved and evaluating retrieval quality

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

### Basic Chat Mode (Assistant-First)

The app uses a dual-API approach for maximum transparency and control:

1. **User Query**: User types a question in the chat interface

2. **Parallel API Calls**:
   - **Assistants API Call**: Uses OpenAI's Assistants API with `file_search` tool to:
     - Query the vector store intelligently (with query rewriting and ranking)
     - Retrieve relevant context automatically
     - Generate a conversational answer
   - **Direct Vector Search Call**: Simultaneously calls `vector-stores.search` API to:
     - Get exact snippets from matched documents
     - Retrieve relevance scores for each snippet
     - Provide traceable source information

3. **Response Display**:
   - **Left Column**: Shows the AI-generated answer from Assistants API
   - **Right Column**: Shows the "Sources" panel with:
     - Each matched document (grouped by filename)
     - Exact snippets that were retrieved
     - Relevance scores (builds trust by showing what the model actually saw)
     - Checkboxes to select snippets for summarization

4. **Sources Panel Features**:
   - Sources are grouped by filename for easy navigation
   - Each snippet shows a preview (2-3 sentences) with full content available
   - Relevance scores are displayed to show retrieval quality
   - Filters can be applied (Client, Year, Topic) to narrow search results
   - Selected snippets can be summarized using the "Summarize Selected Snippets" button

**Why This Approach?**
- **Transparency**: Users can see exactly what documents the model used
- **Trust**: Relevance scores show why certain documents were retrieved
- **Control**: Users can verify the model looked at the right documents
- **Debugging**: Helps identify noisy documents or poor chunks that should be removed

### Research Mode (Search-First)

Research Mode provides direct document exploration without the chat interface:

1. **Enable Research Mode**: Toggle the checkbox in the sidebar

2. **Two-Pane Layout**:
   - **Left Pane**: 
     - Filter controls (Client, Year, Topic)
     - Search query input
     - Summary area (when snippets are selected)
   - **Right Pane**:
     - Search results grouped by filename
     - Each snippet with checkbox for selection
     - Relevance scores displayed

3. **Workflow**:
   - Enter search query (e.g., "fiscal management")
   - Optionally set filters (e.g., Client = "DCC", Year = "2024")
   - Click "Search" to retrieve matching documents
   - Select relevant snippets using checkboxes
   - Click "Summarize Selected" to generate AI summary in left pane

4. **Use Cases**:
   - Exploring documents without asking questions
   - Finding specific information across multiple documents
   - Research workflows where you need to see raw snippets first
   - Quality evaluation: checking if your vector store contains relevant information

**Key Difference from Chat Mode**:
- Research Mode uses direct vector search only (no Assistants API for initial retrieval)
- You see raw search results first, then optionally summarize selected snippets
- Better for exploratory research and document discovery

### Debug Mode

Debug Mode reveals technical details for developers and evaluators:

1. **Enable Debug Mode**: Toggle the checkbox in the sidebar

2. **What You'll See**:
   - **File IDs**: Technical identifiers for each document
   - **Relevance Scores**: Exact numerical scores from vector search
   - **API Metadata**: Response details from OpenAI APIs
   - **Session State**: Current app state (message count, sources count, etc.)
   - **Vector Store ID**: Your configured vector store identifier

3. **Use Cases**:
   - **Quality Evaluation**: Check if retrieval is working correctly
   - **Debugging**: Understand why certain documents were or weren't retrieved
   - **Optimization**: Identify noisy documents or poor chunks to remove/re-chunk
   - **Development**: Verify API calls and responses

4. **Logging**:
   - Each query is automatically logged to `query_log.jsonl`:
     - Timestamp
     - Question asked
     - Assistant's answer
     - Top 5 search results (filename + score)
     - Applied filters
   - Logs help answer: "Why did it say that?" and "Did it look at the right documents?"

### Filtering

Filters allow you to narrow search results by metadata:

- **Client**: Filter by client name (if stored in document metadata)
- **Year**: Filter by year (if stored in document metadata)
- **Topic**: Filter by topic/category (if stored in document metadata)

**How Filters Work**:
- Filters are applied client-side after retrieval (metadata filtering at API level coming soon)
- Empty filter values are ignored (all documents searched)
- Filters work in both Chat Mode (Sources panel) and Research Mode
- Useful for large vector stores with diverse document types

### Summarize Selected Snippets

This feature lets you control exactly which snippets are summarized:

1. **Select Snippets**: Use checkboxes in the Sources panel to select specific snippets
2. **Click "Summarize Selected Snippets"**: Button appears when snippets are selected
3. **AI Summary**: Selected snippets are sent to Assistants API with a summarization prompt
4. **Result**: Summary appears in chat history, showing which exact passages were used

**Benefits**:
- Traceable summaries: you know exactly what was summarized
- Quality control: you choose which snippets are relevant
- Better for research workflows where you want to review sources first

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

## Architecture Details

### API Integration

The app uses two complementary OpenAI APIs:

1. **Assistants API** (`client.beta.assistants`):
   - Used for conversational chat responses
   - Provides intelligent query rewriting and ranking
   - Handles file_search tool automatically
   - Creates temporary assistants for RAG queries (can be optimized for production)

2. **Vector Store Search API** (`client.beta.vector_stores.search`):
   - Used for direct document retrieval
   - Returns exact snippets with relevance scores
   - Provides traceable source information
   - Enables filtering and quality evaluation

### Session State Management

The app maintains several session state variables:

- `messages`: Chat conversation history
- `message_sources`: Sources for each assistant message (indexed by message position)
- `current_sources`: Sources for the current/last question
- `selected_snippets`: Indices of selected snippets for summarization
- `filters`: Current filter values (client, year, topic)
- `research_mode`: Whether Research Mode is enabled
- `debug_mode`: Whether Debug Mode is enabled
- `research_results`: Results from Research Mode searches

### File Structure

```
├── app.py                 # Main Streamlit application with two-column layout
├── openai_client.py       # OpenAI API integration:
│                          #   - search_vectors(): Simple vector search helper
│                          #   - direct_vector_search(): Direct API call
│                          #   - get_rag_response(): Assistants API with sources
│                          #   - get_sources_for_query(): Formatted source retrieval
├── config.py             # Configuration management (env vars)
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this, gitignored)
├── .env.example          # Environment template
├── query_log.jsonl       # Query logging (gitignored)
├── run.sh                # Helper script to run app
└── README.md             # This file
```

## Notes

- Chat history is session-only and will reset when you restart the app
- Sources panel shows results from direct vector search (parallel to Assistants API)
- Relevance scores help build trust and enable quality evaluation
- Query logging helps debug retrieval issues and identify problematic documents
- Filters are currently applied client-side (server-side filtering coming in future API updates)
- Each Assistants API query creates a temporary assistant (can be optimized by reusing assistants in production)

