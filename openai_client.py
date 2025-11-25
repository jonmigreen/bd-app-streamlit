"""OpenAI API client for vector store queries and chat completions."""
from typing import List, Dict, Optional
import openai
from openai import OpenAI
from config import Config


class OpenAIClient:
    """Client for interacting with OpenAI vector store and chat API."""
    
    def __init__(self):
        """Initialize OpenAI client with API key."""
        Config.validate()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.vector_store_id = Config.OPENAI_VECTOR_STORE_ID
        self.model = Config.OPENAI_MODEL
        self.temperature = Config.OPENAI_TEMPERATURE
    
    def query_vector_store(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query the OpenAI vector store for relevant context.
        
        This method uses the Assistants API with file_search tool to retrieve
        relevant context from the vector store. The vector store must be
        associated with files that have been indexed.
        
        Args:
            query: The search query string
            top_k: Number of results to return (approximate, as Assistants API handles retrieval)
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Use Assistants API with file_search to query vector store
            # Create a lightweight assistant for RAG queries
            assistant = self.client.beta.assistants.create(
                name="RAG Query Assistant",
                model=self.model,
                tools=[{"type": "file_search"}],
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [self.vector_store_id]
                    }
                },
                instructions="You are a helpful assistant that retrieves relevant information from the knowledge base."
            )
            
            # Create a thread and add the query
            thread = self.client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Retrieve relevant information about: {query}"
                    }
                ]
            )
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Poll for completion
            import time
            max_wait = 30  # Maximum wait time in seconds
            start_time = time.time()
            
            while run.status in ['queued', 'in_progress']:
                if time.time() - start_time > max_wait:
                    raise Exception("Vector store query timed out")
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status != 'completed':
                raise Exception(f"Vector store query failed with status: {run.status}")
            
            # Retrieve the assistant's response with context
            messages = self.client.beta.threads.messages.list(
                thread_id=thread.id,
                order="desc",
                limit=1
            )
            
            results = []
            if messages.data:
                assistant_message = messages.data[0]
                for content_block in assistant_message.content:
                    if content_block.type == 'text':
                        text_value = content_block.text.value
                        # Extract annotations for file citations
                        annotations = getattr(content_block.text, 'annotations', [])
                        file_ids = []
                        for annotation in annotations:
                            if hasattr(annotation, 'file_citation'):
                                file_ids.append(annotation.file_citation.file_id)
                        
                        results.append({
                            'content': text_value,
                            'metadata': {'file_ids': file_ids} if file_ids else {}
                        })
            
            # Clean up assistant
            try:
                self.client.beta.assistants.delete(assistant.id)
            except:
                pass
            
            return results[:top_k] if results else []
            
        except Exception as e:
            error_msg = str(e)
            if "vector_store" in error_msg.lower() or "vector store" in error_msg.lower():
                raise Exception(f"Vector store query failed: {error_msg}. "
                              f"Ensure your vector store ID '{self.vector_store_id}' is correct and contains indexed files.")
            elif "timeout" in error_msg.lower():
                raise Exception("Vector store query timed out. The query took too long to process.")
            else:
                raise Exception(f"Vector store query failed: {error_msg}")
    
    def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
        stream: bool = False
    ):
        """
        Get chat completion from OpenAI API with optional RAG context.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            context: Optional context string from vector store to include in system message
            stream: Whether to stream the response
            
        Returns:
            Chat completion response or generator for streaming
        """
        # Prepare messages with context if provided
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant."
        }
        
        if context:
            system_message["content"] += f"\n\nUse the following context to answer questions:\n\n{context}"
        
        # Insert system message at the beginning if not already present
        prepared_messages = [system_message]
        if messages and messages[0].get("role") != "system":
            prepared_messages.extend(messages)
        else:
            prepared_messages = messages
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prepared_messages,
                temperature=self.temperature,
                stream=stream
            )
            
            return response
            
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise ValueError(f"Authentication failed: {error_msg}. Please check your OPENAI_API_KEY.")
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                raise Exception(f"Rate limit exceeded: {error_msg}. Please wait a moment and try again.")
            else:
                raise Exception(f"Chat completion failed: {error_msg}")
    
    def get_rag_response(self, user_query: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Get RAG-enhanced response by querying vector store and generating chat completion.
        
        Args:
            user_query: The user's question or message
            conversation_history: Previous messages in the conversation
            
        Returns:
            Assistant's response text
        """
        # Step 1: Query vector store for relevant context
        vector_results = self.query_vector_store(user_query)
        
        # Step 2: Combine context from vector store results
        context_parts = []
        for result in vector_results:
            content = result.get('content', '')
            if content:
                context_parts.append(content)
        
        context = "\n\n".join(context_parts) if context_parts else None
        
        # Step 3: Prepare messages with user query
        messages = conversation_history.copy() if conversation_history else []
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        # Step 4: Get chat completion with context
        response = self.get_chat_completion(messages, context=context, stream=False)
        
        # Step 5: Extract response text
        if hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            raise Exception("No response generated from chat completion")

