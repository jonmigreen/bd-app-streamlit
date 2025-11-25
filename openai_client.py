"""OpenAI API client for vector store queries and chat completions."""
from typing import List, Dict, Optional
from openai import OpenAI
import requests
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
    
    def search_vectors(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Simple helper function to search vector store and return formatted results.
        
        This is a wrapper around direct_vector_search that provides a cleaner interface
        for the UI. Returns simple list with filename, snippet, score, and metadata.
        
        Args:
            query: The search query string
            top_k: Number of results to return (default: 10)
            filters: Optional metadata filters dict (e.g., {"client": "DCC", "year": "2024"})
            
        Returns:
            List of results: [{"filename": "...", "snippet": "...", "score": 0.89, "metadata": {...}}]
        """
        # Try direct search first, fallback to Assistants API method if not available
        results_already_formatted = False
        try:
            results = self.direct_vector_search(query, max_num_results=top_k)
        except Exception as e:
            # If direct search API is not available, use Assistants API method
            error_msg = str(e).lower()
            if "not available" in error_msg or "method" in error_msg or "attribute" in error_msg:
                # Fallback to Assistants API approach
                raw_results = self.query_vector_store(query, top_k=top_k)
                # Convert to search_vectors format
                results = []
                for result in raw_results:
                    content = result.get('content', '')
                    if content:  # Only include results with content
                        # Get filename
                        filename = (
                            result.get('metadata', {}).get('filename') or
                            'Document'
                        )
                        # Create preview
                        sentences = content.split('. ')
                        preview = '. '.join(sentences[:3]) + ('...' if len(sentences) > 3 else '')
                        
                        results.append({
                            'filename': filename,
                            'snippet': content,  # Full snippet
                            'preview': preview,  # Short preview
                            'score': None,  # Assistants API doesn't provide scores
                            'file_id': result.get('file_id'),
                            'metadata': result.get('metadata', {})
                        })
                results_already_formatted = True
            else:
                # Re-raise if it's a different error
                raise
        
        # Format results for simple display (only if not already formatted)
        if results_already_formatted:
            formatted = results
        else:
            formatted = []
            for result in results:
                snippet = result.get('content', '')
                # Ensure we have content
                if not snippet:
                    continue  # Skip results without content
                
                # Truncate snippet to first 2-3 sentences for preview
                sentences = snippet.split('. ')
                preview_snippet = '. '.join(sentences[:3]) + ('...' if len(sentences) > 3 else '')
                
                # Get filename from multiple possible locations
                filename = (
                    result.get('metadata', {}).get('filename') or
                    result.get('filename') or
                    'Document'
                )
                
                formatted.append({
                    'filename': filename,
                    'snippet': snippet,  # Full snippet
                    'preview': preview_snippet,  # Short preview
                    'score': result.get('score'),
                    'file_id': result.get('file_id'),
                    'metadata': result.get('metadata', {})
                })
        
        # Apply filters if provided (client-side filtering for now)
        if filters:
            filtered = []
            for item in formatted:
                match = True
                for key, value in filters.items():
                    if value:  # Only filter if value is provided
                        item_metadata = item.get('metadata', {})
                        if key not in item_metadata or item_metadata[key] != value:
                            match = False
                            break
                if match:
                    filtered.append(item)
            return filtered
        
        return formatted
    
    def direct_vector_search(self, query: str, max_num_results: int = 10) -> List[Dict]:
        """
        Directly search the vector store using the search API endpoint.
        
        This method uses the direct vector store search API, which is faster
        and more efficient than using the Assistants API. Useful for research
        mode and source retrieval.
        
        Args:
            query: The search query string
            max_num_results: Maximum number of results to return (default: 10)
            
        Returns:
            List of search results with content, metadata, and scores
        """
        try:
            # Try beta API first (for older SDK versions)
            if hasattr(self.client.beta, 'vector_stores') and hasattr(self.client.beta.vector_stores, 'search'):
                response = self.client.beta.vector_stores.search(
                    vector_store_id=self.vector_store_id,
                    query=query,
                    max_num_results=max_num_results
                )
            # Try non-beta API (for newer SDK versions)
            elif hasattr(self.client, 'vector_stores') and hasattr(self.client.vector_stores, 'search'):
                response = self.client.vector_stores.search(
                    vector_store_id=self.vector_store_id,
                    query=query,
                    max_num_results=max_num_results
                )
            else:
                # Neither method available, use REST API fallback
                return self._vector_search_via_rest_api(query, max_num_results)
            
            results = []
            if hasattr(response, 'data') and response.data:
                for item in response.data:
                    result = {
                        'content': '',
                        'metadata': {},
                        'score': getattr(item, 'score', None),
                        'file_id': None
                    }
                    
                    # Extract content
                    if hasattr(item, 'content'):
                        if isinstance(item.content, list) and len(item.content) > 0:
                            # Content is a list of content blocks
                            content_texts = []
                            for content_block in item.content:
                                if hasattr(content_block, 'text'):
                                    content_texts.append(content_block.text)
                                elif isinstance(content_block, dict) and 'text' in content_block:
                                    content_texts.append(content_block['text'])
                            result['content'] = '\n\n'.join(content_texts)
                        elif isinstance(item.content, str):
                            result['content'] = item.content
                    
                    # Extract metadata
                    if hasattr(item, 'metadata'):
                        result['metadata'] = item.metadata if isinstance(item.metadata, dict) else {}
                    
                    # Extract file_id
                    if hasattr(item, 'file_id'):
                        result['file_id'] = item.file_id
                    elif hasattr(item, 'id'):
                        result['file_id'] = item.id
                    
                    # Extract filename from multiple possible locations
                    filename = None
                    if hasattr(item, 'filename') and item.filename:
                        filename = item.filename
                    elif hasattr(item, 'name') and item.name:
                        filename = item.name
                    elif isinstance(item.metadata, dict) and 'filename' in item.metadata:
                        filename = item.metadata['filename']
                    
                    # If we have a file_id but no filename, try to fetch it
                    if not filename and result['file_id']:
                        try:
                            file_info = self.client.files.retrieve(result['file_id'])
                            if hasattr(file_info, 'filename'):
                                filename = file_info.filename
                            elif hasattr(file_info, 'name'):
                                filename = file_info.name
                        except Exception:
                            # If file retrieval fails, use file_id as fallback
                            pass
                    
                    # Set filename in metadata
                    if filename:
                        result['metadata']['filename'] = filename
                    elif result['file_id']:
                        # Use a shortened file_id as fallback
                        file_id_short = result['file_id'][:8] + '...' if len(result['file_id']) > 8 else result['file_id']
                        result['metadata']['filename'] = f"Document ({file_id_short})"
                    else:
                        result['metadata']['filename'] = "Document"
                    
                    results.append(result)
            
            return results
            
        except AttributeError:
            # SDK method not available, try REST API fallback
            return self._vector_search_via_rest_api(query, max_num_results)
        except Exception as e:
            error_msg = str(e)
            # If it's an attribute/method error, try REST API fallback
            if "method" in error_msg.lower() or "attribute" in error_msg.lower():
                try:
                    return self._vector_search_via_rest_api(query, max_num_results)
                except Exception as rest_error:
                    raise Exception(
                        f"Vector store search failed via both SDK and REST API. "
                        f"SDK error: {error_msg}. REST API error: {str(rest_error)}"
                    )
            # Provide more detailed error information for other errors
            if "vector_store" in error_msg.lower() or "vector store" in error_msg.lower():
                raise Exception(
                    f"Vector store search failed: {error_msg}. "
                    f"Ensure your vector store ID '{self.vector_store_id}' is correct and contains indexed files."
                )
            else:
                raise Exception(f"Vector store search failed: {error_msg}")
    
    def _vector_search_via_rest_api(self, query: str, max_num_results: int = 10) -> List[Dict]:
        """
        Fallback method: Search vector store using REST API directly.
        Used when SDK method is not available.
        
        Args:
            query: The search query string
            max_num_results: Maximum number of results to return
            
        Returns:
            List of search results with content, metadata, and scores
        """
        if not Config.OPENAI_API_KEY:
            raise Exception("OPENAI_API_KEY not configured for REST API fallback")
        
        url = f"https://api.openai.com/v1/vector_stores/{self.vector_store_id}/search"
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        data = {
            "query": query,
            "max_num_results": max_num_results
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            
            results = []
            if 'data' in response_data and response_data['data']:
                for item in response_data['data']:
                    result = {
                        'content': '',
                        'metadata': {},
                        'score': item.get('score'),
                        'file_id': item.get('file_id') or item.get('id')
                    }
                    
                    # Extract content
                    content = item.get('content', '')
                    if isinstance(content, list):
                        content_texts = []
                        for content_block in content:
                            if isinstance(content_block, dict):
                                content_texts.append(content_block.get('text', ''))
                            elif isinstance(content_block, str):
                                content_texts.append(content_block)
                        result['content'] = '\n\n'.join(filter(None, content_texts))
                    elif isinstance(content, str):
                        result['content'] = content
                    
                    # Extract metadata
                    result['metadata'] = item.get('metadata', {})
                    
                    # Extract filename
                    filename = (
                        item.get('filename') or
                        item.get('name') or
                        result['metadata'].get('filename')
                    )
                    
                    # If we have a file_id but no filename, try to fetch it
                    if not filename and result['file_id']:
                        try:
                            file_url = f"https://api.openai.com/v1/files/{result['file_id']}"
                            file_response = requests.get(
                                file_url,
                                headers={"Authorization": f"Bearer {Config.OPENAI_API_KEY}"},
                                timeout=10
                            )
                            file_response.raise_for_status()
                            file_data = file_response.json()
                            filename = file_data.get('filename') or file_data.get('name')
                        except Exception:
                            pass  # If file retrieval fails, continue without filename
                    
                    # Set filename in metadata
                    if filename:
                        result['metadata']['filename'] = filename
                    elif result['file_id']:
                        file_id_short = result['file_id'][:8] + '...' if len(result['file_id']) > 8 else result['file_id']
                        result['metadata']['filename'] = f"Document ({file_id_short})"
                    else:
                        result['metadata']['filename'] = "Document"
                    
                    results.append(result)
            
            return results
            
        except requests.exceptions.RequestException as e:
            raise Exception(
                f"REST API vector store search failed: {str(e)}. "
                f"Ensure your vector store ID '{self.vector_store_id}' is correct and contains indexed files."
            )
    
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
                        file_info_map = {}  # Map file_id to filename
                        
                        for annotation in annotations:
                            if hasattr(annotation, 'file_citation'):
                                file_id = annotation.file_citation.file_id
                                file_ids.append(file_id)
                                
                                # Try to fetch filename from file_id
                                if file_id and file_id not in file_info_map:
                                    try:
                                        file_info = self.client.files.retrieve(file_id)
                                        filename = getattr(file_info, 'filename', None) or getattr(file_info, 'name', None)
                                        if filename:
                                            file_info_map[file_id] = filename
                                    except Exception:
                                        # If file retrieval fails, use file_id as fallback
                                        file_info_map[file_id] = f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})"
                        
                        # Build metadata with filenames
                        metadata = {}
                        if file_ids:
                            metadata['file_ids'] = file_ids
                            # Add first filename if available (for display purposes)
                            if file_ids and file_ids[0] in file_info_map:
                                metadata['filename'] = file_info_map[file_ids[0]]
                        
                        results.append({
                            'content': text_value,
                            'metadata': metadata,
                            'file_id': file_ids[0] if file_ids else None  # Store first file_id for compatibility
                        })
            
            # Clean up assistant
            try:
                self.client.beta.assistants.delete(assistant.id)
            except Exception:
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
    
    def get_rag_response(self, user_query: str, conversation_history: List[Dict[str, str]]) -> tuple[str, List[Dict]]:
        """
        Get RAG-enhanced response by querying vector store and generating chat completion.
        Also extracts source file IDs from the Assistants API response.
        
        Args:
            user_query: The user's question or message
            conversation_history: Previous messages in the conversation
            
        Returns:
            Tuple of (response_text, sources_list) where sources_list contains:
            [{"file_id": "...", "content": "...", "metadata": {...}}]
        """
        # Step 1: Query vector store for relevant context using Assistants API
        vector_results = self.query_vector_store(user_query)
        
        # Step 2: Extract source file IDs from vector store results
        source_file_ids = []
        for result in vector_results:
            file_ids = result.get('metadata', {}).get('file_ids', [])
            source_file_ids.extend(file_ids)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_file_ids = []
        for file_id in source_file_ids:
            if file_id and file_id not in seen:
                seen.add(file_id)
                unique_file_ids.append(file_id)
        
        # Step 3: Combine context from vector store results
        context_parts = []
        for result in vector_results:
            content = result.get('content', '')
            if content:
                context_parts.append(content)
        
        context = "\n\n".join(context_parts) if context_parts else None
        
        # Step 4: Prepare messages with user query
        messages = conversation_history.copy() if conversation_history else []
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        # Step 5: Get chat completion with context
        response = self.get_chat_completion(messages, context=context, stream=False)
        
        # Step 6: Extract response text
        if hasattr(response, 'choices') and len(response.choices) > 0:
            response_text = response.choices[0].message.content
            
            # Step 7: Build sources list from file IDs with filenames
            sources = []
            for file_id in unique_file_ids:
                # Try to get filename from file_id
                filename = None
                try:
                    file_info = self.client.files.retrieve(file_id)
                    filename = getattr(file_info, 'filename', None) or getattr(file_info, 'name', None)
                except Exception:
                    pass
                
                sources.append({
                    'file_id': file_id,
                    'content': '',  # Will be filled by get_sources_for_query if needed
                    'filename': filename or f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})",
                    'metadata': {'filename': filename} if filename else {}
                })
            
            return response_text, sources
        else:
            raise Exception("No response generated from chat completion")
    
    def get_sources_for_query(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Get source documents for a query using direct vector store search.
        Formats results for clean display (hides technical details).
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of formatted source documents:
            [{"content": "...", "filename": "...", "metadata": {...}}]
        """
        try:
            # Use direct search if available
            results = self.direct_vector_search(query, max_num_results=max_results)
        except Exception:
            # Fallback to Assistants API method
            results = self.query_vector_store(query, top_k=max_results)
        
        # Format results for display (hide technical IDs, show readable content)
        formatted_results = []
        for result in results:
            # Extract filename from metadata or use fallback
            filename = (
                result.get('metadata', {}).get('filename') or
                'Document'
            )
            
            formatted = {
                'content': result.get('content', ''),
                'filename': filename,
                'metadata': {k: v for k, v in result.get('metadata', {}).items() 
                           if k != 'file_ids'},  # Hide file_ids from user-facing metadata
                # Store technical details separately for debug mode
                '_debug': {
                    'file_id': result.get('file_id'),
                    'score': result.get('score')
                }
            }
            formatted_results.append(formatted)
        
        return formatted_results

