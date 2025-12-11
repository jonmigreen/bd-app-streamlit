"""OpenAI API client for vector store queries and chat completions."""
from typing import List, Dict, Optional
from openai import OpenAI
import requests
from config import Config
from logger import api_error_logger


class OpenAIClient:
    """Client for interacting with OpenAI vector store and chat API."""
    
    def __init__(self):
        """Initialize OpenAI client with API key."""
        Config.validate()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.vector_store_id = Config.OPENAI_VECTOR_STORE_ID
        self.model = Config.OPENAI_MODEL
        self.temperature = Config.OPENAI_TEMPERATURE
        self.last_filtered_count = 0
        # Track Responses API availability and usage
        try:
            self.responses_api_available = hasattr(self.client, 'responses') and hasattr(self.client.responses, 'create')
        except Exception:
            self.responses_api_available = False
        self.last_api_used = None  # Track which API was last used: "responses_api", "direct_vector_search", or "fallback_rag"
        self.last_error = None  # Track last error message for debugging
        self.responses_api_attempted = False  # Track if we attempted to use Responses API
        self.conversation_id = None  # Track conversation ID for stateful Responses API calls
    
    def _ensure_conversation(self):
        """Create a conversation if one doesn't exist for stateful Responses API calls."""
        if not self.conversation_id:
            try:
                if hasattr(self.client, 'conversations') and hasattr(self.client.conversations, 'create'):
                    convo = self.client.conversations.create()
                    self.conversation_id = convo.id
                else:
                    # Conversations API not available
                    return None
            except Exception as e:
                api_error_logger.error(f"Failed to create conversation: {str(e)}")
                return None
        return self.conversation_id
    
    def _filter_by_relevance(self, sources: List[Dict], min_score: float) -> tuple[List[Dict], int]:
        """
        Filter sources by relevance score.
        
        Args:
            sources: List of source dictionaries with optional 'score' field
            min_score: Minimum relevance score threshold
            
        Returns:
            Tuple of (filtered_sources, filtered_count)
            - Sources with score >= min_score are kept
            - Sources with score is None are kept (fallback cases)
            - Sources with score < min_score are filtered out
        """
        filtered = []
        filtered_count = 0
        
        for source in sources:
            score = source.get('score')
            # Keep sources with no score (fallback) or score above threshold
            if score is None or score >= min_score:
                filtered.append(source)
            else:
                filtered_count += 1
        
        return filtered, filtered_count
    
    def search_vectors(self, query: str, top_k: int = 50, filters: Optional[Dict] = None, min_relevance_score: Optional[float] = None) -> List[Dict]:
        """
        Simple helper function to search vector store and return formatted results.
        
        This is a wrapper around direct_vector_search that provides a cleaner interface
        for the UI. Returns simple list with filename, snippet, score, and metadata.
        
        Args:
            query: The search query string
            top_k: Number of results to return (default: 50)
            filters: Optional metadata filters dict (e.g., {"client": "DCC", "year": "2024"})
            min_relevance_score: Optional minimum relevance score threshold for filtering
            
        Returns:
            List of results: [{"filename": "...", "snippet": "...", "score": 0.89, "metadata": {...}}]
        """
        # Try direct search first, fallback to Responses API method if not available
        results_already_formatted = False
        try:
            results = self.direct_vector_search(query, max_num_results=top_k)
        except Exception as e:
            # If direct search API is not available, use Responses API method
            error_msg = str(e)
            api_error_logger.error(f"Direct vector search in search_vectors failed: {error_msg} | Query: {query[:100]}...")
            error_msg_lower = error_msg.lower()
            if "not available" in error_msg_lower or "method" in error_msg_lower or "attribute" in error_msg_lower:
                # Fallback to Responses API approach
                api_error_logger.error("Falling back to Responses API method")
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
                            'score': None,  # Responses API doesn't provide scores in this context
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
            formatted = filtered
        
        # Apply relevance filtering if threshold provided
        if min_relevance_score is not None:
            formatted, filtered_count = self._filter_by_relevance(formatted, min_relevance_score)
            # Store filtered count in result metadata for debug logging
            if filtered_count > 0:
                # Add debug info to first result if not present
                if formatted and '_debug' not in formatted[0]:
                    formatted[0]['_debug'] = {}
                if formatted:
                    formatted[0]['_debug']['filtered_count'] = filtered_count
        
        return formatted
    
    def direct_vector_search(self, query: str, max_num_results: int = 50) -> List[Dict]:
        """
        Directly search the vector store using the search API endpoint.
        
        This method uses the direct vector store search API, which is faster
        and more efficient than using the Assistants API. Useful for research
        mode and source retrieval.
        
        Args:
            query: The search query string
            max_num_results: Maximum number of results to return (default: 50)
            
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
            # Log API error
            api_error_logger.error(f"Direct vector search (SDK) failed: {error_msg} | Query: {query[:100]}...")
            
            # If it's an attribute/method error, try REST API fallback
            if "method" in error_msg.lower() or "attribute" in error_msg.lower():
                api_error_logger.error("Attempting REST API fallback...")
                try:
                    return self._vector_search_via_rest_api(query, max_num_results)
                except Exception as rest_error:
                    api_error_logger.error(f"REST API fallback also failed: {str(rest_error)}")
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
    
    def _vector_search_via_rest_api(self, query: str, max_num_results: int = 50) -> List[Dict]:
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
            error_msg = str(e)
            api_error_logger.error(f"REST API vector store search failed: {error_msg} | Query: {query[:100]}...")
            raise Exception(
                f"REST API vector store search failed: {error_msg}. "
                f"Ensure your vector store ID '{self.vector_store_id}' is correct and contains indexed files."
            )
    
    def query_vector_store(self, query: str, top_k: int = 50) -> List[Dict]:
        """
        Query the OpenAI vector store using Responses API.
        
        Uses Responses API with file_search tool to retrieve relevant context.
        Simpler and faster than Assistants API approach. Falls back to direct
        vector search if Responses API is not available.
        
        Args:
            query: The search query string
            top_k: Number of results to return (approximate)
            
        Returns:
            List of relevant document chunks with metadata and citations
        """
        # Check if Responses API is available
        if not hasattr(self.client, 'responses'):
            # Responses API not available, fallback to direct vector search
            self.last_api_used = "direct_vector_search"
            return self.direct_vector_search(query, max_num_results=top_k)
        
        try:
            # Use Responses API with file_search tool (following cookbook pattern)
            self.last_api_used = "responses_api"
            # Cookbook pattern: vector_store_ids in tools dict
            tools_config = {
                "type": "file_search",
                "vector_store_ids": [self.vector_store_id],
                "max_num_results": top_k
            }
            
            response = self.client.responses.create(
                model=self.model,
                input=query,  # Use input for stateless queries (cookbook pattern)
                tools=[tools_config]
            )
            
            # Extract results from response following cookbook pattern
            # Cookbook shows: response.output[1].content[0].annotations
            results = []
            file_info_map = {}
            citation_quotes = []
            text_content = ""
            
            # Try cookbook pattern first: response.output
            if hasattr(response, 'output') and response.output:
                # Find the assistant message (usually output[1] after file_search call)
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content in output_item.content:
                            # Extract text content
                            if hasattr(content, 'text'):
                                text_content = content.text
                            elif hasattr(content, 'type') and content.type == 'text':
                                if hasattr(content, 'text'):
                                    text_content = content.text
                                elif hasattr(content, 'value'):
                                    text_content = content.value
                            
                            # Extract citations from annotations (cookbook pattern)
                            annotations = None
                            if hasattr(content, 'annotations'):
                                annotations = content.annotations
                            elif hasattr(output_item, 'annotations'):
                                annotations = output_item.annotations
                            
                            if annotations:
                                for annotation in annotations:
                                    if hasattr(annotation, 'file_citation'):
                                        citation = annotation.file_citation
                                        file_id = getattr(citation, 'file_id', None)
                                        
                                        if file_id:
                                            # Extract quote text if available
                                            quote_text = getattr(citation, 'quote', None)
                                            
                                            # If quote is not directly available, try to extract from text using indices
                                            if not quote_text:
                                                start_idx = getattr(citation, 'start_index', None)
                                                end_idx = getattr(citation, 'end_index', None)
                                                if start_idx is not None and end_idx is not None and text_content:
                                                    try:
                                                        quote_text = text_content[start_idx:end_idx] if end_idx <= len(text_content) else None
                                                    except Exception:
                                                        quote_text = None
                                            
                                            # Fetch filename
                                            if file_id not in file_info_map:
                                                try:
                                                    file_info = self.client.files.retrieve(file_id)
                                                    filename = getattr(file_info, 'filename', None) or getattr(file_info, 'name', None)
                                                    if filename:
                                                        file_info_map[file_id] = filename
                                                except Exception:
                                                    file_info_map[file_id] = f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})"
                                            
                                            citation_quotes.append({
                                                'file_id': file_id,
                                                'quote': quote_text or '',
                                                'filename': file_info_map.get(file_id, f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})"),
                                                'start_index': start_idx,
                                                'end_index': end_idx,
                                                'has_quote': quote_text is not None
                                            })
            
            # Fallback to items pattern if output pattern doesn't work
            if not citation_quotes and hasattr(response, 'items') and response.items:
                for item in response.items:
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'role') and item.role == 'assistant':
                            # Extract text content
                            if hasattr(item, 'content') and item.content:
                                for content in item.content:
                                    if hasattr(content, 'type') and content.type == 'text':
                                        if hasattr(content, 'text'):
                                            text_content = content.text
                                        elif hasattr(content, 'value'):
                                            text_content = content.value
                                        
                                        # Extract citations from annotations
                                        if hasattr(content, 'annotations'):
                                            annotations = content.annotations
                                            for annotation in annotations:
                                                if hasattr(annotation, 'file_citation'):
                                                    citation = annotation.file_citation
                                                    file_id = getattr(citation, 'file_id', None)
                                                    
                                                    if file_id:
                                                        # Extract quote text if available
                                                        quote_text = getattr(citation, 'quote', None)
                                                        
                                                        # If quote is not directly available, try to extract from text using indices
                                                        if not quote_text:
                                                            start_idx = getattr(citation, 'start_index', None)
                                                            end_idx = getattr(citation, 'end_index', None)
                                                            if start_idx is not None and end_idx is not None:
                                                                try:
                                                                    quote_text = text_content[start_idx:end_idx] if end_idx <= len(text_content) else None
                                                                except Exception:
                                                                    quote_text = None
                                                        
                                                        # Fetch filename
                                                        if file_id not in file_info_map:
                                                            try:
                                                                file_info = self.client.files.retrieve(file_id)
                                                                filename = getattr(file_info, 'filename', None) or getattr(file_info, 'name', None)
                                                                if filename:
                                                                    file_info_map[file_id] = filename
                                                            except Exception:
                                                                file_info_map[file_id] = f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})"
                                                        
                                                        citation_quotes.append({
                                                            'file_id': file_id,
                                                            'quote': quote_text or '',
                                                            'filename': file_info_map.get(file_id, f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})"),
                                                            'start_index': start_idx,
                                                            'end_index': end_idx,
                                                            'has_quote': quote_text is not None
                                                        })
            
            if text_content or citation_quotes:
                file_ids = [c['file_id'] for c in citation_quotes] if citation_quotes else []
                results.append({
                    'content': text_content,
                    'metadata': {
                        'file_ids': file_ids,
                        'filename': citation_quotes[0]['filename'] if citation_quotes else None,
                        'citation_quotes': citation_quotes
                    },
                    'file_id': file_ids[0] if file_ids else None,
                    'citation_quotes': citation_quotes
                })
            
            return results[:top_k] if results else []
            
        except AttributeError:
            # Responses API not available, fallback to direct vector search
            self.last_api_used = "direct_vector_search"
            return self.direct_vector_search(query, max_num_results=top_k)
        except Exception as e:
            error_msg = str(e)
            # Log API error
            api_error_logger.error(f"Responses API (query_vector_store) failed: {error_msg} | Query: {query[:100]}...")
            
            # Check if it's a Responses API availability issue
            if "responses" in error_msg.lower() or "not available" in error_msg.lower() or "attribute" in error_msg.lower():
                # Fallback to direct vector search if Responses API not available
                api_error_logger.error("Falling back to direct vector search")
                self.last_api_used = "direct_vector_search"
                return self.direct_vector_search(query, max_num_results=top_k)
            elif "vector_store" in error_msg.lower() or "vector store" in error_msg.lower():
                raise Exception(f"Vector store query failed: {error_msg}. "
                              f"Ensure your vector store ID '{self.vector_store_id}' is correct and contains indexed files.")
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
            api_error_logger.error(f"Chat completion failed: {error_msg}")
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                api_error_logger.error("Authentication error - check OPENAI_API_KEY")
                raise ValueError(f"Authentication failed: {error_msg}. Please check your OPENAI_API_KEY.")
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                api_error_logger.error("Rate limit exceeded")
                raise Exception(f"Rate limit exceeded: {error_msg}. Please wait a moment and try again.")
            else:
                raise Exception(f"Chat completion failed: {error_msg}")
    
    def get_rag_response(self, user_query: str, conversation_history: List[Dict[str, str]], min_relevance_score: Optional[float] = None) -> tuple[str, List[Dict]]:
        """
        Get RAG-enhanced response using Responses API end-to-end.
        Uses Responses API with file_search tool to retrieve context and generate response.
        Extracts source file IDs and citation quotes from the Responses API response.
        
        Args:
            user_query: The user's question or message
            conversation_history: Previous messages in the conversation
            min_relevance_score: Optional minimum relevance score threshold (not used with Responses API, kept for compatibility)
            
        Returns:
            Tuple of (response_text, sources_list) where sources_list contains:
            [{"file_id": "...", "snippet": "...", "content": "...", "filename": "...", "metadata": {...}}]
            - snippet/content: Targeted citation quote (exact text used by model) when available
            - metadata['is_citation_quote']: True if this is a targeted citation quote
        """
        # Check if Responses API is available
        if not self.responses_api_available:
            # Fallback: Use direct vector search + Chat Completions
            self.last_api_used = "fallback_rag"
            self.responses_api_attempted = False
            return self._get_rag_response_fallback(user_query, conversation_history, min_relevance_score)
        
        try:
            # Use Responses API end-to-end with file_search tool
            self.last_api_used = "responses_api"
            self.responses_api_attempted = True
            self.last_error = None  # Clear previous error
            
            # Follow cookbook pattern: vector_store_ids in tools dict, use conversation for stateful calls
            tools_config = {
                "type": "file_search",
                "vector_store_ids": [self.vector_store_id],
                "max_num_results": 50  # Limit retrieval to avoid token limits
            }
            
            # Use conversation parameter for stateful conversations (like the example)
            conversation_id = self._ensure_conversation()
            
            if conversation_id:
                # Use stateful conversation - only pass current user message
                response = self.client.responses.create(
                    model=self.model,
                    conversation=conversation_id,
                    input=[{"role": "user", "content": user_query}],
                    tools=[tools_config]
                )
            else:
                # Fallback: use input as string for stateless queries
                response = self.client.responses.create(
                    model=self.model,
                    input=user_query,
                    tools=[tools_config]
                )
            
            # Extract response text and citations following cookbook pattern
            # Try simple output_text attribute first (like the example)
            response_text = ""
            citation_quotes = []
            file_info_map = {}
            unique_file_ids = set()
            
            # Try simple output_text attribute first (like the example)
            if hasattr(response, 'output_text') and response.output_text:
                response_text = response.output_text
            # Fallback to cookbook pattern: response.output[1].content[0].annotations
            elif hasattr(response, 'output') and response.output:
                # Find the assistant message (usually output[1] after file_search call)
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content in output_item.content:
                            # Extract text content
                            if hasattr(content, 'text'):
                                response_text = content.text
                            elif hasattr(content, 'type') and content.type == 'text':
                                if hasattr(content, 'text'):
                                    response_text = content.text
                                elif hasattr(content, 'value'):
                                    response_text = content.value
                            
                            # Extract citations from annotations (cookbook pattern)
                            annotations = None
                            if hasattr(content, 'annotations'):
                                annotations = content.annotations
                            elif hasattr(output_item, 'annotations'):
                                annotations = output_item.annotations
                            
                            if annotations:
                                for annotation in annotations:
                                    if hasattr(annotation, 'file_citation'):
                                        citation = annotation.file_citation
                                        file_id = getattr(citation, 'file_id', None)
                                        
                                        if file_id:
                                            unique_file_ids.add(file_id)
                                            
                                            # Extract quote text if available
                                            quote_text = getattr(citation, 'quote', None)
                                            
                                            # If quote is not directly available, try to extract from text using indices
                                            if not quote_text:
                                                start_idx = getattr(citation, 'start_index', None)
                                                end_idx = getattr(citation, 'end_index', None)
                                                if start_idx is not None and end_idx is not None and response_text:
                                                    try:
                                                        quote_text = response_text[start_idx:end_idx] if end_idx <= len(response_text) else None
                                                    except Exception:
                                                        quote_text = None
                                            
                                            # Fetch filename
                                            if file_id not in file_info_map:
                                                try:
                                                    file_info = self.client.files.retrieve(file_id)
                                                    filename = getattr(file_info, 'filename', None) or getattr(file_info, 'name', None)
                                                    if filename:
                                                        file_info_map[file_id] = filename
                                                except Exception:
                                                    file_info_map[file_id] = f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})"
                                            
                                            citation_quotes.append({
                                                'file_id': file_id,
                                                'quote': quote_text or '',
                                                'filename': file_info_map.get(file_id, f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})"),
                                                'start_index': start_idx,
                                                'end_index': end_idx,
                                                'has_quote': quote_text is not None
                                            })
            
            # Fallback to items pattern if output pattern doesn't work
            if not response_text and hasattr(response, 'items') and response.items:
                for item in response.items:
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'role') and item.role == 'assistant':
                            # Extract text content
                            if hasattr(item, 'content') and item.content:
                                for content in item.content:
                                    if hasattr(content, 'type') and content.type == 'text':
                                        if hasattr(content, 'text'):
                                            response_text = content.text
                                        elif hasattr(content, 'value'):
                                            response_text = content.value
                                        
                                        # Extract citations from annotations
                                        if hasattr(content, 'annotations'):
                                            annotations = content.annotations
                                            for annotation in annotations:
                                                if hasattr(annotation, 'file_citation'):
                                                    citation = annotation.file_citation
                                                    file_id = getattr(citation, 'file_id', None)
                                                    
                                                    if file_id:
                                                        unique_file_ids.add(file_id)
                                                        
                                                        # Extract quote text if available
                                                        quote_text = getattr(citation, 'quote', None)
                                                        
                                                        # If quote is not directly available, try to extract from text using indices
                                                        if not quote_text:
                                                            start_idx = getattr(citation, 'start_index', None)
                                                            end_idx = getattr(citation, 'end_index', None)
                                                            if start_idx is not None and end_idx is not None:
                                                                try:
                                                                    quote_text = response_text[start_idx:end_idx] if end_idx <= len(response_text) else None
                                                                except Exception:
                                                                    quote_text = None
                                                        
                                                        # Fetch filename
                                                        if file_id not in file_info_map:
                                                            try:
                                                                file_info = self.client.files.retrieve(file_id)
                                                                filename = getattr(file_info, 'filename', None) or getattr(file_info, 'name', None)
                                                                if filename:
                                                                    file_info_map[file_id] = filename
                                                            except Exception:
                                                                file_info_map[file_id] = f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})"
                                                        
                                                        citation_quotes.append({
                                                            'file_id': file_id,
                                                            'quote': quote_text or '',
                                                            'filename': file_info_map.get(file_id, f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})"),
                                                            'start_index': start_idx,
                                                            'end_index': end_idx,
                                                            'has_quote': quote_text is not None
                                                        })
            
            if not response_text:
                raise Exception("No response text generated from Responses API")
            
            # Build sources list from citations
            sources = []
            for file_id in unique_file_ids:
                # Get all citation quotes for this file_id
                file_citations = [c for c in citation_quotes if c['file_id'] == file_id]
                filename = file_citations[0]['filename'] if file_citations else file_info_map.get(file_id, f"Document ({file_id[:8]}...)" if len(file_id) > 8 else f"Document ({file_id})")
                
                # Use the first quote as content/snippet, or empty string if no quote
                content = file_citations[0]['quote'] if file_citations and file_citations[0].get('quote') else ''
                
                sources.append({
                    'file_id': file_id,
                    'content': content,
                    'snippet': content,  # Use quote as snippet
                    'filename': filename,
                    'metadata': {
                        'filename': filename,
                        'is_citation_quote': bool(content),
                        'citation_quotes': file_citations
                    }
                })
            
            # Set filtered count to 0 (Responses API handles filtering internally)
            self.last_filtered_count = 0
            
            return response_text, sources
            
        except Exception as e:
            error_msg = str(e)
            self.last_error = error_msg  # Store error for debugging
            
            # Log API error
            api_error_logger.error(f"Responses API (get_rag_response) failed: {error_msg} | Query: {user_query[:100]}...")
            api_error_logger.error("Falling back to direct vector search + Chat Completions")
            
            # Always fallback on any error - Responses API might exist but not work correctly
            # This handles cases where Responses API exists but fails for various reasons
            self.last_api_used = "fallback_rag"
            return self._get_rag_response_fallback(user_query, conversation_history, min_relevance_score)
    
    def _get_rag_response_fallback(self, user_query: str, conversation_history: List[Dict[str, str]], min_relevance_score: Optional[float] = None) -> tuple[str, List[Dict]]:
        """
        Fallback method when Responses API is not available.
        Uses direct vector search + Chat Completions API.
        
        Args:
            user_query: The user's question or message
            conversation_history: Previous messages in the conversation
            min_relevance_score: Optional minimum relevance score threshold for filtering
            
        Returns:
            Tuple of (response_text, sources_list)
        """
        # Track that we're using fallback method
        self.last_api_used = "fallback_rag"
        # Step 1: Query vector store for relevant context using direct search
        vector_results = self.direct_vector_search(user_query, max_num_results=50)
        
        # Filter by relevance score if threshold provided
        if min_relevance_score is not None:
            vector_results, filtered_count = self._filter_by_relevance(vector_results, min_relevance_score)
            self.last_filtered_count = filtered_count
        else:
            self.last_filtered_count = 0
        
        # Step 2: Extract source file IDs from vector store results
        source_file_ids = []
        for result in vector_results:
            file_id = result.get('file_id')
            if file_id:
                source_file_ids.append(file_id)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_file_ids = []
        for file_id in source_file_ids:
            if file_id and file_id not in seen:
                seen.add(file_id)
                unique_file_ids.append(file_id)
        
        # Step 3: Combine context from vector store results (limit to avoid token issues)
        # Use only citation quotes if available, otherwise truncate content
        context_parts = []
        for result in vector_results[:20]:  # Limit to top 20 to avoid token limits
            # Prefer citation quotes, fallback to content
            citation_quotes = result.get('citation_quotes', [])
            if citation_quotes:
                for quote in citation_quotes[:3]:  # Limit quotes per result
                    if quote.get('quote'):
                        context_parts.append(quote['quote'])
            else:
                content = result.get('content', '')
                if content:
                    # Truncate to first 500 chars to avoid token bloat
                    truncated = content[:500] + "..." if len(content) > 500 else content
                    context_parts.append(truncated)
        
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
    
    def get_sources_for_query(self, query: str, max_results: int = 50, min_relevance_score: Optional[float] = None) -> List[Dict]:
        """
        Get source documents for a query using direct vector store search.
        Formats results for clean display (hides technical details).
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return
            min_relevance_score: Optional minimum relevance score threshold for filtering
            
        Returns:
            List of formatted source documents:
            [{"content": "...", "filename": "...", "metadata": {...}}]
        """
        try:
            # Use direct search if available
            results = self.direct_vector_search(query, max_num_results=max_results)
        except Exception as e:
            # Fallback to Responses API method
            error_msg = str(e)
            api_error_logger.error(f"Direct vector search failed in get_sources_for_query, falling back to Responses API: {error_msg} | Query: {query[:100]}...")
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
        
        # Apply relevance filtering if threshold provided
        if min_relevance_score is not None:
            formatted_results, filtered_count = self._filter_by_relevance(formatted_results, min_relevance_score)
            # Store filtered count in debug info
            if filtered_count > 0 and formatted_results:
                if '_debug' not in formatted_results[0]:
                    formatted_results[0]['_debug'] = {}
                formatted_results[0]['_debug']['filtered_count'] = filtered_count
        
        return formatted_results

