"""OpenAI API client for vector store queries and chat completions."""
import re
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import requests
from config import Config
from logger import api_error_logger


# Constants
MAX_SEARCH_RESULTS = 50
MAX_CONTEXT_RESULTS = 20
MAX_CONTENT_LENGTH = 500
CITATION_MARKER_PATTERN = r'【\d+:\d+†[^】]*】'


class OpenAIClient:
    """Client for interacting with OpenAI vector store and chat API."""
    
    def __init__(self):
        """Initialize OpenAI client with API key."""
        Config.validate()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.vector_store_id = Config.OPENAI_VECTOR_STORE_ID
        self.model = Config.OPENAI_MODEL
        self.temperature = Config.OPENAI_TEMPERATURE
        
        # State tracking
        self.last_filtered_count = 0
        self.last_api_used = None
        self.last_error = None
        
        # File info cache to avoid repeated API calls
        self._file_info_cache: Dict[str, str] = {}
        
        # Check if Responses API is available
        try:
            self.responses_api_available = (
                hasattr(self.client, 'responses') and 
                hasattr(self.client.responses, 'create')
            )
        except Exception:
            self.responses_api_available = False
    
    def _get_filename(self, file_id: str) -> str:
        """Get filename for a file_id, with caching."""
        if not file_id:
            return "Document"
        
        # Check cache first
        if file_id in self._file_info_cache:
            return self._file_info_cache[file_id]
        
        # Try to retrieve from API
        try:
            file_info = self.client.files.retrieve(file_id)
            filename = getattr(file_info, 'filename', None) or getattr(file_info, 'name', None)
            if filename:
                self._file_info_cache[file_id] = filename
                return filename
        except Exception:
            pass
        
        # Fallback to shortened file_id
        short_id = file_id[:8] + '...' if len(file_id) > 8 else file_id
        fallback = f"Document ({short_id})"
        self._file_info_cache[file_id] = fallback
        return fallback
    
    def _filter_by_relevance(self, sources: List[Dict], min_score: float) -> Tuple[List[Dict], int]:
        """
        Filter sources by relevance score.
        
        Sources with score >= min_score or score is None are kept.
        Returns tuple of (filtered_sources, filtered_count).
        """
        filtered = []
        filtered_count = 0
        
        for source in sources:
            score = source.get('score')
            if score is None or score >= min_score:
                filtered.append(source)
            else:
                filtered_count += 1
        
        return filtered, filtered_count
    
    def _extract_citations_from_response(self, response) -> Tuple[str, List[Dict]]:
        """
        Extract text and citations from an OpenAI Responses API response.
        
        Returns tuple of (response_text, citation_list) where citation_list contains:
        [{
            'file_id': str,
            'filename': str,
            'start_index': int,
            'end_index': int
        }]
        """
        response_text = ""
        citations = []
        
        # Try simple output_text first
        if hasattr(response, 'output_text') and response.output_text:
            response_text = response.output_text
        
        # Extract from output array (standard pattern)
        if hasattr(response, 'output') and response.output:
            for output_item in response.output:
                if not hasattr(output_item, 'content') or not output_item.content:
                    continue
                
                for content in output_item.content:
                    # Extract text if not already found
                    if not response_text:
                        if hasattr(content, 'text'):
                            response_text = content.text
                        elif hasattr(content, 'type') and content.type == 'text':
                            response_text = getattr(content, 'text', '') or getattr(content, 'value', '')
                    
                    # Extract annotations
                    annotations = getattr(content, 'annotations', None)
                    if not annotations:
                        annotations = getattr(output_item, 'annotations', None)
                    
                    if annotations:
                        for annotation in annotations:
                            citation_data = self._parse_annotation(annotation)
                            if citation_data:
                                citations.append(citation_data)
        
        # Fallback to items pattern (older API versions)
        if not response_text and hasattr(response, 'items') and response.items:
            for item in response.items:
                if not (hasattr(item, 'type') and item.type == 'message'):
                    continue
                if not (hasattr(item, 'role') and item.role == 'assistant'):
                    continue
                
                if hasattr(item, 'content') and item.content:
                    for content in item.content:
                        if hasattr(content, 'type') and content.type == 'text':
                            response_text = getattr(content, 'text', '') or getattr(content, 'value', '')
                            
                            annotations = getattr(content, 'annotations', None)
                            if annotations:
                                for annotation in annotations:
                                    citation_data = self._parse_annotation(annotation)
                                    if citation_data:
                                        citations.append(citation_data)
        
        return response_text, citations
    
    def _parse_annotation(self, annotation) -> Optional[Dict]:
        """Parse a single annotation into citation data."""
        # Check for file_citation attribute (nested structure)
        if hasattr(annotation, 'file_citation'):
            citation = annotation.file_citation
            file_id = getattr(citation, 'file_id', None)
        # Check for direct file_id (flat structure)
        elif hasattr(annotation, 'file_id'):
            file_id = annotation.file_id
        else:
            return None
        
        if not file_id:
            return None
        
        return {
            'file_id': file_id,
            'filename': self._get_filename(file_id),
            'start_index': getattr(annotation, 'start_index', None),
            'end_index': getattr(annotation, 'end_index', None)
        }
    
    def _extract_file_search_results(self, response) -> Dict[str, Dict]:
        """
        Extract file_search_call results from response output.
        
        Returns dict mapping file_id to {content, score, file_id, filename}.
        When include=["file_search_call.results"] is used, the response contains
        file_search_call items with full chunk content.
        """
        results_by_file_id: Dict[str, Dict] = {}
        
        if not hasattr(response, 'output') or not response.output:
            return results_by_file_id
        
        for output_item in response.output:
            # Look for file_search_call type items
            item_type = getattr(output_item, 'type', None)
            if item_type != 'file_search_call':
                continue
            
            # Extract results from the file_search_call
            results = getattr(output_item, 'results', None)
            if not results:
                continue
            
            for result in results:
                file_id = getattr(result, 'file_id', None)
                if not file_id:
                    continue
                
                # Extract content - may be string or list of content blocks
                content = ''
                raw_content = getattr(result, 'content', None) or getattr(result, 'text', None)
                
                if isinstance(raw_content, list):
                    # Content blocks format
                    texts = []
                    for block in raw_content:
                        if hasattr(block, 'text'):
                            texts.append(block.text)
                        elif isinstance(block, dict) and 'text' in block:
                            texts.append(block['text'])
                    content = '\n\n'.join(texts)
                elif isinstance(raw_content, str):
                    content = raw_content
                
                score = getattr(result, 'score', None)
                filename = getattr(result, 'filename', None) or self._get_filename(file_id)
                
                # Store by file_id (first occurrence wins, typically highest relevance)
                if file_id not in results_by_file_id:
                    results_by_file_id[file_id] = {
                        'file_id': file_id,
                        'content': content,
                        'score': score,
                        'filename': filename
                    }
        
        return results_by_file_id
    
    def _clean_citation_markers(self, text: str, citations: List[Dict]) -> Tuple[str, Dict[str, int]]:
        """
        Replace OpenAI citation markers with clean [n] format.
        
        Returns tuple of (cleaned_text, file_id_to_number_map).
        """
        if not text:
            return text, {}
        
        # Build mapping of file_ids to citation numbers
        file_id_to_num: Dict[str, int] = {}
        num = 1
        for citation in citations:
            fid = citation.get('file_id')
            if fid and fid not in file_id_to_num:
                file_id_to_num[fid] = num
                num += 1
        
        # Find all citation markers and replace them
        def replace_marker(match):
            # Try to map marker to a file_id based on position
            # OpenAI markers are like 【4:0†source】
            # For now, just replace with sequential numbers
            return ""  # Remove markers; sources shown separately
        
        cleaned = re.sub(CITATION_MARKER_PATTERN, replace_marker, text)
        
        # Intentionally avoid collapsing whitespace to preserve model formatting
        
        return cleaned, file_id_to_num
    
    def search_vectors(
        self, 
        query: str, 
        top_k: int = MAX_SEARCH_RESULTS, 
        filters: Optional[Dict] = None,
        min_relevance_score: Optional[float] = None
    ) -> List[Dict]:
        """
        Search vector store and return formatted results.
        
        Args:
            query: The search query string
            top_k: Number of results to return
            filters: Optional metadata filters (client-side filtering)
            min_relevance_score: Optional minimum relevance threshold
            
        Returns:
            List of results: [{"filename": "...", "snippet": "...", "score": 0.89, ...}]
        """
        # Get raw results
        results = self.direct_vector_search(query, max_num_results=top_k)
        
        # Format results
        formatted = []
        for result in results:
            content = result.get('content', '')
            if not content:
                continue
            
            filename = (
                result.get('metadata', {}).get('filename') or
                result.get('filename') or
                self._get_filename(result.get('file_id'))
            )
            
            # Create preview (first 2-3 sentences)
            sentences = content.split('. ')
            preview = '. '.join(sentences[:3]) + ('...' if len(sentences) > 3 else '')
            
            formatted.append({
                'filename': filename,
                'snippet': content,
                'preview': preview,
                'score': result.get('score'),
                'file_id': result.get('file_id'),
                'content': content,
                'metadata': result.get('metadata', {})
            })
        
        # Apply metadata filters (client-side)
        if filters:
            formatted = [
                item for item in formatted
                if all(
                    not value or item.get('metadata', {}).get(key) == value
                    for key, value in filters.items()
                )
            ]
        
        # Apply relevance filtering
        if min_relevance_score is not None:
            formatted, filtered_count = self._filter_by_relevance(formatted, min_relevance_score)
            if filtered_count > 0 and formatted:
                if '_debug' not in formatted[0]:
                    formatted[0]['_debug'] = {}
                formatted[0]['_debug']['filtered_count'] = filtered_count
        
        return formatted
    
    def direct_vector_search(self, query: str, max_num_results: int = MAX_SEARCH_RESULTS) -> List[Dict]:
        """
        Search the vector store using the direct search API.
        
        Args:
            query: The search query string
            max_num_results: Maximum number of results
            
        Returns:
            List of search results with content, metadata, and scores
        """
        try:
            # Try SDK methods
            if hasattr(self.client, 'beta') and hasattr(self.client.beta, 'vector_stores'):
                if hasattr(self.client.beta.vector_stores, 'search'):
                    response = self.client.beta.vector_stores.search(
                        vector_store_id=self.vector_store_id,
                        query=query,
                        max_num_results=max_num_results
                    )
                    return self._parse_search_response(response)
            
            if hasattr(self.client, 'vector_stores') and hasattr(self.client.vector_stores, 'search'):
                response = self.client.vector_stores.search(
                    vector_store_id=self.vector_store_id,
                    query=query,
                    max_num_results=max_num_results
                )
                return self._parse_search_response(response)
            
            # Fallback to REST API
            return self._vector_search_via_rest_api(query, max_num_results)
            
        except AttributeError:
            return self._vector_search_via_rest_api(query, max_num_results)
        except Exception as e:
            error_msg = str(e)
            api_error_logger.error(f"Direct vector search failed: {error_msg}")
            
            if "method" in error_msg.lower() or "attribute" in error_msg.lower():
                return self._vector_search_via_rest_api(query, max_num_results)
            
            raise Exception(f"Vector store search failed: {error_msg}")
    
    def _parse_search_response(self, response) -> List[Dict]:
        """Parse SDK search response into standard format."""
        results = []
        
        if not hasattr(response, 'data') or not response.data:
            return results
        
        for item in response.data:
            result = {
                'content': '',
                'metadata': {},
                'score': getattr(item, 'score', None),
                'file_id': getattr(item, 'file_id', None) or getattr(item, 'id', None)
            }
            
            # Extract content
            if hasattr(item, 'content'):
                if isinstance(item.content, list):
                    texts = []
                    for block in item.content:
                        if hasattr(block, 'text'):
                            texts.append(block.text)
                        elif isinstance(block, dict) and 'text' in block:
                            texts.append(block['text'])
                    result['content'] = '\n\n'.join(texts)
                elif isinstance(item.content, str):
                    result['content'] = item.content
            
            # Extract metadata
            if hasattr(item, 'metadata') and isinstance(item.metadata, dict):
                result['metadata'] = item.metadata
            
            # Get filename
            filename = (
                getattr(item, 'filename', None) or
                getattr(item, 'name', None) or
                result['metadata'].get('filename') or
                self._get_filename(result['file_id'])
            )
            result['metadata']['filename'] = filename
            
            results.append(result)
        
        return results
    
    def _vector_search_via_rest_api(self, query: str, max_num_results: int) -> List[Dict]:
        """Fallback: Search vector store using REST API directly."""
        url = f"https://api.openai.com/v1/vector_stores/{self.vector_store_id}/search"
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        data = {"query": query, "max_num_results": max_num_results}
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            
            results = []
            for item in response_data.get('data', []):
                result = {
                    'content': '',
                    'metadata': item.get('metadata', {}),
                    'score': item.get('score'),
                    'file_id': item.get('file_id') or item.get('id')
                }
                
                # Extract content
                content = item.get('content', '')
                if isinstance(content, list):
                    texts = [
                        b.get('text', '') if isinstance(b, dict) else str(b)
                        for b in content
                    ]
                    result['content'] = '\n\n'.join(filter(None, texts))
                elif isinstance(content, str):
                    result['content'] = content
                
                # Get filename
                filename = (
                    item.get('filename') or
                    item.get('name') or
                    result['metadata'].get('filename') or
                    self._get_filename(result['file_id'])
                )
                result['metadata']['filename'] = filename
                
                results.append(result)
            
            return results
            
        except requests.exceptions.RequestException as e:
            api_error_logger.error(f"REST API search failed: {str(e)}")
            raise Exception(f"Vector store search failed: {str(e)}")
    
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
            context: Optional context string from vector store
            stream: Whether to stream the response
            
        Returns:
            Chat completion response
        """
        # Build system message with context
        system_content = "You are a helpful assistant that answers questions based on the provided context."
        if context:
            system_content += f"\n\nUse the following context to answer questions:\n\n{context}"
        
        # Prepare messages list
        prepared_messages = [{"role": "system", "content": system_content}]
        
        # Add user messages (skip any existing system message)
        for msg in messages:
            if msg.get("role") != "system":
                prepared_messages.append(msg)
        
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=prepared_messages,
                temperature=self.temperature,
                stream=stream
            )
        except Exception as e:
            error_msg = str(e)
            api_error_logger.error(f"Chat completion failed: {error_msg}")
            
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise ValueError(f"Authentication failed: {error_msg}")
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                raise Exception(f"Rate limit exceeded: {error_msg}")
            else:
                raise Exception(f"Chat completion failed: {error_msg}")
    
    def ask_about_chunks(
        self,
        question: str,
        chunks: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Ask a question about selected document chunks.
        
        Args:
            question: The user's question about the chunks
            chunks: List of chunk dicts with 'filename', 'content'/'snippet', and optionally 'score'
            
        Returns:
            Tuple of (answer_text, cited_sources) where cited_sources contains
            the chunks that were referenced in the answer
        """
        if not chunks:
            raise ValueError("No chunks provided to analyze")
        
        # Build context from chunks with numbered references
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get('filename', f'Document {i}')
            content = chunk.get('content') or chunk.get('snippet', '')
            context_parts.append(f"[Source {i}: {filename}]\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # System prompt requiring citations
        system_prompt = """You are a helpful research assistant. Answer the user's question based ONLY on the provided source documents.

Rules:
1. Only use information from the provided sources
2. Cite your sources using [Source N] format when making claims
3. If the sources don't contain enough information to answer, say so
4. Be concise but thorough

Sources:
""" + context
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            
            if not response.choices:
                raise Exception("No response generated")
            
            answer = response.choices[0].message.content
            
            # Identify which sources were cited in the response
            cited_sources = []
            for i, chunk in enumerate(chunks, 1):
                if f"[Source {i}]" in answer or f"Source {i}" in answer:
                    cited_sources.append(chunk)
            
            # If no explicit citations found, include all as potential sources
            if not cited_sources:
                cited_sources = chunks
            
            return answer, cited_sources
            
        except Exception as e:
            error_msg = str(e)
            api_error_logger.error(f"Ask about chunks failed: {error_msg}")
            raise Exception(f"Failed to analyze chunks: {error_msg}")
    
    def get_rag_response(
        self, 
        user_query: str, 
        conversation_history: List[Dict[str, str]],
        min_relevance_score: Optional[float] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Get RAG-enhanced response using Responses API with file_search.
        
        Args:
            user_query: The user's question
            conversation_history: Previous messages (not used with Responses API)
            min_relevance_score: Optional minimum relevance threshold
            
        Returns:
            Tuple of (response_text, sources_list)
        """
        self.last_filtered_count = 0
        
        # Use Responses API if available
        if self.responses_api_available:
            try:
                return self._get_rag_response_via_responses_api(user_query)
            except Exception as e:
                self.last_error = str(e)
                api_error_logger.error(f"Responses API failed: {str(e)}, using fallback")
        
        # Fallback to direct search + chat completion
        return self._get_rag_response_fallback(user_query, conversation_history, min_relevance_score)
    
    def _get_rag_response_via_responses_api(self, user_query: str) -> Tuple[str, List[Dict]]:
        """Get RAG response using Responses API with file_search tool."""
        self.last_api_used = "responses_api"
        
        tools_config = {
            "type": "file_search",
            "vector_store_ids": [self.vector_store_id],
            "max_num_results": MAX_SEARCH_RESULTS
        }
        
        response = self.client.responses.create(
            model=self.model,
            input=user_query,
            tools=[tools_config],
            include=["file_search_call.results"]  # Get full chunk content
        )
        
        # Extract text and citations
        response_text, citations = self._extract_citations_from_response(response)
        
        if not response_text:
            raise Exception("No response text generated")
        
        # Extract file_search results (full chunk content with scores)
        file_search_results = self._extract_file_search_results(response)
        
        # Clean citation markers from text
        response_text, _ = self._clean_citation_markers(response_text, citations)
        
        # Build sources from unique file_ids, matching with file_search results
        sources = []
        seen_file_ids = set()
        
        for citation in citations:
            file_id = citation.get('file_id')
            if not file_id or file_id in seen_file_ids:
                continue
            seen_file_ids.add(file_id)
            
            # Get full content from file_search results
            search_result = file_search_results.get(file_id, {})
            full_content = search_result.get('content', '')
            score = search_result.get('score')
            filename = (
                citation.get('filename') or 
                search_result.get('filename') or 
                self._get_filename(file_id)
            )
            
            sources.append({
                'file_id': file_id,
                'filename': filename,
                'content': full_content,  # Full passage from file_search results
                'snippet': full_content[:300] if full_content else '',  # For backward compat
                'score': score,  # Relevance score from file_search results
                'metadata': {
                    'filename': filename
                }
            })
        
        return response_text, sources
    
    def _get_rag_response_fallback(
        self, 
        user_query: str, 
        conversation_history: List[Dict[str, str]],
        min_relevance_score: Optional[float] = None
    ) -> Tuple[str, List[Dict]]:
        """Fallback: Use direct vector search + chat completion."""
        self.last_api_used = "fallback_rag"
        
        # Get context from vector search
        vector_results = self.direct_vector_search(user_query, max_num_results=MAX_SEARCH_RESULTS)
        
        # Filter by relevance
        if min_relevance_score is not None:
            vector_results, self.last_filtered_count = self._filter_by_relevance(
                vector_results, min_relevance_score
            )
        
        # Build context from top results
        context_parts = []
        for result in vector_results[:MAX_CONTEXT_RESULTS]:
            content = result.get('content', '')
            if content:
                truncated = content[:MAX_CONTENT_LENGTH] + "..." if len(content) > MAX_CONTENT_LENGTH else content
                context_parts.append(truncated)
        
        context = "\n\n".join(context_parts) if context_parts else None
        
        # Build messages
        messages = conversation_history.copy() if conversation_history else []
        messages.append({"role": "user", "content": user_query})
        
        # Get completion
        response = self.get_chat_completion(messages, context=context, stream=False)
        
        if not hasattr(response, 'choices') or not response.choices:
            raise Exception("No response generated")
        
        response_text = response.choices[0].message.content
        
        # Build sources list
        sources = []
        seen_file_ids = set()
        
        for result in vector_results:
            file_id = result.get('file_id')
            if file_id and file_id in seen_file_ids:
                continue
            if file_id:
                seen_file_ids.add(file_id)
            
            filename = (
                result.get('metadata', {}).get('filename') or
                result.get('filename') or
                self._get_filename(file_id)
            )
            
            sources.append({
                'file_id': file_id,
                'filename': filename,
                'content': result.get('content', ''),
                'snippet': result.get('content', ''),
                'score': result.get('score'),
                'metadata': result.get('metadata', {})
            })
        
        return response_text, sources
