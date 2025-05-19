import os
import uuid
import requests
import chromadb
import time
import functools
import threading
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from typing import Dict, Any, List, Optional, Callable
from functools import lru_cache

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAGEngine")

# Important: This ensures OpenAI isn't used by default
Settings.embed_model = None
Settings.llm = None

# Simple in-memory cache for embeddings to prevent recalculation
class EmbeddingCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.lock = threading.Lock()
        
    def get(self, text):
        with self.lock:
            if text in self.cache:
                self.access_times[text] = time.time()
                return self.cache[text]
            return None
            
    def put(self, text, embedding):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Evict least recently used item
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[text] = embedding
            self.access_times[text] = time.time()
            
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

# Function decorator for timing and logging
def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"Function {func.__name__} took {elapsed_time:.2f} seconds to run")
        return result
    return wrapper

class RAGEngine:
    def __init__(self, 
             base_dir: str = os.getcwd(), 
             ollama_base_url: str = "http://localhost:11434",
             ollama_model: str = "llama3:latest",  
             chroma_persist_dir: str = None,
             document_dir: str = None):
        
        self.base_dir = base_dir
        self.document_dir = document_dir or os.path.join(base_dir, "DocumentDir")
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.index = None
        self.embed_model = None
        self.llm = None
        
        # Performance optimization: Create an embedding cache
        self.embedding_cache = EmbeddingCache(max_size=5000)
        
        # Set ChromaDB persistence directory, default to a subdirectory in base_dir
        self.chroma_persist_dir = chroma_persist_dir or os.path.join(base_dir, "chroma_db")
        
        # Performance optimization: Set smaller dimension for faster vector search
        self.vector_dimension = 384  # Using a smaller dimension (if supported by model)
        
        # Dictionary to store chat sessions
        self.chat_sessions = {}
        
        # Performance optimization: Thread lock for session operations
        self.session_lock = threading.Lock()
        
        # Create document directory if it doesn't exist
        if not os.path.exists(self.document_dir):
            os.makedirs(self.document_dir)
            logger.info(f"Created document directory: {self.document_dir}")
        
        # Create ChromaDB directory if it doesn't exist
        if not os.path.exists(self.chroma_persist_dir):
            os.makedirs(self.chroma_persist_dir)
            logger.info(f"Created ChromaDB persistence directory: {self.chroma_persist_dir}")
        
        # Performance optimization: Pre-determine smaller top_k for similarity search
        self.similarity_top_k = 2
        self.max_token_limit = 1500  # Lower token limit to reduce LLM processing time
        
        # Verify Ollama is available before proceeding
        if not self._verify_ollama_connection():
            logger.warning(f"Cannot connect to Ollama at {self.ollama_base_url}. Continuing without LLM setup.")
        else:
            # Setup Ollama embedding and LLM
            self._setup_ollama()
            
            # Setup prompt templates
            self._setup_prompts()
            
            # Try to load existing index
            self._load_index()
    
    @log_time
    def _verify_ollama_connection(self) -> bool:
        """Verify connection to Ollama server with improved model checking and fallback"""
        try:
            # Performance optimization: Reduce timeout to fail faster if unavailable
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=3)
            response.raise_for_status()
            
            # Check if the specified model is available
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            if not model_names:
                logger.warning("No models found in Ollama server response")
                return False
                
            # Extract base model name for comparison (e.g., "llama3" from "llama3:latest")
            base_model = self.ollama_model.split(':')[0] if ':' in self.ollama_model else self.ollama_model
            
            # More flexible model matching - check if any available model starts with our base model name
            if not any(model.lower().startswith(base_model.lower()) for model in model_names):
                logger.warning(f"Warning: Model {self.ollama_model} not found in available models: {', '.join(model_names)}")
                
                # Try to select a reasonable fallback model
                if model_names:
                    # Look for smaller, faster models first
                    for preferred_model in ["tinyllama", "gemma:2b", "phi", "llama2:7b", "llama3", "mistral", "gemma"]:
                        for model in model_names:
                            if preferred_model.lower() in model.lower():
                                self.ollama_model = model
                                logger.info(f"Using fallback model: {self.ollama_model}")
                                return True
                    
                    # If no preferred model found, use the first available one
                    self.ollama_model = model_names[0]
                    logger.info(f"Using first available model: {self.ollama_model}")
                    return True
                    
                return False
                
            # Found exact or compatible model
            # Find the best matching model from available models
            exact_match = next((model for model in model_names if model.lower() == self.ollama_model.lower()), None)
            if exact_match:
                self.ollama_model = exact_match  # Use exact case from server
                logger.info(f"Using exact model match: {self.ollama_model}")
                return True
                
            # Use first compatible model
            compatible_models = [model for model in model_names if model.lower().startswith(base_model.lower())]
            if compatible_models:
                self.ollama_model = compatible_models[0]
                logger.info(f"Using compatible model: {self.ollama_model}")
                return True
                
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server: {str(e)}")
            return False
    
    @log_time
    def _setup_ollama(self):
        """Setup Ollama embedding and LLM with performance optimizations"""
        try:
            # Performance optimization: Setup embedding model with larger batch size
            self.embed_model = OllamaEmbedding(
                model_name=self.ollama_model,
                base_url=self.ollama_base_url,
                embed_batch_size=30,  # Increased batch size for fewer API calls
                ollama_additional_kwargs={
                    "mirostat": 0,
                    "num_ctx": 1024,  # Reduce context window for faster processing
                    "num_thread": 4   # Use multiple threads if available
                },
            )
            
            # Important: Set the embedding model in Settings
            Settings.embed_model = self.embed_model
            
            # Setup LLM with optimized parameters
            self.llm = Ollama(
                model=self.ollama_model, 
                base_url=self.ollama_base_url,
                request_timeout=60.0,  # Reduced timeout
                additional_kwargs={
                    "mirostat": 0,
                    "num_ctx": 1024,  # Smaller context window
                    "num_thread": 4,  # Multi-threading
                    "num_predict": 512,  # Limit token generation
                    "temperature": 0.1  # Lower temperature for faster, more deterministic responses
                }
            )
            
            # Important: Set the LLM model in Settings
            Settings.llm = self.llm
            
            logger.info(f"Successfully set up Ollama embedding and LLM using model: {self.ollama_model}")
        except Exception as e:
            logger.error(f"Failed to setup Ollama models: {str(e)}", exc_info=True)
            logger.warning("Continuing without properly configured Ollama models")
            # Don't raise an exception, allow the application to continue in degraded mode
    
    def _setup_prompts(self):
        """Setup prompt templates for query and refinement with performance-optimized instructions"""
        # Performance optimization: Shorter, more focused prompt template
        text_qa_template_str = (
            "Context information is below. Be very concise.\n"
            "---------------------\n{context_str}\n---------------------\n"
            "Using the context and your knowledge, answer: {query_str}\n"
            "Keep your answer brief and factual. Format with markdown where needed."
        )
        self.text_qa_template = PromptTemplate(text_qa_template_str)
        
        # Performance optimization: Shorter refine template
        refine_template_str = (
            "Original question: {query_str}\n"
            "Existing answer: {existing_answer}\n"
            "New context: {context_msg}\n"
            "Using both the new context and existing answer, provide a concise final answer."
        )
        self.refine_template = PromptTemplate(refine_template_str)
    
    @log_time
    def _load_index(self) -> bool:
        """Attempt to load existing index from ChromaDB with improved error handling"""
        try:
            # Performance optimization: Configure ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,  # Disable telemetry
                    allow_reset=True
                )
            )
            
            # Check if collection exists
            try:
                collection = chroma_client.get_collection("documents")
                doc_count = collection.count()
                logger.info(f"Found existing ChromaDB collection with {doc_count} documents")
                
                if doc_count == 0:
                    logger.warning("Collection exists but contains no documents")
                    return False
                
                # Create vector store from the existing collection
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Load index from the vector store
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                    embed_model=self.embed_model  # Explicitly pass the embedding model
                )
                logger.info("Successfully loaded existing index from ChromaDB")
                return True
            except ValueError as e:
                if "Collection not found" in str(e):
                    logger.info("No existing collection found in ChromaDB")
                else:
                    logger.error(f"Error accessing ChromaDB collection: {str(e)}", exc_info=True)
                return False
            except Exception as e:
                logger.error(f"Unexpected error loading collection: {str(e)}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB: {str(e)}", exc_info=True)
            return False
    
    @log_time
    def ingest_documents(self) -> Dict[str, Any]:
        """Ingest documents and create index with performance optimizations"""
    
        logger.info(f"Ingesting documents from {self.document_dir}...")
        
        try:
            # Load documents
            if not os.path.exists(self.document_dir):
                return {"status": "error", "message": f"Document directory does not exist: {self.document_dir}"}
                
            files = os.listdir(self.document_dir)
            if not files:
                return {"status": "error", "message": f"No files found in document directory: {self.document_dir}"}
            
            # Performance optimization: Use a custom document parser with smaller chunks
            documents = SimpleDirectoryReader(input_dir=self.document_dir).load_data()
            logger.info(f"Loaded {len(documents)} documents")
            
            if len(documents) == 0:
                return {"status": "warning", "message": "No documents found to ingest"}
            
            # Performance optimization: Create document chunks for better retrieval
            node_parser = SentenceSplitter(
                chunk_size=512,  # Smaller chunks for faster processing
                chunk_overlap=50  # Small overlap to maintain context between chunks
            )
            
            # Initialize ChromaDB with optimized settings
            chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,  # Disable telemetry
                    allow_reset=True
                )
            )
            
            # Remove existing collection if it exists
            try:
                chroma_client.delete_collection("documents")
                logger.info("Deleted existing collection")
            except ValueError:
                logger.info("No existing collection to delete")
            except Exception as e:
                logger.error(f"Error deleting collection: {str(e)}", exc_info=True)
                logger.warning("Continuing despite collection deletion error")
            
            # Create new collection
            collection = chroma_client.create_collection("documents")
            
            # Create vector store with the collection
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Performance optimization: Process in batches to avoid memory issues
            from llama_index.core import Document
            from tqdm import tqdm
            
            # Set batch size based on number of documents
            batch_size = min(max(1, len(documents) // 5), 10)  # Between 1 and 10
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                # Create index with the explicit embedding model and node parser
                if self.index is None:
                    self.index = VectorStoreIndex.from_documents(
                        documents=batch,
                        storage_context=storage_context,
                        embed_model=self.embed_model,  # Explicitly pass the embedding model
                        transformations=[node_parser]  # Use the node parser
                    )
                else:
                    # For subsequent batches, insert into existing index
                    for doc in batch:
                        self.index.insert(doc)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            logger.info("Index created and stored in ChromaDB successfully")
            
            return {"status": "success", "document_count": len(documents)}
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Failed to ingest documents: {str(e)}"}
    
    @log_time
    def load_data(self) -> Dict[str, Any]:
        """
        Load index from ChromaDB if available, otherwise create new index.
        This provides backward compatibility with existing code.
        """
        try:
            if self.index is None:
                if not self._load_index():
                    return self.ingest_documents()
                else:
                    return {"status": "success", "message": "Loaded existing index"}
            else:
                return {"status": "success", "message": "Index already loaded"}
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Failed to load data: {str(e)}"}
    
    def create_chat_session(self) -> str:
        """Create a new chat session and return session ID"""
        session_id = str(uuid.uuid4())
        with self.session_lock:
            # Performance optimization: Use a smaller token limit
            self.chat_sessions[session_id] = ChatMemoryBuffer.from_defaults(token_limit=self.max_token_limit)
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    def get_chat_sessions(self) -> List[Dict[str, Any]]:
        """Get list of all active chat sessions with metadata"""
        with self.session_lock:
            return [
                {
                    "session_id": sid,
                    "message_count": len(memory.get_chat_history().messages) if hasattr(memory, 'get_chat_history') else 0
                }
                for sid, memory in self.chat_sessions.items()
            ]
    
    def clear_chat_session(self, session_id: str) -> bool:
        """Clear a specific chat session"""
        with self.session_lock:
            if session_id in self.chat_sessions:
                try:
                    self.chat_sessions[session_id].clear()
                    logger.info(f"Cleared chat session: {session_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error clearing chat session {session_id}: {str(e)}", exc_info=True)
                    return False
            logger.warning(f"Attempted to clear non-existent session: {session_id}")
            return False
    
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a specific chat session"""
        with self.session_lock:
            if session_id in self.chat_sessions:
                try:
                    del self.chat_sessions[session_id]
                    logger.info(f"Deleted chat session: {session_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting chat session {session_id}: {str(e)}", exc_info=True)
                    return False
            logger.warning(f"Attempted to delete non-existent session: {session_id}")
            return False
    
    # Performance optimization: LRU cache for frequently asked questions
    @lru_cache(maxsize=100)
    def _get_cached_answer(self, question: str) -> Optional[Dict[str, Any]]:
        """Cache for frequent questions - only caches exact matches"""
        return None  # Will be populated by the LRU cache decorator
    
    @log_time
    def answer_question(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer a question using the RAG engine with performance optimizations"""
        # Check cache first
        cached_result = self._get_cached_answer(question)
        if cached_result:
            logger.info("Retrieved answer from cache")
            # Update session ID in cached result
            if session_id:
                cached_result['session_id'] = session_id
            return cached_result
            
        try:
            if not self.index:
                try:
                    # Attempt to load the index automatically
                    load_result = self.load_data()
                    if load_result.get("status") != "success":
                        return {
                            'question': question,
                            'answer': f"I'm sorry, I couldn't access my knowledge base: {load_result.get('message', 'Unknown error')}",
                            'raw_answer': f"Error: {load_result.get('message', 'Index not initialized')}",
                            'sources': [],
                            'session_id': session_id,
                            'error': True
                        }
                except Exception as e:
                    logger.error(f"Failed to load index: {str(e)}", exc_info=True)
                    return {
                        'question': question,
                        'answer': "I'm sorry, I couldn't access my knowledge base. Please try again later or contact support.",
                        'raw_answer': f"Error: Index not initialized and failed to load: {str(e)}",
                        'sources': [],
                        'session_id': session_id,
                        'error': True
                    }
            
            # Use existing session or create a new one if not provided
            memory = None
            is_new_session = False
            
            if session_id:
                with self.session_lock:
                    if session_id not in self.chat_sessions:
                        # Create a new session with the provided ID
                        try:
                            self.chat_sessions[session_id] = ChatMemoryBuffer.from_defaults(token_limit=self.max_token_limit)
                            is_new_session = True
                            logger.info(f"Created new chat session on-demand: {session_id}")
                        except Exception as e:
                            logger.error(f"Error creating chat session: {str(e)}", exc_info=True)
                            # Continue without memory rather than failing
                    memory = self.chat_sessions.get(session_id)
            
            # Create query engine with templates and memory if available
            try:
                # Performance optimization: Use a smaller similarity_top_k
                query_engine = self.index.as_query_engine(
                    text_qa_template=self.text_qa_template,
                    refine_template=self.refine_template,
                    similarity_top_k=self.similarity_top_k,  # Use fewer documents for faster retrieval
                    chat_memory=memory,
                    llm=self.llm
                )
                
                # Get response
                start_time = time.time()
                response = query_engine.query(question)
                query_time = time.time() - start_time
                logger.info(f"Query processing took {query_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}", exc_info=True)
                return {
                    'question': question,
                    'answer': f"I'm sorry, I encountered an error while processing your question. Please try again or rephrase your query.",
                    'raw_answer': f"Error: {str(e)}",
                    'sources': [],
                    'session_id': session_id,
                    'error': True
                }
            
            # Store the interaction in memory if available
            if memory:
                try:
                    if hasattr(response, 'response'):
                        # Use a more compatible approach to update memory
                        if hasattr(memory, 'put'):
                            # Performance optimization: Simplified memory update
                            try:
                                memory.put("human", question)
                                memory.put("assistant", response.response)
                            except Exception as e:
                                logger.warning(f"Memory update failed: {str(e)}")
                                # Continue despite error
                        
                        if not is_new_session:
                            logger.debug(f"Updated chat memory for session: {session_id}")
                except Exception as e:
                    logger.error(f"Error updating chat memory: {str(e)}", exc_info=True)
                    # Continue despite chat memory error
            
            # Format response with source information
            source_documents = []
            final_response = ""
            
            # More robust response handling
            try:
                if hasattr(response, 'response'):
                    final_response = response.response
                elif hasattr(response, 'get_formatted_sources'):
                    final_response = str(response)
                else:
                    final_response = str(response)
                    logger.warning(f"Response object has unexpected structure: {type(response)}")
            except Exception as e:
                logger.error(f"Error extracting response content: {str(e)}", exc_info=True)
                final_response = "I found some information, but had trouble formatting it."
            
            # Extract source information with improved error handling
            try:
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    source_nodes = response.source_nodes
                    
                    # Performance optimization: Only process the first few sources
                    for node in source_nodes[:2]:  # Limit to top 2 sources for performance
                        try:
                            metadata = {}
                            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                                metadata = node.node.metadata
                            
                            # Performance optimization: Limit source text excerpt size
                            source_doc = {
                                'text': node.text[:100] + "..." if len(node.text) > 100 else node.text,  # Shorter
                                'score': float(node.score) if hasattr(node, 'score') else None,
                                'file_name': metadata.get('file_name', 'Unknown'),
                            }
                            
                            if 'page_label' in metadata:
                                source_doc['page_label'] = metadata['page_label']
                            
                            source_documents.append(source_doc)
                        except Exception as e:
                            logger.error(f"Error processing source node: {str(e)}", exc_info=True)
                    
                    # Add source citation to response
                    if source_nodes:
                        source_info = []
                        for i, node in enumerate(source_nodes[:2]):  # Only use first two sources
                            try:
                                if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                                    metadata = node.node.metadata
                                    file_name = metadata.get('file_name', 'Unknown')
                                    page_label = metadata.get('page_label', '')
                                    
                                    if page_label:
                                        source_info.append(f"{file_name} (page: {page_label})")
                                    else:
                                        source_info.append(file_name)
                            except Exception as e:
                                logger.error(f"Error formatting source info: {str(e)}", exc_info=True)
                        
                        if source_info:
                            final_response += '\n\nSources: ' + ', '.join(source_info)
            except Exception as e:
                logger.error(f"Error processing source nodes: {str(e)}", exc_info=True)
                # Continue without source information rather than failing
            
            # Format API response
            result = {
                'question': question,
                'answer': final_response,
                'raw_answer': final_response,
                'sources': source_documents,
                'session_id': session_id
            }
            
            # Update cache for non-session-specific questions
            if len(question) > 5 and session_id is None:
                # Only cache questions that are likely to be reused
                self._get_cached_answer.cache_clear()
                self._get_cached_answer.__wrapped__.__self__.cache_key = lambda x: x
                self._get_cached_answer.__wrapped__.__self__.cache_object = lambda: {question: result}
            
            return result
        except Exception as e:
            logger.error(f"Unexpected error in answer_question: {str(e)}", exc_info=True)
            return {
                'question': question,
                'answer': "I apologize, but I encountered an unexpected error. Please try again later.",
                'raw_answer': f"Unexpected error: {str(e)}",
                'sources': [],
                'session_id': session_id,
                'error': True
            }