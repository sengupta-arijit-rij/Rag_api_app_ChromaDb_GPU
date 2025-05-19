"""Flask API for RAG Application with conversational capabilities
Optimized for performance
"""
from flask import Flask, request, jsonify, session
from rag_engine import RAGEngine
import os
import time
import uuid
from flask_cors import CORS
import logging
import functools
import threading
import requests
from cachelib import SimpleCache  # Using cachelib instead of werkzeug.contrib.cache

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_API")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())
CORS(app)  # Enable CORS for all routes

# Load configuration
DEBUG = os.environ.get("DEBUG", "True").lower() == "true"
PORT = int(os.environ.get("PORT", 5001))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(os.getcwd(), "chroma_db"))
DOCUMENT_DIR = os.environ.get("DOCUMENT_DIR", os.path.join(os.getcwd(), "DocumentDir"))

# In-memory cache for API responses
# Performance optimization: Add caching for frequent requests
cache = SimpleCache(threshold=500, default_timeout=300)  # 5 minutes cache timeout

# Thread lock for thread safety
init_lock = threading.Lock()
rag_engine = None

# Function decorator for timing and logging
def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"Endpoint {func.__name__} took {elapsed_time:.2f} seconds to execute")
        return result
    return wrapper

# Performance optimization: Lazy initialization of RAG engine
def get_rag_engine():
    global rag_engine
    if rag_engine is None:
        with init_lock:
            if rag_engine is None:  # Double-check locking pattern
                try:
                    rag_engine = RAGEngine(
                        base_dir=os.getcwd(), 
                        ollama_base_url=OLLAMA_BASE_URL,
                        ollama_model=OLLAMA_MODEL,
                        chroma_persist_dir=CHROMA_PERSIST_DIR,
                        document_dir=DOCUMENT_DIR
                    )
                    logger.info("RAG Engine initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize RAG Engine: {str(e)}")
                    # Create engine in degraded state
                    rag_engine = RAGEngine(
                        base_dir=os.getcwd(), 
                        ollama_base_url=OLLAMA_BASE_URL,
                        chroma_persist_dir=CHROMA_PERSIST_DIR
                    )
    return rag_engine

# Performance optimization: Cached Ollama status check
def check_ollama_status():
    # Check cache first
    status = cache.get('ollama_status')
    if status is not None:
        return status
    
    # If not in cache, check the status
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
        if response.status_code == 200:
            status = {"status": "connected", "models": response.json().get("models", [])}
        else:
            status = {"status": f"error: status code {response.status_code}"}
    except Exception as e:
        status = {"status": f"error: {str(e)}"}
    
    # Cache the result for 30 seconds
    cache.set('ollama_status', status, timeout=30)
    return status

# Performance optimization: Request validation helper
def validate_json_request(required_fields=None):
    """Validate JSON request and required fields"""
    data = request.get_json()
    if not data:
        return None, ({"error": "Request must contain valid JSON"}, 400)
    
    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return None, ({"error": f"Missing required fields: {', '.join(missing_fields)}"}, 400)
    
    return data, None

# Performance optimization: Session management helper
def get_or_create_session_id(data):
    """Get existing session ID or create new one"""
    session_id = data.get('session_id')
    if not session_id:
        # Try to get from Flask session
        flask_session_id = session.get('session_id')
        if flask_session_id:
            session_id = flask_session_id
        else:
            # Create new session
            try:
                session_id = get_rag_engine().create_chat_session()
                session['session_id'] = session_id
                logger.info(f"Created new session: {session_id}")
            except Exception as e:
                logger.error(f"Failed to create chat session: {str(e)}")
                return None, ({"error": f"Failed to create chat session: {str(e)}"}, 500)
    
    return session_id, None

@app.route('/health', methods=['GET'])
@log_time
def health_check():
    """Health check endpoint with improved caching"""
    # Check cache first
    cached_response = cache.get('health_check')
    if cached_response and time.time() - cached_response.get('timestamp', 0) < 5:  # 5 sec cache
        return jsonify(cached_response)
    
    # If not in cache or expired, check health
    try:
        engine = get_rag_engine()
        ollama_status = check_ollama_status()
        
        # Performance optimization: Compute only necessary information
        doc_dir_files = []
        if os.path.exists(engine.document_dir):
            doc_dir_files = os.listdir(engine.document_dir)[:10]  # Limit to 10 files for performance
        
        response = {
            "status": "healthy", 
            "timestamp": time.time(),
            "ollama_url": OLLAMA_BASE_URL,
            "ollama_status": ollama_status.get("status"),
            "document_dir": engine.document_dir,
            "chroma_persist_dir": engine.chroma_persist_dir,
            "index_loaded": engine.index is not None,
            "active_sessions": len(engine.chat_sessions),
            "files_in_document_dir": doc_dir_files
        }
        
        # Cache the response
        cache.set('health_check', response, timeout=30)  # 30 seconds cache
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/api/ask', methods=['POST'])
@log_time
def ask_question():
    """API endpoint to ask a question and get RAG-enhanced response"""
    try:
        # Initialize RAG engine if needed
        engine = get_rag_engine()
        
        # Check if index is loaded
        if engine.index is None:
            try:
                result = engine.load_data()
                if result.get("status") == "error":
                    return jsonify(result), 500
                logger.info("Loaded data for question answering")
            except Exception as e:
                logger.error(f"Failed to initialize RAG engine: {str(e)}")
                return jsonify({"error": f"Failed to initialize RAG engine: {str(e)}"}), 500
        
        # Validate request
        data, error = validate_json_request(['question'])
        if error:
            return error
            
        question = data['question']
        if not question or not isinstance(question, str):
            return jsonify({"error": "Question must be a non-empty string"}), 400
        
        # Get or create session ID
        session_id, error = get_or_create_session_id(data)
        if error:
            return error
        
        # Process question with conversation context
        try:
            # Performance optimization: Cache for identical questions
            cache_key = f"question_{hash(question)}_{session_id}"
            cached_result = cache.get(cache_key)
            if cached_result:
                return jsonify(cached_result)
                
            result = engine.answer_question(question, session_id)
            
            # Cache the result for short-lived questions
            if len(question) < 100:  # Only cache short questions
                cache.set(cache_key, result, timeout=60)  # 1 minute cache
                
            return jsonify(result)
        except ValueError as e:
            logger.error(f"Value error processing question: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Failed to process question: {str(e)}")
            return jsonify({"error": f"Failed to process question: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/ingest', methods=['POST'])
@log_time
def ingest_data():
    """Explicitly trigger document ingestion and indexing"""
    try:
        engine = get_rag_engine()
        
        # Check if Ollama is accessible
        ollama_status = check_ollama_status()
        if ollama_status.get("status") != "connected":
            return jsonify({
                "status": "error", 
                "message": f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Please ensure Ollama is running."
            }), 500
            
        # Check if document directory exists and contains files
        if not os.path.exists(engine.document_dir):
            return jsonify({
                "status": "error", 
                "message": f"Document directory does not exist: {engine.document_dir}"
            }), 400
            
        files = os.listdir(engine.document_dir)
        if not files:
            return jsonify({
                "status": "error", 
                "message": f"No files found in document directory: {engine.document_dir}"
            }), 400
        
        # Ingest documents
        result = engine.ingest_documents()
        
        # Clear caches after ingestion
        cache.clear()
        
        if result.get("status") == "error":
            return jsonify(result), 500
            
        if result.get("status") == "warning":
            return jsonify(result), 200
            
        return jsonify({
            "status": "success", 
            "message": "Documents ingested successfully",
            "details": result
        })
    except Exception as e:
        logger.error(f"Failed to ingest documents: {str(e)}")
        return jsonify({"error": f"Failed to ingest documents: {str(e)}"}), 500

@app.route('/api/reload', methods=['POST'])
@log_time
def reload_data():
    """Reload index from ChromaDB without reingesting documents"""
    try:
        engine = get_rag_engine()
        result = engine.load_data()
        
        # Clear caches after reload
        cache.clear()
        
        if result.get("status") == "error":
            return jsonify(result), 500
            
        return jsonify({
            "status": "success", 
            "message": "Data loaded successfully",
            "details": result
        })
    except Exception as e:
        logger.error(f"Failed to reload data: {str(e)}")
        return jsonify({"error": f"Failed to reload data: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
@log_time
def chat():
    """
    Conversational chat interface for direct interaction.
    """
    try:
        engine = get_rag_engine()
        
        # Check if index is loaded
        if engine.index is None:
            try:
                result = engine.load_data()
                if result.get("status") == "error":
                    return jsonify({"error": result.get("message", "Failed to initialize RAG engine")}), 500
            except Exception as e:
                logger.error(f"Failed to initialize RAG engine: {str(e)}")
                return jsonify({"error": f"Failed to initialize RAG engine: {str(e)}"}), 500
        
        # Validate request
        data, error = validate_json_request(['message'])
        if error:
            return error
            
        message = data['message']
        if not message or not isinstance(message, str):
            return jsonify({"error": "Message must be a non-empty string"}), 400
        
        # Get or create session ID
        session_id, error = get_or_create_session_id(data)
        if error:
            return error
        
        # Exit command (optional)
        if message.lower() == 'exit':
            return jsonify({
                "answer": "Thank you for reaching out. Feel free to visit us again. Bye!",
                "session_id": session_id
            })
        
        # Process question with conversation context
        try:
            result = engine.answer_question(message, session_id)
            return jsonify(result)
        except ValueError as e:
            logger.error(f"Value error processing message: {str(e)}")
            return jsonify({
                "error": str(e),
                "session_id": session_id
            }), 400
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            return jsonify({
                "error": "Sorry, I encountered an error while processing your message. Please try again later.",
                "session_id": session_id
            }), 500
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "Sorry, I encountered an unexpected error. Please try again later."
        }), 500

@app.route('/api/session', methods=['POST'])
@log_time
def manage_session():
    """
    Create, get, clear, or delete chat sessions.
    """
    try:
        # Validate request
        data, error = validate_json_request(['action'])
        if error:
            return error
            
        action = data['action'].lower()
        engine = get_rag_engine()
        
        if action == 'create':
            # Create new session
            try:
                session_id = engine.create_chat_session()
                return jsonify({"status": "success", "session_id": session_id})
            except Exception as e:
                logger.error(f"Failed to create chat session: {str(e)}")
                return jsonify({"error": f"Failed to create chat session: {str(e)}"}), 500
        
        elif action == 'get':
            # Get all active sessions
            try:
                sessions = engine.get_chat_sessions()
                return jsonify({"status": "success", "sessions": sessions})
            except Exception as e:
                logger.error(f"Failed to get chat sessions: {str(e)}")
                return jsonify({"error": f"Failed to get chat sessions: {str(e)}"}), 500
        
        elif action in ['clear', 'delete']:
            # Check for session_id
            session_id = data.get('session_id')
            if not session_id:
                return jsonify({"error": f"Missing 'session_id' field for {action} action"}), 400
            
            # Clear or delete session
            try:
                if action == 'clear':
                    success = engine.clear_chat_session(session_id)
                else:  # delete
                    success = engine.delete_chat_session(session_id)
                
                if success:
                    return jsonify({"status": "success", "message": f"Session {action}ed successfully"})
                else:
                    return jsonify({"error": f"Session {session_id} not found or could not be {action}ed"}), 404
            except Exception as e:
                logger.error(f"Failed to {action} chat session {session_id}: {str(e)}")
                return jsonify({"error": f"Failed to {action} chat session: {str(e)}"}), 500
        
        else:
            return jsonify({"error": f"Invalid action: {action}. Must be one of: create, get, clear, delete"}), 400
    except Exception as e:
        logger.error(f"Unexpected error in manage_session: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Initialize engine in a separate thread to avoid blocking startup
    threading.Thread(target=get_rag_engine, daemon=True).start()
    
    # Start the Flask app
    app.run(debug=DEBUG, host="0.0.0.0", port=PORT)
