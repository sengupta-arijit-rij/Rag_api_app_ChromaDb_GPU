"""Flask API for RAG Application with conversational capabilities
GPU-accelerated version using LM Studio, LlamaIndex, and Llama3
"""
from flask import Flask, request, jsonify, session
from rag_engine import RAGEngine
import os
import time
import uuid
from flask_cors import CORS
import logging

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
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama3")
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(os.getcwd(), "chroma_db"))
DOCUMENT_DIR = os.environ.get("DOCUMENT_DIR", os.path.join(os.getcwd(), "DocumentDir"))
DEVICE = os.environ.get("DEVICE", "cuda")  # Use "cuda" for GPU, "cpu" for CPU


# Initialize RAG engine with ChromaDB persistence
try:
    rag_engine = RAGEngine(
        base_dir=os.getcwd(), 
        lm_studio_url=LM_STUDIO_URL,
        model_name=LLAMA_MODEL,
        chroma_persist_dir=CHROMA_PERSIST_DIR,
        document_dir=DOCUMENT_DIR,
        device=DEVICE
    )
    logger.info("RAG Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Engine: {str(e)}")
    # Still create the engine, but it might be in a degraded state
    rag_engine = RAGEngine(
        base_dir=os.getcwd(), 
        lm_studio_url=LM_STUDIO_URL,
        chroma_persist_dir=CHROMA_PERSIST_DIR
    )

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        lm_status = "connected"
        try:
            import requests
            response = requests.get(f"{LM_STUDIO_URL}/models", timeout=2)
            if response.status_code != 200:
                lm_status = f"error: status code {response.status_code}"
        except Exception as e:
            lm_status = f"error: {str(e)}"
        
        # Check GPU status
        gpu_status = "not available"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_status = f"available - {torch.cuda.get_device_name(0)}"
                gpu_memory = f"{torch.cuda.memory_allocated(0)/1024**3:.2f}GB / {torch.cuda.memory_reserved(0)/1024**3:.2f}GB"
            else:
                gpu_status = "not available - CUDA not found"
        except ImportError:
            gpu_status = "not available - PyTorch not installed"
        except Exception as e:
            gpu_status = f"error: {str(e)}"
        
        return jsonify({
            "status": "healthy", 
            "timestamp": time.time(),
            "lm_studio_url": LM_STUDIO_URL,
            "lm_status": lm_status,
            "gpu_status": gpu_status,
            "device": rag_engine.device,
            "model": rag_engine.model_name,
            "document_dir": rag_engine.document_dir,
            "chroma_persist_dir": rag_engine.chroma_persist_dir,
            "index_loaded": rag_engine.index is not None,
            "active_sessions": len(rag_engine.chat_sessions),
            "files_in_document_dir": os.listdir(rag_engine.document_dir) if os.path.exists(rag_engine.document_dir) else []
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint to ask a question and get RAG-enhanced response"""
    try:
        # Check if RAG engine is initialized
        if rag_engine.index is None:
            try:
                result = rag_engine.load_data()
                if result.get("status") == "error":
                    return jsonify(result), 500
                logger.info("Loaded data for question answering")
            except Exception as e:
                logger.error(f"Failed to initialize RAG engine: {str(e)}")
                return jsonify({"error": f"Failed to initialize RAG engine: {str(e)}"}), 500
        
        # Get question from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must contain valid JSON"}), 400
            
        if 'question' not in data:
            return jsonify({"error": "Missing 'question' field in request"}), 400
        
        question = data['question']
        if not question or not isinstance(question, str):
            return jsonify({"error": "Question must be a non-empty string"}), 400
        
        # Get or create session ID
        session_id = data.get('session_id')
        if not session_id:
            # Try to get from Flask session first
            flask_session_id = session.get('session_id')
            if flask_session_id:
                session_id = flask_session_id
            else:
                # Create new session if none exists
                try:
                    session_id = rag_engine.create_chat_session()
                    session['session_id'] = session_id
                    logger.info(f"Created new session: {session_id}")
                except Exception as e:
                    logger.error(f"Failed to create chat session: {str(e)}")
                    return jsonify({"error": f"Failed to create chat session: {str(e)}"}), 500
        
        # Process question with conversation context
        try:
            result = rag_engine.answer_question(question, session_id)
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
def ingest_data():
    """Explicitly trigger document ingestion and indexing"""
    try:
        # Check if LM Studio is accessible
        try:
            import requests
            response = requests.get(f"{LM_STUDIO_URL}/models", timeout=2)
            if response.status_code != 200:
                return jsonify({
                    "status": "error", 
                    "message": f"Cannot connect to LM Studio at {LM_STUDIO_URL}. Please ensure LM Studio is running."
                }), 500
        except Exception as e:
            return jsonify({
                "status": "error", 
                "message": f"Cannot connect to LM Studio: {str(e)}. Please ensure LM Studio is running."
            }), 500
            
        # Check if document directory exists and contains files
        if not os.path.exists(rag_engine.document_dir):
            return jsonify({
                "status": "error", 
                "message": f"Document directory does not exist: {rag_engine.document_dir}"
            }), 400
            
        files = os.listdir(rag_engine.document_dir)
        if not files:
            return jsonify({
                "status": "error", 
                "message": f"No files found in document directory: {rag_engine.document_dir}"
            }), 400
        
        # Ingest documents
        result = rag_engine.ingest_documents()
        
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
def reload_data():
    """Reload index from ChromaDB without reingesting documents"""
    try:
        result = rag_engine.load_data()
        
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
def chat():
    """
    Conversational chat interface for direct interaction.
    
    Expected JSON input:
    {
        "message": "Your question here",
        "session_id": "optional-session-id" 
    }
    
    Returns a JSON response with the answer and session information.
    """
    try:
        # Check if RAG engine is initialized
        if rag_engine.index is None:
            try:
                result = rag_engine.load_data()
                if result.get("status") == "error":
                    return jsonify({"error": result.get("message", "Failed to initialize RAG engine")}), 500
            except Exception as e:
                logger.error(f"Failed to initialize RAG engine: {str(e)}")
                return jsonify({"error": f"Failed to initialize RAG engine: {str(e)}"}), 500
        
        # Get message from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must contain valid JSON"}), 400
            
        if 'message' not in data:
            return jsonify({"error": "Missing 'message' field in request"}), 400
        
        message = data['message']
        if not message or not isinstance(message, str):
            return jsonify({"error": "Message must be a non-empty string"}), 400
        
        # Get or create session ID
        session_id = data.get('session_id')
        if not session_id:
            # Try to get from Flask session first
            flask_session_id = session.get('session_id')
            if flask_session_id:
                session_id = flask_session_id
            else:
                # Create new session if none exists
                try:
                    session_id = rag_engine.create_chat_session()
                    session['session_id'] = session_id
                except Exception as e:
                    logger.error(f"Failed to create chat session: {str(e)}")
                    return jsonify({"error": f"Failed to create chat session: {str(e)}"}), 500
        
        # Exit command (optional)
        if message.lower() == 'exit':
            return jsonify({
                "answer": "Thank you for reaching out. Feel free to visit us again. Bye!",
                "session_id": session_id
            })
        
        # Process question with conversation context
        try:
            result = rag_engine.answer_question(message, session_id)
            # Return the result as JSON
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
def manage_session():
    """
    Create, get, clear, or delete chat sessions.
    
    Expected JSON input:
    {
        "action": "create"|"get"|"clear"|"delete",
        "session_id": "optional-session-id-for-clear-or-delete" 
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must contain valid JSON"}), 400
            
        if 'action' not in data:
            return jsonify({"error": "Missing 'action' field in request"}), 400
        
        action = data['action'].lower()
        
        if action == 'create':
            # Create new session
            try:
                session_id = rag_engine.create_chat_session()
                return jsonify({"status": "success", "session_id": session_id})
            except Exception as e:
                logger.error(f"Failed to create chat session: {str(e)}")
                return jsonify({"error": f"Failed to create chat session: {str(e)}"}), 500
        
        elif action == 'get':
            # Get all active sessions
            try:
                sessions = rag_engine.get_chat_sessions()
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
                    success = rag_engine.clear_chat_session(session_id)
                else:  # delete
                    success = rag_engine.delete_chat_session(session_id)
                
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
    # Try to load existing index on startup
    try:
        load_result = rag_engine.load_data()
        logger.info(f"Index loading status: {load_result}")
    except Exception as e:
        logger.error(f"Warning: Failed to load index on startup: {e}")
        logger.info("You can load data later using the /api/reload endpoint")
        logger.info("Or ingest documents using the /api/ingest endpoint")
    
    # Start the Flask app
    app.run(debug=DEBUG, host="0.0.0.0", port=PORT)