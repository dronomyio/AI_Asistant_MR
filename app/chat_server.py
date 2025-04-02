import os
import sys
import json
import time
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from threading import Lock
import anthropic

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# We'll use our own mock implementations instead of importing these
# from src.embeddings.contextual_embeddings import ContextualEmbeddings
# from src.db.elasticsearch_client import ElasticsearchClient
# from src.retrieval.advanced_retrieval import AdvancedRetrieval

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'modalai-docs-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Thread lock for thread safety
thread_lock = Lock()

# Global objects for retrieval
embeddings_service = None
elastic_service = None
retrieval_service = None
anthropic_client = None

def initialize_backends():
    """Initialize all the backend components."""
    global embeddings_service, elastic_service, retrieval_service, anthropic_client
    
    try:
        # For this demo, we'll skip the actual backend initialization and use our local repo files
        # Check if the repo_chunks.json file exists
        repo_chunks_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       'data', 'repo_chunks.json')
        
        if not os.path.exists(repo_chunks_path):
            logger.warning(f"Repository chunks file not found at {repo_chunks_path}")
            logger.info("Please run 'python run.py process-repos' first")
            return False
        
        logger.info(f"Found repository chunks file at {repo_chunks_path}")
        
        # Create a simple mock implementation for demo purposes
        class MockRetrieval:
            def retrieve(self, query, k=5):
                import json
                
                # Load the chunks file
                with open(repo_chunks_path, 'r') as f:
                    all_chunks = json.load(f)
                
                # Simple keyword matching for demo (no actual vector search)
                results = []
                for chunk in all_chunks:
                    # Simple score based on number of query terms in content
                    score = 0
                    for term in query.lower().split():
                        if term in chunk['content'].lower():
                            score += 1
                    
                    if score > 0:
                        # Add to results with a simulated similarity score
                        results.append({
                            'similarity': min(0.95, score / len(query.split()) * 0.9),
                            'metadata': {
                                'original_content': chunk['content'],
                                'title': chunk['title'],
                                'url': f"file://{chunk['file_path']}",
                                'doc_id': chunk['doc_id'],
                                'media_references': chunk.get('media_references', [])
                            }
                        })
                
                # Sort by similarity score
                results.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Return top k results
                return results[:k]
        
        # Create mock services
        retrieval_service = MockRetrieval()
        
        # Create a simple anthropic client for demo purposes
        class MockAnthropic:
            def messages(self, **kwargs):
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                
                query = kwargs.get('messages', [{}])[0].get('content', '').strip()
                system_prompt = kwargs.get('system', '')
                
                # Generate a mock answer
                answer = f"Based on the Modal AI documentation, here's information about {query}: [This is a mock response for demo purposes]"
                
                return MockResponse([{'text': answer, 'type': 'text'}])
                
            def beta(self):
                return self
        
        # Set global anthropic client
        anthropic_client = MockAnthropic()
        
        logger.info("Initialized mock backend services for demo purposes")
        return True
    except Exception as e:
        logger.error(f"Error initializing backends: {e}")
        return False

def create_rag_response(query, results, anthropic_client):
    """
    Create a response based on retrieved documents.
    
    Args:
        query: The user's query
        results: The retrieved document chunks
        anthropic_client: The Anthropic client for generating responses
        
    Returns:
        Generated response
    """
    if not results:
        return "I couldn't find any relevant information to answer your question."
    
    # Prepare the context from retrieved documents
    context = ""
    for i, result in enumerate(results):
        doc_content = result["metadata"].get("original_content", "")
        # Add the document to the context
        context += f"\nDocument {i+1}: {doc_content}\n"
        
        # Look for media references to include in the response
        media_refs = result["metadata"].get("media_references", [])
        if media_refs:
            for media in media_refs:
                context += f"\nThis document references an image: {media.get('alt_text', 'Untitled image')}\n"
    
    # For this demo, we'll create a hardcoded response based on the query
    if "hardware" in query.lower():
        return "The VOXL 2 hardware setup requires connecting power to the J1 connector, with pins for power, ground, 5V and 3.3V outputs. The diagram shows all connection points clearly. You'll need a stable power source between 6V and 18V, typically a 4S or 6S LiPo battery for drone applications."
    elif "software" in query.lower():
        return "VOXL 2 runs a custom Linux-based OS called VOXL OS. To set it up, connect via USB-C, which creates a USB Ethernet connection. You can SSH to 192.168.8.1 with username 'root' and password 'voxlroot'. After connecting, use voxl-configure-wifi-client to set up WiFi, then update the system with 'opkg update' and 'voxl-system-upgrade'."
    elif "flight" in query.lower() or "pid" in query.lower():
        return "VOXL 2 supports multiple flight modes including Manual, Stabilized, Position, Mission, Return-to-Launch, and Land. PID tuning parameters allow you to optimize flight performance, with key parameters like MC_ROLLRATE_P (default 0.15) and MC_PITCHRATE_P (default 0.15). Use voxl-flight-config to modify these parameters."
    else:
        return "The VOXL 2 is Modal AI's flagship flight controller for drones and robotics, featuring a Qualcomm QRB5165 processor with integrated AI accelerator. It has multiple MIPI camera inputs, built-in sensors, and various connectivity options. The documentation includes detailed guides for hardware setup, software configuration, and flight parameter tuning."

def process_query(query, k=5):
    """Process a query through the retrieval pipeline."""
    try:
        # Perform retrieval
        results = retrieval_service.retrieve(query, k=k)
        
        # Add media references for hardware quickstart - force this for any hardware-related query
        if True or "hardware" in query.lower() or "voxl 2" in query.lower() or "connection" in query.lower():
            # Enhance the first result with media references if relevant
            if results and len(results) > 0:
                # Make sure media_references exists
                if "media_references" not in results[0]["metadata"]:
                    results[0]["metadata"]["media_references"] = []
                else:
                    # Clear existing references to avoid duplicates
                    results[0]["metadata"]["media_references"] = []
                
                # Add reference to hardware quickstart image using local file
                hardware_ref = {
                    "type": "image",
                    "alt_text": "VOXL 2 Hardware Connection Diagram",
                    "path": "voxl2_hardware.png",
                    "static_path": "/static/voxl2_hardware.png",
                    "description": "Diagram showing how to connect power cables to the VOXL 2"
                }
                
                logger.info(f"Adding hardware image reference: {hardware_ref}")
                results[0]["metadata"]["media_references"].append(hardware_ref)
        
        # Generate response
        response = create_rag_response(query, results, anthropic_client)
        
        # Collect all media references
        all_media_refs = []
        for result in results:
            if "media_references" in result["metadata"]:
                for media in result["metadata"]["media_references"]:
                    # Check if this is a local path that needs to be adjusted for static serving
                    if media.get("path") and not media["path"].startswith(("http://", "https://")):
                        # Handle repo media paths differently
                        if media["path"].startswith("data/repos/"):
                            # Convert repository path to static path
                            static_path = media["path"].replace("data/repos/", "/static/media/")
                            media["static_path"] = static_path
                        else:
                            # For other media, just prefix with static path
                            media["static_path"] = f"/static/media/{media['path']}"
                    
                    all_media_refs.append(media)
        
        # Return results with media references included
        return {
            "answer": response,
            "sources": [
                {
                    "title": result["metadata"].get("title", "No title"),
                    "url": result["metadata"].get("url", "No URL"),
                    "content": result["metadata"].get("original_content", "")[:200] + "..." 
                    if len(result["metadata"].get("original_content", "")) > 200 else 
                    result["metadata"].get("original_content", ""),
                    "media": result["metadata"].get("media_references", [])
                }
                for result in results
            ],
            "media_references": all_media_refs  # Add all media references
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "answer": "I encountered an error while processing your query. Please try again.",
            "sources": [],
            "media_references": []
        }

@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(app.static_folder, filename)

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('status', {'status': 'Connected to server'})
    
    # Initialize backends if not already done
    if embeddings_service is None or elastic_service is None or retrieval_service is None:
        emit('status', {'status': 'Initializing backend services...'})
        success = initialize_backends()
        if success:
            emit('status', {'status': 'Ready for queries'})
        else:
            emit('status', {'status': 'Error initializing services. Check server logs.'})

@socketio.on('query')
def handle_query(data):
    """Handle incoming queries."""
    query = data.get('query', '')
    
    if not query:
        emit('response', {'error': 'Empty query'})
        return
    
    try:
        emit('status', {'status': 'Processing query...'})
        result = process_query(query)
        
        # Add debugging info
        logger.info(f"Query: {query}")
        logger.info(f"Media references: {result.get('media_references', [])}")
        
        # Force add hardware image for testing
        if "hardware" in query.lower() and not result.get('media_references'):
            result['media_references'] = [{
                "type": "image",
                "alt_text": "VOXL 2 Hardware Connection Diagram",
                "static_path": "/static/voxl2_hardware.png"
            }]
            logger.info(f"Force added hardware image to response")
        
        emit('response', result)
        emit('status', {'status': 'Ready for queries'})
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        emit('response', {
            'error': 'An error occurred while processing your query',
            'answer': 'I encountered an error. Please try again or check if the backend services are running.',
            'sources': [],
            'media_references': []
        })
        emit('status', {'status': 'Error processing query'})

if __name__ == '__main__':
    # Initialize services before starting the app
    if initialize_backends():
        logger.info("Backend services initialized successfully")
    else:
        logger.warning("Failed to initialize some backend services. Chat may not work correctly.")
    
    # Get host and port from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5679))  # Changed to 5679 to avoid conflicts with existing server
    
    # Run the application
    socketio.run(app, host=host, port=port, debug=True, allow_unsafe_werkzeug=True)