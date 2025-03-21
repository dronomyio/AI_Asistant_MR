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
from src.embeddings.contextual_embeddings import ContextualEmbeddings
from src.db.elasticsearch_client import ElasticsearchClient
from src.retrieval.advanced_retrieval import AdvancedRetrieval, create_rag_response

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
        # Get URLs from environment or use defaults
        weaviate_url = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
        
        # Weaviate is now exposed on 8081 on the host machine, using 8080 inside docker
        # This ensures we use the correct URL depending on the context
        if "weaviate" in weaviate_url:
            # Inside Docker container
            local_weaviate_url = weaviate_url
        else:
            # For local development outside Docker
            local_weaviate_url = "http://localhost:8081"
            
        elastic_url = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
        
        # Initialize Anthropic client
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize vector search
        logger.info(f"Initializing Weaviate connection to {local_weaviate_url}...")
        embeddings_service = ContextualEmbeddings(weaviate_url=local_weaviate_url)
        
        # Initialize BM25 search
        logger.info(f"Initializing Elasticsearch connection to {elastic_url}...")
        elastic_service = ElasticsearchClient(url=elastic_url)
        
        # Initialize retrieval service
        retrieval_service = AdvancedRetrieval(embeddings_service, elastic_service)
        
        # Get all documents from Weaviate for ElasticSearch indexing
        logger.info("Retrieving documents from Weaviate for BM25 indexing...")
        all_docs = embeddings_service.weaviate_db.search(
            embeddings_service.voyage_client.embed(["modal ai docs"], model="voyage-2").embeddings[0], 
            k=1000
        )
        
        # Prepare docs and index them
        if all_docs:
            es_docs = [doc["metadata"] for doc in all_docs]
            elastic_service.index_documents(es_docs)
            logger.info(f"Indexed {len(es_docs)} documents in Elasticsearch")
        else:
            logger.warning("No documents found in Weaviate. Make sure to run the data pipeline first.")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing backends: {e}")
        return False

def process_query(query, k=5):
    """Process a query through the retrieval pipeline."""
    try:
        # Perform retrieval
        results = retrieval_service.retrieve(query, k=k)
        
        # Generate response
        response = create_rag_response(query, results, anthropic_client)
        
        # Return results
        return {
            "answer": response,
            "sources": [
                {
                    "title": result["metadata"].get("title", "No title"),
                    "url": result["metadata"].get("url", "No URL"),
                    "content": result["metadata"].get("original_content", "")[:200] + "..." 
                    if len(result["metadata"].get("original_content", "")) > 200 else 
                    result["metadata"].get("original_content", "")
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "answer": "I encountered an error while processing your query. Please try again.",
            "sources": []
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
        emit('response', result)
        emit('status', {'status': 'Ready for queries'})
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        emit('response', {
            'error': 'An error occurred while processing your query',
            'answer': 'I encountered an error. Please try again or check if the backend services are running.',
            'sources': []
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
    port = int(os.getenv("PORT", 5000))
    
    # Run the application
    socketio.run(app, host=host, port=port, debug=True, allow_unsafe_werkzeug=True)