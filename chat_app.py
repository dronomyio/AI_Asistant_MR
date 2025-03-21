import os
import json
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from threading import Lock
from contextual_embeddings_weaviate import ContextualEmbeddings
from advanced_retrieval_weaviate import ElasticsearchBM25, hybrid_search, rerank_results, create_rag_response
import anthropic
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'modalai-docs-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Thread lock for thread safety
thread_lock = Lock()

# Global objects for retrieval
embeddings_processor = None
es_bm25 = None
anthropic_client = None

def initialize_backends():
    """Initialize all the backend components."""
    global embeddings_processor, es_bm25, anthropic_client
    
    try:
        # Initialize Anthropic client
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize vector search
        logger.info("Initializing Weaviate connection...")
        embeddings_processor = ContextualEmbeddings(class_name="ModalAIDocument")
        
        # Initialize BM25 search
        logger.info("Initializing Elasticsearch connection...")
        es_bm25 = ElasticsearchBM25()
        
        # Get all documents from Weaviate for ElasticSearch
        logger.info("Retrieving documents from Weaviate for BM25 indexing...")
        all_docs = embeddings_processor.weaviate_db.search(
            embeddings_processor.voyage_client.embed(["modal ai docs"], model="voyage-2").embeddings[0], 
            k=1000
        )
        
        # Prepare docs and index them
        if all_docs:
            es_docs = [doc["metadata"] for doc in all_docs]
            es_bm25.index_documents(es_docs)
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
        # Perform hybrid search
        hybrid_results = hybrid_search(query, embeddings_processor, es_bm25, k=k*2)
        
        # Rerank results
        reranked_results = rerank_results(query, hybrid_results, k=k)
        
        # Generate response
        response = create_rag_response(query, reranked_results, anthropic_client)
        
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
                for result in reranked_results
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

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    socketio.emit('status', {'status': 'Connected to server'})
    
    # Initialize backends if not already done
    if embeddings_processor is None or es_bm25 is None or anthropic_client is None:
        socketio.emit('status', {'status': 'Initializing backend services...'})
        success = initialize_backends()
        if success:
            socketio.emit('status', {'status': 'Ready for queries'})
        else:
            socketio.emit('status', {'status': 'Error initializing services. Check server logs.'})

@socketio.on('query')
def handle_query(data):
    """Handle incoming queries."""
    query = data.get('query', '')
    
    if not query:
        socketio.emit('response', {'error': 'Empty query'})
        return
    
    try:
        socketio.emit('status', {'status': 'Processing query...'})
        result = process_query(query)
        socketio.emit('response', result)
        socketio.emit('status', {'status': 'Ready for queries'})
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        socketio.emit('response', {
            'error': 'An error occurred while processing your query',
            'answer': 'I encountered an error. Please try again or check if the backend services are running.',
            'sources': []
        })
        socketio.emit('status', {'status': 'Error processing query'})

if __name__ == '__main__':
    # Initialize services before starting the app
    if initialize_backends():
        logger.info("Backend services initialized successfully")
    else:
        logger.warning("Failed to initialize some backend services. Chat may not work correctly.")
    
    # Run the application
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)