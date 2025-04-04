#!/usr/bin/env python3
"""
AI Assistant for querying Weaviate Cloud with existing embeddings
"""
import os
import sys
import json
import logging
import anthropic
from dotenv import load_dotenv
from weaviate_db import WeaviateDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def initialize_clients():
    """Initialize Weaviate and Anthropic clients"""
    # Check for required environment variables
    required_vars = ["WEAVIATE_URL", "WEAVIATE_API_KEY", "ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file or environment")
        sys.exit(1)
    
    # Optional API keys with messages
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set - query embeddings will be generated using text search only")
    
    if not os.environ.get("COHERE_API_KEY"):
        logger.warning("COHERE_API_KEY not set - reranking will not be available")
    
    try:
        # Initialize Weaviate client
        logger.info(f"Connecting to Weaviate at {os.environ['WEAVIATE_URL']}...")
        db = WeaviateDB(
            cluster_url=os.environ["WEAVIATE_URL"],
            api_key=os.environ["WEAVIATE_API_KEY"],
            cohere_api_key=os.environ.get("COHERE_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            create_schema=False  # Don't create new collections
        )
        
        # Initialize Anthropic client
        logger.info("Initializing Anthropic Claude...")
        claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        
        return db, claude
    except Exception as e:
        logger.error(f"Error initializing clients: {e}")
        sys.exit(1)

def prepare_context(search_results):
    """Prepare context for Claude from search results"""
    text_context = []
    media_references = []
    
    # Process text chunks
    if "TextChunk" in search_results:
        for i, result in enumerate(search_results["TextChunk"]):
            metadata = result["metadata"]
            
            # Add document content
            text_context.append(f"Document {i+1}: {metadata.get('title', 'No title')}")
            if "source_file" in metadata:
                text_context.append(f"Source: {metadata['source_file']}")
            
            # Add the content (prefer text over content for ETL pipeline compatibility)
            content = metadata.get("text", metadata.get("content", ""))
            text_context.append(f"Content: {content}")
            text_context.append("")
            
            # Collect media references
            if "media_references" in metadata:
                media_references.extend(metadata["media_references"])
    
    # Process image embeddings
    if "ImageEmbedding" in search_results:
        text_context.append("Related Images:")
        for i, result in enumerate(search_results["ImageEmbedding"]):
            metadata = result["metadata"]
            img_path = metadata.get("image_path", "")
            img_desc = metadata.get("alt_text", metadata.get("file_name", ""))
            
            # Add image reference to context
            text_context.append(f"Image {i+1}: {img_desc}")
            if img_path:
                text_context.append(f"Path: {img_path}")
            text_context.append("")
            
            # Add to media references
            if img_path:
                media_references.append({
                    "type": "image",
                    "path": img_path,
                    "alt_text": img_desc,
                    "static_path": f"/static/media/{os.path.basename(img_path)}"
                })
    
    # Deduplicate media references
    unique_media = []
    seen_paths = set()
    for media in media_references:
        path = media.get("path", "")
        if path and path not in seen_paths:
            seen_paths.add(path)
            unique_media.append(media)
    
    return "\n".join(text_context), unique_media

def generate_response(query, context, media_refs, claude):
    """Generate a response using Claude"""
    # Create prompt with context
    prompt = f"""
    You are a helpful assistant for Modal AI drone technology. Answer the user's question based only on the provided context.
    If the context doesn't contain the information needed to answer the question, say that you don't have enough information.
    Don't make up information that's not in the context.
    
    Context:
    {context}
    
    User question: {query}
    """
    
    try:
        # Generate response
        response = claude.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text, media_refs
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I encountered an error while trying to answer your question: {str(e)}", []

def process_query(user_query, db, claude):
    """Process a user query and return a response"""
    logger.info(f"Processing query: {user_query}")
    
    try:
        # Search across all collections with automatic embedding generation
        search_results = db.search(
            query_text=user_query,
            limit=10  # Retrieve more results for better context
        )
        
        # If no results from text search, try hybrid search
        total_results = sum(len(results) for results in search_results.values())
        if total_results == 0 and os.environ.get("OPENAI_API_KEY"):
            logger.info("No results from text search, trying hybrid search...")
            search_results = db.search_hybrid(
                query=user_query,
                limit=10
            )
        
        # Generate contextual answer with Anthropic Claude
        context, media_refs = prepare_context(search_results)
        response, media = generate_response(user_query, context, media_refs, claude)
        
        return response, media
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"I encountered an error while processing your query: {str(e)}", []

def interactive_mode():
    """Run the assistant in interactive mode"""
    db, claude = initialize_clients()
    
    print("\nModal AI Assistant")
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() in ('exit', 'quit'):
            break
            
        response, media = process_query(query, db, claude)
        
        print("\nResponse:")
        print(response)
        
        if media:
            print("\nRelevant Media:")
            for i, m in enumerate(media[:3]):  # Show only first 3 media items
                print(f"- {m.get('alt_text', 'Image')} ({m.get('type', 'media')})")
            
            if len(media) > 3:
                print(f"  ...and {len(media) - 3} more media items")

def api_handler(query):
    """Handle API request with a query"""
    db, claude = initialize_clients()
    response, media = process_query(query, db, claude)
    
    return {
        "answer": response,
        "media_references": media
    }

if __name__ == "__main__":
    # Check if query is provided as argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        db, claude = initialize_clients()
        response, media = process_query(query, db, claude)
        print(response)
    else:
        # Start interactive mode
        interactive_mode()