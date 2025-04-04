"""
Weaviate client for storing and retrieving embeddings from the ETL pipeline.
Uses Weaviate client v4 API.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import HybridFusion
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeaviateDB:
    """Client for storing and retrieving embeddings from Weaviate."""
    
    def __init__(
        self,
        cluster_url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        create_schema: bool = False
    ):
        """
        Initialize the Weaviate client.
        
        Args:
            cluster_url: URL of the Weaviate cluster (e.g., from Weaviate Cloud Service)
            api_key: API key for Weaviate Cloud
            embedding_api_key: Generic API key for embedding provider
            openai_api_key: Specific API key for OpenAI
            cohere_api_key: Specific API key for Cohere
            create_schema: Whether to create the schema if it doesn't exist
        """
        # Get credentials from environment variables if not provided
        self.cluster_url = cluster_url or os.environ.get("WEAVIATE_URL", "http://localhost:8080")
        self.api_key = api_key or os.environ.get("WEAVIATE_API_KEY")
        
        # Set up embedding API keys with priority
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.cohere_api_key = cohere_api_key or os.environ.get("COHERE_API_KEY")
        self.embedding_api_key = embedding_api_key  # Generic fallback
        
        # Set up headers for embedding provider
        headers = {}
        if self.openai_api_key:
            headers["X-OpenAI-Api-Key"] = self.openai_api_key
            logger.info("Using OpenAI for embeddings")
        if self.cohere_api_key:
            headers["X-Cohere-Api-Key"] = self.cohere_api_key
            logger.info("Using Cohere for reranking")
        
        # Connect to Weaviate - try cloud first, then local
        try:
            if "weaviate.cloud" in self.cluster_url and self.api_key:
                # Connect to Weaviate Cloud
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.cluster_url,
                    auth_credentials=Auth.api_key(self.api_key),
                    headers=headers
                )
                logger.info(f"Connected to Weaviate Cloud at {self.cluster_url}")
            else:
                # Connect to local or custom Weaviate instance
                auth_credentials = Auth.api_key(self.api_key) if self.api_key else None
                connection_params = {
                    "url": self.cluster_url,
                }
                if auth_credentials:
                    connection_params["auth_credentials"] = auth_credentials
                if headers:
                    connection_params["headers"] = headers
                
                self.client = weaviate.connect_to_local(**connection_params)
                logger.info(f"Connected to Weaviate at {self.cluster_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise ConnectionError(f"Failed to connect to Weaviate at {self.cluster_url}: {e}")
        
        # Check if connection is ready
        if not self.client.is_ready():
            raise ConnectionError(f"Weaviate client is not ready at {self.cluster_url}")
        
        # Create schema if requested
        if create_schema:
            self._create_schema()
        else:
            # Just check if our collections exist
            collections = self.client.collections.list_all()
            collection_names = [c.name for c in collections]
            if "TextChunk" in collection_names:
                self.text_collection = self.client.collections.get("TextChunk")
                logger.info("Found existing TextChunk collection")
            if "ImageEmbedding" in collection_names:
                self.image_collection = self.client.collections.get("ImageEmbedding")
                logger.info("Found existing ImageEmbedding collection")
    
    def _create_schema(self):
        """Create the schema for document chunks and image embeddings."""
        # Check if our collections exist first
        collections = self.client.collections.list_all()
        collection_names = [c.name for c in collections]
        
        # Create TextChunk collection if it doesn't exist
        if "TextChunk" not in collection_names:
            vectorizer = Configure.Vectorizer.none()
            if self.openai_api_key:
                vectorizer = Configure.Vectorizer.openai(
                    model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
                )
            elif self.cohere_api_key:
                vectorizer = Configure.Vectorizer.cohere()
                
            self.text_collection = self.client.collections.create(
                name="TextChunk",
                description="A chunk of text from a document",
                properties=[
                    Property(
                        name="text", 
                        data_type=DataType.TEXT,
                        description="The text content of the chunk"
                    ),
                    Property(
                        name="type", 
                        data_type=DataType.TEXT,
                        description="The type of the text (Title, Text, etc.)"
                    ),
                    Property(
                        name="source_file", 
                        data_type=DataType.TEXT,
                        description="The source file path"
                    ),
                    Property(
                        name="element_id", 
                        data_type=DataType.TEXT,
                        description="Unique ID for the element"
                    ),
                    # Add compatibility with legacy fields
                    Property(
                        name="content", 
                        data_type=DataType.TEXT,
                        description="The text content (legacy compatibility)"
                    ),
                    Property(
                        name="contextualContent", 
                        data_type=DataType.TEXT,
                        description="Contextual information (legacy compatibility)"
                    ),
                    Property(
                        name="title", 
                        data_type=DataType.TEXT,
                        description="Document title (legacy compatibility)"
                    ),
                    Property(
                        name="mediaReferences", 
                        data_type=DataType.TEXT_ARRAY,
                        description="References to media items (legacy compatibility)"
                    )
                ],
                vectorizer_config=vectorizer
            )
            logger.info("Created TextChunk collection")
        else:
            self.text_collection = self.client.collections.get("TextChunk")
        
        # Create ImageEmbedding collection if it doesn't exist
        if "ImageEmbedding" not in collection_names:
            vectorizer = Configure.Vectorizer.none()
            if self.openai_api_key:
                vectorizer = Configure.Vectorizer.openai(
                    model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
                )
                
            self.image_collection = self.client.collections.create(
                name="ImageEmbedding",
                description="Embedding for an image",
                properties=[
                    Property(
                        name="image_path", 
                        data_type=DataType.TEXT,
                        description="Path to the image"
                    ),
                    Property(
                        name="file_name", 
                        data_type=DataType.TEXT,
                        description="File name of the image"
                    ),
                    # Add compatibility with legacy fields
                    Property(
                        name="alt_text", 
                        data_type=DataType.TEXT,
                        description="Alternative text for the image"
                    ),
                    Property(
                        name="source_file", 
                        data_type=DataType.TEXT,
                        description="Source document that references this image"
                    )
                ],
                vectorizer_config=vectorizer
            )
            logger.info("Created ImageEmbedding collection")
        else:
            self.image_collection = self.client.collections.get("ImageEmbedding")
    
    def store_text_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Store text chunks with embeddings in Weaviate.
        
        Args:
            chunks: List of text chunks with embeddings
            batch_size: Number of objects to batch together in each request
        """
        # Get the collection
        if not hasattr(self, 'text_collection'):
            self.text_collection = self.client.collections.get("TextChunk")
        
        # Use batch processing
        with self.text_collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                if i % batch_size == 0:
                    logger.info(f"Processing chunks {i} to {min(i+batch_size, len(chunks))}")
                
                # Skip if no embedding or text
                if "embedding" not in chunk or not (chunk.get("text") or chunk.get("content")):
                    continue
                
                # Extract properties with compatibility between ETL and legacy fields
                properties = {
                    "text": chunk.get("text", chunk.get("content", "")),
                    "type": chunk.get("type", "Text"),
                    "source_file": chunk.get("source_file", ""),
                    "element_id": chunk.get("element_id", chunk.get("chunk_id", "")),
                    # Legacy compatibility
                    "content": chunk.get("content", chunk.get("text", "")),
                    "contextualContent": chunk.get("contextualContent", chunk.get("context", "")),
                    "title": chunk.get("title", "")
                }
                
                # Handle media references
                if "media_references" in chunk and chunk["media_references"]:
                    if isinstance(chunk["media_references"][0], str):
                        properties["mediaReferences"] = chunk["media_references"]
                    else:
                        # Convert dict to JSON strings
                        properties["mediaReferences"] = [json.dumps(ref) for ref in chunk["media_references"]]
                
                # Add to batch with vector
                batch.add_object(
                    properties=properties,
                    vector=chunk["embedding"]
                )
                
        logger.info(f"Stored {len(chunks)} text chunks in Weaviate")
    
    def store_image_embeddings(self, images: List[Dict[str, Any]], batch_size: int = 50):
        """
        Store image embeddings in Weaviate.
        
        Args:
            images: List of image paths with embeddings
            batch_size: Number of objects to batch together in each request
        """
        # Get the collection
        if not hasattr(self, 'image_collection'):
            self.image_collection = self.client.collections.get("ImageEmbedding")
        
        # Use batch processing
        with self.image_collection.batch.dynamic() as batch:
            for i, image in enumerate(images):
                if i % batch_size == 0:
                    logger.info(f"Processing images {i} to {min(i+batch_size, len(images))}")
                
                # Skip if no embedding or path
                if "embedding" not in image or "image_path" not in image:
                    continue
                
                # Get file name from path
                file_name = Path(image.get("image_path", "")).name
                
                # Add to batch with vector
                batch.add_object(
                    properties={
                        "image_path": image.get("image_path", ""),
                        "file_name": file_name,
                        "alt_text": image.get("alt_text", image.get("description", file_name)),
                        "source_file": image.get("source_file", "")
                    },
                    vector=image["embedding"]
                )
                
        logger.info(f"Stored {len(images)} image embeddings in Weaviate")
    
    def search(self, 
               query_text: Optional[str] = None, 
               query_embedding: Optional[List[float]] = None, 
               limit: int = 10, 
               collection_names: Optional[List[str]] = None,
               alpha: float = 0.5) -> Dict[str, Any]:
        """
        Search for similar objects using a query text and/or embedding.
        
        Args:
            query_text: The text query to search with
            query_embedding: The embedding vector to search with
            limit: Maximum number of results to return
            collection_names: List of collection names to search in (default: all)
            alpha: Weight between vector and keyword search (1.0 = vector only)
            
        Returns:
            Dictionary with search results by collection
        """
        collections = collection_names or ["TextChunk", "ImageEmbedding"]
        results = {}
        
        # Both text and embedding provided - use hybrid search
        for collection_name in collections:
            try:
                collection = self.client.collections.get(collection_name)
                
                # Determine search type based on inputs
                if query_text and query_embedding is not None:
                    # Hybrid search
                    query_result = collection.query.hybrid(
                        query=query_text,
                        vector=query_embedding,
                        alpha=alpha,
                        fusion_type=HybridFusion.RELATIVE_SCORE,
                        limit=limit,
                        properties=["text", "content", "title", "image_path", "file_name", 
                                   "source_file", "type", "element_id", "mediaReferences"]
                    ).with_additional(["distance", "score"]).do()
                elif query_embedding is not None:
                    # Vector search only
                    query_result = collection.query.near_vector(
                        near_vector=query_embedding,
                        limit=limit,
                        with_distance=True
                    ).with_additional(["distance"]).with_fields(
                        "text content title image_path file_name source_file type element_id mediaReferences"
                    ).do()
                elif query_text:
                    # BM25 search only
                    query_result = collection.query.bm25(
                        query=query_text,
                        limit=limit,
                        properties=["text", "content", "title", "file_name"]
                    ).with_additional(["score"]).with_fields(
                        "text content title image_path file_name source_file type element_id mediaReferences"
                    ).do()
                else:
                    # Invalid search parameters
                    logger.warning("Either query_text or query_embedding must be provided")
                    results[collection_name] = []
                    continue
                
                # Process results - convert to standard format
                processed_results = []
                for obj in query_result.objects:
                    # Create a metadata dict with all properties
                    metadata = dict(obj.properties)
                    
                    # Add legacy fields if needed for compatibility
                    if "text" in metadata and "content" not in metadata:
                        metadata["content"] = metadata["text"]
                    if "content" in metadata and "text" not in metadata:
                        metadata["text"] = metadata["content"]
                    
                    # Process media references if they exist
                    if "mediaReferences" in metadata and metadata["mediaReferences"]:
                        try:
                            # Parse JSON strings to objects if needed
                            if isinstance(metadata["mediaReferences"][0], str):
                                metadata["media_references"] = [json.loads(ref) for ref in metadata["mediaReferences"]]
                            else:
                                metadata["media_references"] = metadata["mediaReferences"]
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Could not parse media references: {metadata['mediaReferences']}")
                    
                    # Get score or distance (prefer score for hybrid search)
                    score = 0.0
                    if hasattr(obj.metadata, 'score'):
                        score = obj.metadata.score
                    elif hasattr(obj.metadata, 'distance'):
                        # Convert distance to similarity score (1.0 - distance)
                        score = 1.0 - obj.metadata.distance
                    
                    processed_results.append({
                        "metadata": metadata,
                        "score": score,
                        "id": obj.uuid
                    })
                
                results[collection_name] = processed_results
            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")
                results[collection_name] = []
                
        return results
    
    def search_hybrid(self, query: str, query_embedding: Optional[List[float]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search across all collections and return a combined result list.
        
        Args:
            query: The text query
            query_embedding: Optional embedding vector (will be generated if not provided)
            limit: Maximum number of results to return
            
        Returns:
            List of search results from all collections, sorted by score
        """
        # Generate embedding if needed
        if query_embedding is None and self.openai_api_key:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.embeddings.create(
                model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
                input=query
            )
            query_embedding = response.data[0].embedding
            logger.info("Generated OpenAI embedding for query")
        
        # Search all collections
        search_results = self.search(
            query_text=query,
            query_embedding=query_embedding,
            limit=limit * 2  # Get more results to merge
        )
        
        # Combine results from all collections
        combined_results = []
        for collection, results in search_results.items():
            # Add collection name to metadata
            for result in results:
                result["metadata"]["collection"] = collection
                combined_results.append(result)
        
        # Sort by score and take top results
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:limit]

# For backward compatibility
WeaviateClient = WeaviateDB