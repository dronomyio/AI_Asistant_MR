import os
import time
import json
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import HybridFusion
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional

class WeaviateClient:
    def __init__(
        self, 
        collection_name="ModalAIDocument", 
        url=None, 
        weaviate_api_key=None,
        cohere_api_key=None,
        openai_api_key=None,
        use_cloud=False,
        create_collection=False,
        embedding_model="text-embedding-ada-002"
    ):
        """
        Initialize a connection to Weaviate vector database (v4).
        
        Args:
            collection_name: Name of the Weaviate collection to store documents
            url: URL of the Weaviate instance (for local or custom deployments)
            weaviate_api_key: Optional API key for authentication
            cohere_api_key: Optional Cohere API key for hybrid search
            openai_api_key: Optional OpenAI API key for embeddings
            use_cloud: If True, will connect to Weaviate Cloud using environment variables
            create_collection: If True, will create the collection if it doesn't exist
            embedding_model: OpenAI embedding model to use (only needed when creating collection)
        """
        # Get API keys from environment if not provided
        weaviate_api_key = weaviate_api_key or os.environ.get("WEAVIATE_API_KEY")
        cohere_api_key = cohere_api_key or os.environ.get("COHERE_API_KEY")
        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        
        # Override collection name from environment if set
        if os.environ.get("WEAVIATE_COLLECTION"):
            collection_name = os.environ.get("WEAVIATE_COLLECTION")
            print(f"Using collection name from environment: {collection_name}")
            
        # Connect to Weaviate (either cloud or local instance)
        for _ in range(5):
            try:
                if use_cloud:
                    # Connect to Weaviate Cloud using environment variables
                    weaviate_url = os.environ.get("WEAVIATE_URL")
                    
                    if not weaviate_url or not weaviate_api_key:
                        raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set as environment variables when use_cloud=True")
                    
                    # Connect to Weaviate Cloud with API key
                    headers = {}
                    if cohere_api_key:
                        headers["X-Cohere-Api-Key"] = cohere_api_key
                    if openai_api_key:
                        headers["X-OpenAI-Api-Key"] = openai_api_key
                    
                    self.client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=weaviate_url,
                        auth_credentials=Auth.api_key(weaviate_api_key),
                        headers=headers
                    )
                else:
                    # Connect to local or custom Weaviate instance
                    connection_params = {}
                    
                    # Add URL if provided, otherwise default to localhost
                    if url:
                        connection_params["url"] = url
                    else:
                        url_from_env = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
                        connection_params["url"] = url_from_env
                    
                    # Add authentication if provided
                    if weaviate_api_key:
                        connection_params["auth_credentials"] = Auth.api_key(weaviate_api_key)
                    
                    # Add API keys for vectorizers if provided
                    headers = {}
                    if cohere_api_key:
                        headers["X-Cohere-Api-Key"] = cohere_api_key
                    if openai_api_key:
                        headers["X-OpenAI-Api-Key"] = openai_api_key
                    if headers:
                        connection_params["headers"] = headers
                    
                    # Connect to Weaviate
                    self.client = weaviate.connect_to_local(**connection_params)
                
                # Check connection
                if self.client.is_ready():
                    print("Connected to Weaviate")
                    break
            except Exception as e:
                print(f"Waiting for Weaviate to be ready: {e}")
                time.sleep(5)
        
        self.collection_name = collection_name
        self.cohere_api_key = cohere_api_key
        self.openai_api_key = openai_api_key
        
        # Check if we should create the collection
        if create_collection:
            self._ensure_collection()
        else:
            # Just try to get the collection without creating it
            try:
                self.collection = self.client.collections.get(self.collection_name)
                print(f"Using existing collection: {self.collection_name}")
            except Exception as e:
                print(f"Warning: Could not get collection {self.collection_name}: {e}")
                print("The collection may not exist or you may not have access to it.")
        
    def _ensure_collection(self):
        """Create the Weaviate collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.collections.list_all()
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                print(f"Collection {self.collection_name} already exists")
                self.collection = self.client.collections.get(self.collection_name)
                return
        except Exception as e:
            print(f"Error checking collections: {e}")
        
        # Create new collection with properties
        try:
            # Determine vectorizer config based on available API keys
            if self.openai_api_key:
                print(f"Using OpenAI vectorizer with model: {self.embedding_model}")
                vectorizer_config = weaviate.classes.config.Configure.Vectorizer.openai(
                    model=self.embedding_model
                )
            elif self.cohere_api_key:
                print("Using Cohere vectorizer")
                vectorizer_config = weaviate.classes.config.Configure.Vectorizer.cohere()
            else:
                print("Using 'none' vectorizer (bring your own embeddings)")
                vectorizer_config = weaviate.classes.config.Configure.Vectorizer.none()
            
            self.collection = self.client.collections.create(
                name=self.collection_name,
                description="Modal AI documentation chunks with contextual information",
                vectorizer_config=vectorizer_config,
                properties=[
                    weaviate.classes.config.Property(
                        name="content",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="The original chunk content",
                        skip_vectorization=False
                    ),
                    weaviate.classes.config.Property(
                        name="contextualContent",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="The contextual information for the chunk",
                        skip_vectorization=False
                    ),
                    weaviate.classes.config.Property(
                        name="title",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="Document title",
                        skip_vectorization=False
                    ),
                    weaviate.classes.config.Property(
                        name="url",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="Document URL",
                        skip_vectorization=True
                    ),
                    weaviate.classes.config.Property(
                        name="docId",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="Document ID",
                        skip_vectorization=True
                    ),
                    weaviate.classes.config.Property(
                        name="chunkId",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="Chunk ID",
                        skip_vectorization=True
                    ),
                    weaviate.classes.config.Property(
                        name="originalIndex",
                        data_type=weaviate.classes.config.DataType.INT,
                        description="Original index of the chunk in the document",
                        skip_vectorization=True
                    ),
                    weaviate.classes.config.Property(
                        name="mediaReferences",
                        data_type=weaviate.classes.config.DataType.TEXT_ARRAY,
                        description="References to media items related to this content",
                        skip_vectorization=True
                    )
                ]
            )
            print(f"Created Weaviate collection: {self.collection_name}")
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    def store_embeddings(self, texts, embeddings, metadata, batch_size=100):
        """
        Store document embeddings in Weaviate.
        
        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            batch_size: Size of batches for insertion
        """
        # Create batch for insertion
        with self.client.batch.dynamic() as batch:
            # Configure batch
            batch.batch_size = batch_size
            
            # Add each document with its embedding
            for i, (text, embedding, meta) in enumerate(tqdm(zip(texts, embeddings, metadata), 
                                                         total=len(texts),
                                                         desc="Storing in Weaviate")):
                # Create properties object
                properties = {
                    "content": meta["original_content"],
                    "contextualContent": meta.get("contextualized_content", ""),
                    "title": meta.get("title", ""),
                    "url": meta.get("url", ""),
                    "docId": meta["doc_id"],
                    "chunkId": meta["chunk_id"],
                    "originalIndex": meta["original_index"]
                }
                
                # Add media references if available
                if "media_references" in meta and meta["media_references"]:
                    # Convert media references to JSON strings to store in Weaviate
                    media_refs_json = [json.dumps(ref) for ref in meta["media_references"]]
                    properties["mediaReferences"] = media_refs_json
                
                # Add object with vector
                batch.add_object(
                    collection=self.collection_name,
                    properties=properties,
                    vector=embedding
                )
    
    def vector_search(self, query_embedding, k=20):
        """
        Search for similar vectors in Weaviate using only vector similarity.
        
        Args:
            query_embedding: The vector to search with
            k: Number of results to return
            
        Returns:
            List of document dictionaries with metadata and similarity scores
        """
        try:
            # Get collection if not already set
            if not hasattr(self, 'collection'):
                self.collection = self.client.collections.get(self.collection_name)
                
            # Perform vector search
            response = (
                self.collection.query
                .near_vector(
                    vector=query_embedding,
                    limit=k
                )
                .with_additional(["distance"])
                .with_fields("content contextualContent title url docId chunkId originalIndex mediaReferences")
                .do()
            )
            
            # Format results
            formatted_results = []
            for item in response.objects:
                # Convert to properties dictionary
                props = item.properties
                
                # Prepare metadata
                metadata = {
                    "original_content": props.get("content", ""),
                    "contextualized_content": props.get("contextualContent", ""),
                    "title": props.get("title", ""),
                    "url": props.get("url", ""),
                    "doc_id": props.get("docId", ""),
                    "chunk_id": props.get("chunkId", ""),
                    "original_index": props.get("originalIndex", 0)
                }
                
                # Convert media references back from JSON strings
                if "mediaReferences" in props and props["mediaReferences"]:
                    try:
                        media_references = [json.loads(ref) for ref in props["mediaReferences"]]
                        metadata["media_references"] = media_references
                    except json.JSONDecodeError as e:
                        print(f"Error decoding media references: {e}")
                
                # Calculate similarity score (1 - distance)
                distance = item.metadata.distance
                similarity = 1.0 - distance if distance is not None else 0.0
                
                formatted_results.append({
                    "metadata": metadata,
                    "similarity": similarity
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching Weaviate: {e}")
            return []
    
    def hybrid_search(self, query_text, query_embedding=None, k=20, alpha=0.5):
        """
        Performs hybrid search using both vector similarity and BM25 text search.
        
        Args:
            query_text: The text query for BM25 search
            query_embedding: Optional embedding vector for vector search
            k: Number of results to return
            alpha: Weight of vector search vs BM25 (0.5 = equal weight)
            
        Returns:
            List of document dictionaries with metadata and hybrid scores
        """
        try:
            # Get collection if not already set
            if not hasattr(self, 'collection'):
                self.collection = self.client.collections.get(self.collection_name)
            
            # Create query builder
            query_builder = self.collection.query
            
            # Configure hybrid search
            if query_embedding is not None:
                # Use both vector and keyword search with specified alpha
                hybrid_query = query_builder.hybrid(
                    query=query_text,
                    vector=query_embedding,
                    alpha=alpha,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    properties=["content", "contextualContent", "title"]
                )
            else:
                # Use keyword search only
                hybrid_query = query_builder.bm25(
                    query=query_text,
                    properties=["content", "contextualContent", "title"]
                )
            
            # Execute search
            response = (
                hybrid_query
                .with_limit(k)
                .with_additional(["score", "explainScore"])
                .with_fields("content contextualContent title url docId chunkId originalIndex mediaReferences")
                .do()
            )
            
            # Format results
            formatted_results = []
            for item in response.objects:
                # Convert to properties dictionary
                props = item.properties
                
                # Prepare metadata
                metadata = {
                    "original_content": props.get("content", ""),
                    "contextualized_content": props.get("contextualContent", ""),
                    "title": props.get("title", ""),
                    "url": props.get("url", ""),
                    "doc_id": props.get("docId", ""),
                    "chunk_id": props.get("chunkId", ""),
                    "original_index": props.get("originalIndex", 0)
                }
                
                # Convert media references back from JSON strings
                if "mediaReferences" in props and props["mediaReferences"]:
                    try:
                        media_references = [json.loads(ref) for ref in props["mediaReferences"]]
                        metadata["media_references"] = media_references
                    except json.JSONDecodeError as e:
                        print(f"Error decoding media references: {e}")
                
                # Get hybrid score
                score = item.metadata.score if hasattr(item.metadata, 'score') else 0.0
                
                formatted_results.append({
                    "metadata": metadata,
                    "score": score,
                    "explain": item.metadata.explain_score if hasattr(item.metadata, 'explain_score') else None
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error performing hybrid search: {e}")
            return []
    
    def search(self, query=None, query_embedding=None, k=20, hybrid=True, alpha=0.5):
        """
        Unified search interface that supports vector, keyword, or hybrid search.
        
        Args:
            query: Text query for keyword or hybrid search
            query_embedding: Vector for vector or hybrid search
            k: Number of results to return
            hybrid: Whether to use hybrid search (requires both query and query_embedding)
            alpha: Weight of vector search vs BM25 (0.5 = equal weight)
            
        Returns:
            List of document dictionaries with metadata and scores
        """
        # Determine search type based on inputs
        if hybrid and query and query_embedding is not None:
            # Use hybrid search
            return self.hybrid_search(query, query_embedding, k, alpha)
        elif query_embedding is not None:
            # Use vector search
            return self.vector_search(query_embedding, k)
        elif query:
            # Use keyword search (hybrid search with alpha=0)
            return self.hybrid_search(query, None, k)
        else:
            # No valid search parameters
            print("Error: Either query or query_embedding must be provided")
            return []
    
    def count_objects(self):
        """Return the count of objects in the collection."""
        try:
            if not hasattr(self, 'collection'):
                self.collection = self.client.collections.get(self.collection_name)
            return self.collection.aggregate.over_all().total_count()
        except Exception as e:
            print(f"Error counting objects: {e}")
            return 0
    
    def delete_collection(self):
        """Delete the collection from Weaviate."""
        try:
            self.client.collections.delete(self.collection_name)
            print(f"Deleted collection {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")