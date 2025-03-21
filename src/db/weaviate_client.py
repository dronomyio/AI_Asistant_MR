import os
import time
import weaviate
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

class WeaviateClient:
    def __init__(self, class_name="ModalAIDocument", url="http://localhost:8080", api_key=None):
        """
        Initialize a connection to Weaviate vector database.
        
        Args:
            class_name: Name of the Weaviate class to store documents
            url: URL of the Weaviate instance
            api_key: Optional API key for authentication
        """
        # Retry connection to handle startup delays
        for _ in range(5):
            try:
                # Simple client initialization (works with both v3 and v4)
                if api_key:
                    auth_config = weaviate.auth.AuthApiKey(api_key)
                    self.client = weaviate.Client(url, auth_client_secret=auth_config)
                else:
                    self.client = weaviate.Client(url)
                
                # Check connection
                if self.client.is_ready():
                    print("Connected to Weaviate")
                    break
            except Exception as e:
                print(f"Waiting for Weaviate to be ready: {e}")
                time.sleep(5)
        
        self.class_name = class_name
        self._create_schema()
        
    def _create_schema(self):
        """Create the Weaviate schema for the collection if it doesn't exist."""
        # Check if class already exists
        try:
            schema = self.client.schema.get()
            classes = [cls['class'] for cls in schema['classes']] if 'classes' in schema else []
            
            if self.class_name in classes:
                print(f"Class {self.class_name} already exists")
                return
        except Exception as e:
            print(f"Error checking schema: {e}")
        
        # Define class properties in v3 format for wider compatibility
        properties = [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The original chunk content"
            },
            {
                "name": "contextualContent",
                "dataType": ["text"],
                "description": "The contextual information for the chunk"
            },
            {
                "name": "title",
                "dataType": ["text"],
                "description": "Document title"
            },
            {
                "name": "url",
                "dataType": ["text"],
                "description": "Document URL"
            },
            {
                "name": "docId",
                "dataType": ["string"],
                "description": "Document ID"
            },
            {
                "name": "chunkId",
                "dataType": ["string"],
                "description": "Chunk ID"
            },
            {
                "name": "originalIndex",
                "dataType": ["number"],
                "description": "Original index of the chunk in the document"
            }
        ]
        
        # Create class using v3 API for wider compatibility
        class_obj = {
            "class": self.class_name,
            "description": "Modal AI documentation chunks with contextual information",
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": properties
        }
        
        try:
            self.client.schema.create_class(class_obj)
            print(f"Created Weaviate class: {self.class_name}")
        except Exception as e:
            print(f"Error creating class: {e}")
    
    def store_embeddings(self, texts, embeddings, metadata, batch_size=100):
        """
        Store document embeddings in Weaviate.
        
        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            batch_size: Size of batches for insertion
        """
        # Create a batch process using v3 API
        with self.client.batch as batch:
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
                
                # Add object with vector using v3 API
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    vector=embedding
                )
    
    def search(self, query_embedding, k=20):
        """
        Search for similar vectors in Weaviate.
        
        Args:
            query_embedding: The vector to search with
            k: Number of results to return
            
        Returns:
            List of document dictionaries with metadata and similarity scores
        """
        try:
            # Perform vector search using v3 API
            result = (
                self.client.query
                .get(self.class_name, ["content", "contextualContent", "title", "url", "docId", "chunkId", "originalIndex"])
                .with_near_vector({"vector": query_embedding})
                .with_limit(k)
                .do()
            )
            
            # Process results
            if result and "data" in result and "Get" in result["data"]:
                items = result["data"]["Get"][self.class_name]
                
                # Format the results
                formatted_results = []
                for item in items:
                    formatted_results.append({
                        "metadata": {
                            "original_content": item["content"],
                            "contextualized_content": item["contextualContent"],
                            "title": item["title"],
                            "url": item["url"],
                            "doc_id": item["docId"],
                            "chunk_id": item["chunkId"],
                            "original_index": item["originalIndex"]
                        },
                        "similarity": item.get("_additional", {}).get("certainty", 0)
                    })
                return formatted_results
            
            return []
        except Exception as e:
            print(f"Error searching Weaviate: {e}")
            return []
    
    def count_objects(self):
        """Return the count of objects in the collection."""
        try:
            result = self.client.query.aggregate(self.class_name).with_meta_count().do()
            return result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
        except Exception as e:
            print(f"Error counting objects: {e}")
            return 0
    
    def delete_class(self):
        """Delete the collection from Weaviate."""
        try:
            self.client.schema.delete_class(self.class_name)
            print(f"Deleted class {self.class_name}")
        except Exception as e:
            print(f"Error deleting class: {e}")