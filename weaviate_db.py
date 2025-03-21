import os
import json
import time
import weaviate
from weaviate.auth import AuthApiKey
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

class WeaviateVectorDB:
    def __init__(self, class_name="ModalAIDocument", url="http://localhost:8080", api_key=None):
        """
        Initialize a connection to Weaviate vector database.
        
        Args:
            class_name: Name of the Weaviate class to store documents
            url: URL of the Weaviate instance
            api_key: Optional API key for authentication
        """
        # Connect to Weaviate
        auth_config = AuthApiKey(api_key=api_key) if api_key else None
        
        # Retry connection to handle startup delays
        for _ in range(5):
            try:
                self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
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
        
        # Define class schema
        class_obj = {
            "class": self.class_name,
            "description": "Modal AI documentation chunks with contextual information",
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
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
        }
        
        # Create class
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
        # Create a batch process
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
                
                # Add object with vector
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
            # Perform vector search
            result = (
                self.client.query
                .get(self.class_name, ["content", "contextualContent", "title", "url", "docId", "chunkId", "originalIndex"])
                .with_near_vector({"vector": query_embedding, "certainty": 0.7})
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


# Example usage
def main():
    # Initialize db
    db = WeaviateVectorDB()
    
    # Print count
    count = db.count_objects()
    print(f"Current object count: {count}")
    
    # Test inserting sample data
    if count == 0:
        # Sample data
        texts = ["Sample document 1", "Sample document 2"]
        embeddings = [np.random.rand(1536).tolist(), np.random.rand(1536).tolist()]
        metadata = [
            {
                "original_content": "Sample document 1",
                "contextualized_content": "Context for sample 1",
                "title": "Sample 1",
                "url": "https://example.com/1",
                "doc_id": "doc_1",
                "chunk_id": "chunk_1",
                "original_index": 0
            },
            {
                "original_content": "Sample document 2",
                "contextualized_content": "Context for sample 2",
                "title": "Sample 2",
                "url": "https://example.com/2",
                "doc_id": "doc_2",
                "chunk_id": "chunk_2",
                "original_index": 1
            }
        ]
        
        # Store
        db.store_embeddings(texts, embeddings, metadata)
        
        # Verify
        count = db.count_objects()
        print(f"Updated object count: {count}")
        
    # Test search
    results = db.search(np.random.rand(1536).tolist(), k=2)
    print(f"Found {len(results)} results")
    
if __name__ == "__main__":
    main()