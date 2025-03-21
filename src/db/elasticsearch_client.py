import os
import logging
from elasticsearch import Elasticsearch, NotFoundError, TransportError
from elasticsearch.helpers import bulk
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ElasticsearchClient:
    def __init__(self, index_name="modalai_bm25_index", url="http://localhost:9200"):
        """
        Initialize a connection to Elasticsearch for BM25 search.
        
        Args:
            index_name: Name of the Elasticsearch index
            url: URL of the Elasticsearch instance
        """
        try:
            self.es_client = Elasticsearch(url)
            self.index_name = index_name
            self.create_index()
        except Exception as e:
            logger.error(f"Error initializing Elasticsearch client: {e}")
            self.es_client = None

    def create_index(self):
        """Create Elasticsearch index with appropriate settings for BM25 search."""
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "title": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "keyword", "index": False},
                    "chunk_id": {"type": "keyword", "index": False},
                    "url": {"type": "keyword", "index": False},
                    "original_index": {"type": "integer", "index": False},
                }
            },
        }
        
        # Create the index if it doesn't exist
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            print(f"Created Elasticsearch index: {self.index_name}")

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents in Elasticsearch.
        
        Args:
            documents: List of document dictionaries to index
        
        Returns:
            Number of successfully indexed documents
        """
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "content": doc["original_content"],
                    "contextualized_content": doc["contextualized_content"],
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "doc_id": doc["doc_id"],
                    "chunk_id": doc["chunk_id"],
                    "original_index": doc["original_index"],
                },
            }
            for doc in documents
        ]
        
        success, _ = bulk(self.es_client, actions)
        self.es_client.indices.refresh(index=self.index_name)
        print(f"Indexed {success} documents in Elasticsearch")
        return success

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """
        Search for documents using BM25.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        # Check if Elasticsearch client exists
        if not self.es_client:
            logger.error("Elasticsearch client is not initialized")
            return []
            
        try:
            # Try to refresh the index
            try:
                self.es_client.indices.refresh(index=self.index_name)
            except Exception as e:
                logger.warning(f"Failed to refresh index: {e}")
            
            # Check if index exists
            if not self.es_client.indices.exists(index=self.index_name):
                logger.warning(f"Index {self.index_name} does not exist")
                return []
                
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^1.0", "contextualized_content^0.8", "title^1.5"],
                    }
                },
                "size": k,
            }
            
            response = self.es_client.search(index=self.index_name, body=search_body)
            
            return [
                {
                    "doc_id": hit["_source"]["doc_id"],
                    "original_index": hit["_source"]["original_index"],
                    "content": hit["_source"]["content"],
                    "title": hit["_source"].get("title", ""),
                    "url": hit["_source"].get("url", ""),
                    "contextualized_content": hit["_source"].get("contextualized_content", ""),
                    "score": hit["_score"],
                }
                for hit in response["hits"]["hits"]
            ]
        except NotFoundError:
            logger.warning(f"Index {self.index_name} not found")
            return []
        except TransportError as e:
            logger.error(f"Transport error during search: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {e}")
            return []
        
    def delete_index(self):
        """Delete the Elasticsearch index."""
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)
            print(f"Deleted Elasticsearch index: {self.index_name}")
            
    def count_documents(self):
        """Return the count of documents in the index."""
        if self.es_client.indices.exists(index=self.index_name):
            return self.es_client.count(index=self.index_name)["count"]
        return 0