import os
import json
import cohere
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from contextual_embeddings_weaviate import ContextualEmbeddings
import anthropic
import voyageai
import time

class ElasticsearchBM25:
    def __init__(self, index_name="modalai_bm25_index"):
        self.es_client = Elasticsearch("http://localhost:9200")
        self.index_name = index_name
        self.create_index()

    def create_index(self):
        """Create Elasticsearch index with appropriate settings for BM25."""
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

    def index_documents(self, documents):
        """Index documents in Elasticsearch."""
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

    def search(self, query, k=20):
        """Search for documents using BM25."""
        self.es_client.indices.refresh(index=self.index_name)
        
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
                "contextualized_content": hit["_source"]["contextualized_content"],
                "score": hit["_score"],
            }
            for hit in response["hits"]["hits"]
        ]

def hybrid_search(query, contextual_embeddings, es_bm25, k=5, semantic_weight=0.7, bm25_weight=0.3):
    """
    Perform hybrid search using both vector similarity and BM25.
    
    Args:
        query: The search query
        contextual_embeddings: ContextualEmbeddings instance
        es_bm25: ElasticsearchBM25 instance
        k: Number of results to return
        semantic_weight: Weight for semantic search results
        bm25_weight: Weight for BM25 results
        
    Returns:
        List of result documents
    """
    # Number of candidates to retrieve from each source
    num_candidates = 150
    
    # Semantic search using Weaviate
    semantic_results = contextual_embeddings.search(query, k=num_candidates)
    semantic_ids = [(r['metadata']['doc_id'], r['metadata']['original_index']) for r in semantic_results]
    
    # BM25 search
    bm25_results = es_bm25.search(query, k=num_candidates)
    bm25_ids = [(r['doc_id'], r['original_index']) for r in bm25_results]
    
    # Combine results using reciprocal rank fusion
    all_ids = list(set(semantic_ids + bm25_ids))
    id_scores = {}
    
    for doc_id in all_ids:
        score = 0
        
        # Add semantic search contribution
        if doc_id in semantic_ids:
            semantic_rank = semantic_ids.index(doc_id) + 1
            score += semantic_weight * (1.0 / semantic_rank)
            
        # Add BM25 contribution
        if doc_id in bm25_ids:
            bm25_rank = bm25_ids.index(doc_id) + 1
            score += bm25_weight * (1.0 / bm25_rank)
            
        id_scores[doc_id] = score
    
    # Sort by score
    sorted_ids = sorted(id_scores.keys(), key=lambda x: id_scores[x], reverse=True)[:k]
    
    # Build final results
    results = []
    for doc_id, original_index in sorted_ids:
        # Find the document
        matching_semantic = next((r for r in semantic_results 
                               if r['metadata']['doc_id'] == doc_id and 
                               r['metadata']['original_index'] == original_index), None)
        
        # Use semantic result if available, otherwise construct from BM25
        if matching_semantic:
            result_doc = {
                "metadata": matching_semantic['metadata'],
                "score": id_scores[(doc_id, original_index)],
                "in_semantic": True,
                "in_bm25": (doc_id, original_index) in bm25_ids
            }
        else:
            matching_bm25 = next((r for r in bm25_results
                              if r['doc_id'] == doc_id and
                              r['original_index'] == original_index), None)
            
            if matching_bm25:
                result_doc = {
                    "metadata": {
                        "doc_id": matching_bm25["doc_id"],
                        "original_index": matching_bm25["original_index"],
                        "original_content": matching_bm25["content"],
                        "contextualized_content": matching_bm25["contextualized_content"],
                        "title": matching_bm25["title"],
                        "url": matching_bm25["url"]
                    },
                    "score": id_scores[(doc_id, original_index)],
                    "in_semantic": False,
                    "in_bm25": True
                }
            else:
                continue  # Skip if neither found (shouldn't happen)
        
        results.append(result_doc)
    
    return results

def rerank_results(query, results, k=5):
    """
    Rerank results using Cohere's rerank API.
    
    Args:
        query: Search query
        results: List of search results
        k: Number of results to return after reranking
        
    Returns:
        Reranked results
    """
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    # Format documents for reranking
    documents = [
        f"Title: {r['metadata'].get('title', '')}\n{r['metadata']['original_content']}" 
        for r in results
    ]
    
    # Rerank documents
    rerank_results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=min(k, len(documents))
    )
    
    # Create reranked results
    reranked = []
    for r in rerank_results.results:
        original_result = results[r.index]
        reranked.append({
            "metadata": original_result["metadata"],
            "score": r.relevance_score,
            "in_semantic": original_result.get("in_semantic", False),
            "in_bm25": original_result.get("in_bm25", False)
        })
    
    return reranked

def create_rag_response(query, results, anthropic_client):
    """
    Create a RAG response using Claude.
    
    Args:
        query: User query
        results: Search results
        anthropic_client: Anthropic client
        
    Returns:
        Generated response
    """
    # Format context from search results
    context = []
    for i, result in enumerate(results):
        metadata = result["metadata"]
        context.append(f"Document {i+1}: {metadata.get('title', 'No title')}")
        context.append(f"URL: {metadata.get('url', 'No URL')}")
        context.append(f"Content: {metadata.get('original_content', '')}")
        context.append("")
    
    context_text = "\n".join(context)
    
    # Create prompt for Claude
    prompt = f"""
    You are a helpful assistant for Modal AI drone technology. Answer the user's question based only on the provided context.
    If the context doesn't contain the information needed to answer the question, say that you don't have enough information.
    Don't make up information that's not in the context.
    
    Context:
    {context_text}
    
    User question: {query}
    """
    
    # Generate response
    response = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

def format_results(results):
    """Format search results for display."""
    output = []
    for i, result in enumerate(results):
        metadata = result["metadata"]
        output.append(f"Result {i+1}: {metadata.get('title', 'No title')}")
        output.append(f"URL: {metadata.get('url', 'No URL')}")
        output.append(f"Score: {result.get('score', 0):.4f}")
        
        in_semantic = result.get("in_semantic", "Unknown")
        in_bm25 = result.get("in_bm25", "Unknown")
        if in_semantic != "Unknown":
            output.append(f"Found in: {'Semantic' if in_semantic else ''}{' & ' if in_semantic and in_bm25 else ''}{'BM25' if in_bm25 else ''}")
        
        content = metadata.get('original_content', '')
        if len(content) > 300:
            content = content[:300] + "..."
        output.append(f"Content: {content}")
        output.append("")
    
    return "\n".join(output)

def main():
    # Load the Weaviate DB and ElasticSearch
    processor = ContextualEmbeddings()
    
    # Get all documents from Weaviate for ElasticSearch
    print("Retrieving documents from Weaviate for BM25 indexing...")
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    
    # Create sample query to get all documents (this is a hack)
    query_embedding = voyage_client.embed(["Modal AI documentation"], model="voyage-2").embeddings[0]
    all_docs = processor.weaviate_db.search(query_embedding, k=1000)  # Increase if needed
    
    # Prepare for ElasticSearch
    es_docs = [doc["metadata"] for doc in all_docs]
    
    # Create BM25 index
    es_bm25 = ElasticsearchBM25()
    es_bm25.index_documents(es_docs)
    
    # Initialize Anthropic client
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Interactive search loop
    print("Modal AI Documentation Search")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
            
        # Hybrid search
        print("\nPerforming hybrid search...")
        hybrid_results = hybrid_search(query, processor, es_bm25, k=10)
        
        # Rerank results
        print("Reranking results...")
        reranked_results = rerank_results(query, hybrid_results, k=5)
        
        # Display results
        print("\n=== Search Results ===")
        print(format_results(reranked_results))
        
        # Generate RAG response
        print("\n=== Generated Answer ===")
        rag_response = create_rag_response(query, reranked_results, anthropic_client)
        print(rag_response)
    
    # Clean up
    if es_bm25.es_client.indices.exists(index=es_bm25.index_name):
        es_bm25.es_client.indices.delete(index=es_bm25.index_name)
        print(f"Deleted Elasticsearch index: {es_bm25.index_name}")

if __name__ == "__main__":
    main()