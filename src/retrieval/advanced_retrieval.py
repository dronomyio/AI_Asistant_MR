import os
import json
import cohere
import logging
import sys
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.contextual_embeddings import ContextualEmbeddings
from db.elasticsearch_client import ElasticsearchClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedRetrieval:
    """
    Advanced retrieval system using hybrid search and reranking.
    """
    def __init__(self, embeddings_service, elastic_service):
        """
        Initialize the advanced retrieval system.
        
        Args:
            embeddings_service: ContextualEmbeddings instance
            elastic_service: ElasticsearchClient instance
        """
        self.embeddings_service = embeddings_service
        self.elastic_service = elastic_service
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        
    def hybrid_search(self, query, k=5, semantic_weight=0.7, bm25_weight=0.3):
        """
        Perform hybrid search using both vector similarity and BM25.
        
        Args:
            query: The search query
            k: Number of results to return
            semantic_weight: Weight for semantic search results
            bm25_weight: Weight for BM25 results
            
        Returns:
            List of result documents
        """
        # Check if Weaviate is available
        weaviate_available = hasattr(self.embeddings_service, 'weaviate_db') and self.embeddings_service.weaviate_db is not None
        
        # Number of candidates to retrieve from each source
        num_candidates = 150
        
        # Semantic search using Weaviate
        semantic_results = []
        if weaviate_available:
            try:
                semantic_results = self.embeddings_service.search(query, k=num_candidates)
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
                # Fall back to using only BM25
        # Initialize these variables before using them
        semantic_ids = [(r['metadata']['doc_id'], r['metadata']['original_index']) for r in semantic_results] if semantic_results else []
        
        # BM25 search
        bm25_results = []
        try:
            bm25_results = self.elastic_service.search(query, k=num_candidates)
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            
        bm25_ids = [(r['doc_id'], r['original_index']) for r in bm25_results] if bm25_results else []
        
        # If both search methods failed, return empty results
        if not semantic_results and not bm25_results:
            logger.warning("Both semantic and BM25 search failed. Returning empty results.")
            return []
            
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
    
    def rerank(self, query, results, k=5):
        """
        Rerank results using Cohere's rerank API.
        
        Args:
            query: Search query
            results: List of search results
            k: Number of results to return after reranking
            
        Returns:
            Reranked results
        """
        # If no results, return empty list
        if not results:
            logger.warning("No results to rerank")
            return []
            
        try:
            # Format documents for reranking
            documents = [
                f"Title: {r['metadata'].get('title', '')}\n{r['metadata'].get('original_content', '')}" 
                for r in results
            ]
            
            # Rerank documents
            rerank_results = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=min(k, len(documents))
            )
            
            # Add a small delay to avoid rate limits if needed
            time.sleep(0.1)
            
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
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            # Fall back to original results if reranking fails
            return results[:k]
    
    def retrieve(self, query, k=5):
        """
        Retrieve documents using hybrid search and reranking.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Hybrid search
            hybrid_results = self.hybrid_search(query, k=k*2)
            
            # If no results found in hybrid search, return empty list
            if not hybrid_results:
                logger.warning(f"No results found for query: {query}")
                return []
                
            # Rerank results
            reranked_results = self.rerank(query, hybrid_results, k=k)
            
            return reranked_results
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            return []


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
    try:
        # Check if we have any results
        if not results:
            # Return a fallback response when no results are found
            return "I'm sorry, but I couldn't find any relevant information about that in the Modal AI documentation. Could you try rephrasing your question or asking about a different topic related to Modal AI drones?"
            
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
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return "I'm sorry, but I encountered an error while processing your query. Please try again in a moment."

def format_results(results):
    """
    Format search results for display.
    
    Args:
        results: List of search results
        
    Returns:
        Formatted string
    """
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
    """Main entry point for retrieval."""
    import argparse
    import anthropic
    
    parser = argparse.ArgumentParser(description="Run advanced retrieval for Modal AI documentation")
    parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate URL")
    parser.add_argument("--elastic-url", default="http://localhost:9200", help="Elasticsearch URL")
    parser.add_argument("--query", default=None, help="Query to test (if not provided, runs interactive mode)")
    
    args = parser.parse_args()
    
    # Initialize services
    embeddings_service = ContextualEmbeddings(weaviate_url=args.weaviate_url)
    elastic_service = ElasticsearchClient(url=args.elastic_url)
    retrieval = AdvancedRetrieval(embeddings_service, elastic_service)
    
    # Initialize Anthropic client
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    if args.query:
        # Test a single query
        results = retrieval.retrieve(args.query, k=5)
        print("\n=== Search Results ===")
        print(format_results(results))
        
        print("\n=== Generated Answer ===")
        rag_response = create_rag_response(args.query, results, anthropic_client)
        print(rag_response)
    else:
        # Interactive mode
        print("Modal AI Documentation Search")
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your query: ")
            if query.lower() == 'exit':
                break
                
            # Retrieve and generate response
            print("\nSearching...")
            results = retrieval.retrieve(query, k=5)
            
            print("\n=== Search Results ===")
            print(format_results(results))
            
            print("\n=== Generated Answer ===")
            rag_response = create_rag_response(query, results, anthropic_client)
            print(rag_response)

if __name__ == "__main__":
    main()