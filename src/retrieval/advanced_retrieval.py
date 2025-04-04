import os
import json
import cohere
import logging
import sys
import time
import anthropic
import voyageai

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.contextual_embeddings import ContextualEmbeddings
from embeddings.multimodal_embeddings import MultimodalEmbeddings
from db.elasticsearch_client import ElasticsearchClient
from db.weaviate_client import WeaviateClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedRetrieval:
    """
    Advanced retrieval system using hybrid search and reranking.
    """
    def __init__(
        self, 
        embeddings_service=None, 
        elastic_service=None,
        weaviate_client=None,
        collection_name="ModalAIDocument",
        cohere_api_key=None,
        anthropic_api_key=None,
        voyage_api_key=None,
        openai_api_key=None,
        use_weaviate_cloud=False,
        openai_model="text-embedding-ada-002",
        create_collection=False
    ):
        """
        Initialize the advanced retrieval system.
        
        Args:
            embeddings_service: Optional ContextualEmbeddings instance (legacy)
            elastic_service: Optional ElasticsearchClient instance (legacy)
            weaviate_client: Optional WeaviateClient instance
            collection_name: Name of the Weaviate collection
            cohere_api_key: API key for Cohere (will fall back to env var)
            anthropic_api_key: API key for Anthropic (will fall back to env var)
            voyage_api_key: API key for Voyage AI (will fall back to env var)
            openai_api_key: API key for OpenAI (will fall back to env var)
            use_weaviate_cloud: Whether to use Weaviate Cloud (env vars must be set)
            openai_model: OpenAI embedding model to use
            create_collection: Whether to create collection if it doesn't exist
        """
        # Legacy services support
        self.embeddings_service = embeddings_service
        self.elastic_service = elastic_service
        
        # Setup API keys
        self.cohere_api_key = cohere_api_key or os.environ.get("COHERE_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.voyage_api_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_model = openai_model
        
        # Initialize Weaviate client
        if weaviate_client:
            self.weaviate_client = weaviate_client
        else:
            self.weaviate_client = WeaviateClient(
                collection_name=collection_name,
                cohere_api_key=self.cohere_api_key,
                openai_api_key=self.openai_api_key,
                use_cloud=use_weaviate_cloud,
                create_collection=create_collection,
                embedding_model=openai_model
            )
        
        # Initialize Cohere client if API key provided
        if self.cohere_api_key:
            self.cohere_client = cohere.Client(api_key=self.cohere_api_key)
        else:
            self.cohere_client = None
            logger.warning("Cohere API key not provided. Reranking will not be available.")
            
        # Initialize Anthropic client if API key provided
        if self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        else:
            self.anthropic_client = None
            logger.warning("Anthropic API key not provided. RAG responses will not be available.")
            
        # Initialize Voyage client if API key provided
        if self.voyage_api_key:
            self.voyage_client = voyageai.Client(api_key=self.voyage_api_key)
        else:
            self.voyage_client = None
            logger.warning("Voyage API key not provided. Voyage embeddings will not be available.")
            
        # Initialize OpenAI client if API key provided
        if self.openai_api_key:
            # Import here to avoid unnecessary dependency
            import openai
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not provided. OpenAI embeddings will not be available.")
        
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
    
    def embed_query(self, query_text):
        """
        Generate an embedding for the query using available embedding services.
        Tries OpenAI first, then Voyage AI, then falls back to legacy service.
        
        Args:
            query_text: The query to embed
            
        Returns:
            The query embedding vector or None if no embedding service is available
        """
        # Try using OpenAI for embedding (to match existing DB vectors)
        if hasattr(self, 'openai_client') and self.openai_client:
            try:
                import openai
                response = self.openai_client.embeddings.create(
                    model=getattr(self, 'openai_model', "text-embedding-ada-002"),
                    input=query_text
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Error generating OpenAI embedding: {e}")
                
        # Fall back to Voyage AI if available
        if hasattr(self, 'voyage_client') and self.voyage_client:
            try:
                response = self.voyage_client.embed([query_text], model="voyage-2")
                return response.embeddings[0]
            except Exception as e:
                logger.error(f"Error generating Voyage embedding: {e}")
        
        # Try legacy embeddings service as last resort
        if self.embeddings_service and hasattr(self.embeddings_service, 'embed_query'):
            try:
                return self.embeddings_service.embed_query(query_text)
            except Exception as e:
                logger.error(f"Error using legacy embedding service: {e}")
                
        # If all methods fail, return None
        logger.warning("No embedding service available to generate query embedding")
        return None
    
    def weaviate_search(self, query, query_embedding=None, k=20, hybrid=True, alpha=0.5):
        """
        Search using Weaviate client (v4 API) with hybrid search.
        
        Args:
            query: Text query for keyword or hybrid search
            query_embedding: Vector for vector or hybrid search
            k: Number of results to return
            hybrid: Whether to use hybrid search (requires both query and query_embedding)
            alpha: Weight of vector search vs BM25 (0.5 = equal weight)
            
        Returns:
            List of results from Weaviate
        """
        if not hasattr(self, 'weaviate_client') or not self.weaviate_client:
            return []
            
        try:
            results = self.weaviate_client.search(
                query=query,
                query_embedding=query_embedding,
                k=k,
                hybrid=hybrid,
                alpha=alpha
            )
            return results
        except Exception as e:
            logger.error(f"Error in Weaviate search: {e}")
            return []
    
    def retrieve(self, query, k=5, use_legacy=False, hybrid=True, alpha=0.5):
        """
        Retrieve documents using hybrid search and reranking.
        
        Args:
            query: Search query
            k: Number of results to return
            use_legacy: Whether to use legacy hybrid search implementation
            hybrid: Whether to use hybrid search in Weaviate
            alpha: Weight of vector vs BM25 search for hybrid (0.5 = equal weight)
            
        Returns:
            List of search results
        """
        try:
            results = []
            
            # First try using Weaviate v4 API if available
            if not use_legacy and hasattr(self, 'weaviate_client') and self.weaviate_client:
                # Generate query embedding
                query_embedding = self.embed_query(query)
                
                # Search with Weaviate
                results = self.weaviate_search(
                    query=query,
                    query_embedding=query_embedding,
                    k=min(100, k * 5),  # Get more candidates for reranking
                    hybrid=hybrid,
                    alpha=alpha
                )
            else:
                # Fall back to legacy hybrid search
                logger.info("Using legacy hybrid search implementation")
                results = self.hybrid_search(query, k=k*2)
            
            # If no results found, return empty list
            if not results:
                logger.warning(f"No results found for query: {query}")
                return []
                
            # Rerank results
            if hasattr(self, 'cohere_client') and self.cohere_client:
                reranked_results = self.rerank(query, results, k=k)
                return reranked_results
            else:
                # Return top-k results without reranking
                return results[:k] if len(results) > k else results
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            return []


def create_rag_response(query, results, anthropic_client=None, data_dir=None, include_media=True):
    """
    Create a RAG response using Claude.
    
    Args:
        query: User query
        results: Search results
        anthropic_client: Anthropic client (can be None if results have an instance with anthropic_client)
        data_dir: Directory containing media files
        include_media: Whether to include media references in the response
        
    Returns:
        Generated response and list of media references
    """
    # Try to get Anthropic client from first result if it has one
    if not anthropic_client and results and hasattr(results[0], 'anthropic_client'):
        anthropic_client = results[0].anthropic_client
        
    # If still no client, return error
    if not anthropic_client:
        return "I'm sorry, but I can't generate a response without access to the Anthropic API.", []
    
    try:
        # Check if we have any results
        if not results:
            # Return a fallback response when no results are found
            return "I'm sorry, but I couldn't find any relevant information about that in the Modal AI documentation. Could you try rephrasing your question or asking about a different topic related to Modal AI drones?", []
            
        # Format context from search results
        context = []
        media_references = []
        
        for i, result in enumerate(results):
            metadata = result["metadata"]
            context.append(f"Document {i+1}: {metadata.get('title', 'No title')}")
            if "url" in metadata and metadata["url"]:
                context.append(f"URL: {metadata['url']}")
            
            context.append(f"Content: {metadata.get('original_content', '')}")
            
            # Add media references if available and requested
            if include_media and 'media_references' in metadata and metadata['media_references']:
                context.append("Media references:")
                for j, media in enumerate(metadata['media_references']):
                    media_ref_id = f"doc{i+1}_media{j+1}"
                    media_type = media.get('type', 'unknown')
                    media_desc = media.get('alt_text', '') or media.get('link_text', '') or f"{media_type} file"
                    context.append(f"- {media_ref_id}: {media_desc} ({media_type})")
                    
                    # Add to media references list
                    media_references.append({
                        "id": media_ref_id,
                        "path": media.get('path', ''),
                        "static_path": media.get('static_path', ''),
                        "type": media_type,
                        "alt_text": media_desc,
                        "description": media_desc,
                        "doc_index": i
                    })
            
            context.append("")
        
        context_text = "\n".join(context)
        
        # Create prompt for Claude
        media_instruction = ""
        if media_references:
            media_instruction = """
            When it would be helpful to reference visual information, mention the media reference ID in your response.
            For example: "As shown in doc1_media1, the drone components include..."
            """
        
        prompt = f"""
        You are a helpful assistant for Modal AI drone technology. Answer the user's question based only on the provided context.
        If the context doesn't contain the information needed to answer the question, say that you don't have enough information.
        Don't make up information that's not in the context.
        {media_instruction}
        
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
        
        result_text = response.content[0].text
        
        # Extract referenced media
        referenced_media = []
        if media_references:
            for media in media_references:
                if media["id"] in result_text:
                    referenced_media.append(media)
        
        return result_text, referenced_media
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return f"I'm sorry, but I encountered an error while processing your query: {str(e)}", []

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
        
        # Add media references if available
        media_refs = metadata.get('media_references', [])
        if media_refs:
            output.append(f"Media references: {len(media_refs)} items")
            for j, media in enumerate(media_refs[:3]):  # Show only first 3 media items
                media_type = media.get('type', 'unknown')
                media_desc = media.get('alt_text', '') or media.get('link_text', '') or f"{media_type} file"
                output.append(f"  - Media {j+1}: {media_desc} ({media_type})")
            
            if len(media_refs) > 3:
                output.append(f"  - ... and {len(media_refs) - 3} more media items")
        
        output.append("")
    
    return "\n".join(output)

def main():
    """Main entry point for retrieval."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run advanced retrieval for Modal AI documentation")
    parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate URL")
    parser.add_argument("--elastic-url", default="http://localhost:9200", help="Elasticsearch URL")
    parser.add_argument("--collection-name", default="ModalAIDocument", help="Weaviate collection name")
    parser.add_argument("--query", default=None, help="Query to test (if not provided, runs interactive mode)")
    parser.add_argument("--multimodal", action="store_true", help="Use multimodal embeddings")
    parser.add_argument("--data-dir", default="data", help="Directory containing processed media files")
    parser.add_argument("--use-legacy", action="store_true", help="Use legacy retrieval implementation")
    parser.add_argument("--use-weaviate-cloud", action="store_true", help="Use Weaviate Cloud instead of local")
    
    args = parser.parse_args()
    
    # Create retrieval object - first try newer implementation with direct Weaviate client
    if not args.use_legacy:
        logger.info("Using Weaviate v4 client implementation")
        retrieval = AdvancedRetrieval(
            collection_name=args.collection_name,
            use_weaviate_cloud=args.use_weaviate_cloud
        )
    else:
        # Initialize legacy services
        logger.info("Using legacy implementation with embedding service")
        if args.multimodal:
            logger.info("Using multimodal embeddings")
            embeddings_service = MultimodalEmbeddings(weaviate_url=args.weaviate_url)
            # Load processed media if available
            media_file = os.path.join(args.data_dir, "modalai_processed_media.json")
            if os.path.exists(media_file):
                embeddings_service.load_processed_media(media_file)
        else:
            logger.info("Using text-only embeddings")
            embeddings_service = ContextualEmbeddings(weaviate_url=args.weaviate_url)
        
        elastic_service = ElasticsearchClient(url=args.elastic_url)
        retrieval = AdvancedRetrieval(embeddings_service, elastic_service)
    
    # Check if Anthropic client is available (in retrieval object or directly)
    anthropic_client = None
    if hasattr(retrieval, 'anthropic_client'):
        anthropic_client = retrieval.anthropic_client
    
    if args.query:
        # Test a single query
        print(f"\nSearching for: {args.query}")
        results = retrieval.retrieve(
            args.query, 
            k=5, 
            use_legacy=args.use_legacy
        )
        
        print("\n=== Search Results ===")
        print(format_results(results))
        
        # Generate RAG response if we have results
        if results:
            print("\n=== Generated Answer ===")
            rag_response, media_refs = create_rag_response(
                args.query, 
                results, 
                anthropic_client
            )
            print(rag_response)
            
            # Display media references if available
            if media_refs:
                print("\n=== Referenced Media ===")
                for ref in media_refs:
                    print(f"- {ref.get('id')}: {ref.get('description')}")
                    if 'static_path' in ref and ref['static_path']:
                        print(f"  Path: {ref['static_path']}")
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
            results = retrieval.retrieve(
                query, 
                k=5, 
                use_legacy=args.use_legacy
            )
            
            print("\n=== Search Results ===")
            print(format_results(results))
            
            # Generate RAG response if we have results
            if results:
                print("\n=== Generated Answer ===")
                rag_response, media_refs = create_rag_response(
                    query, 
                    results, 
                    anthropic_client
                )
                print(rag_response)
                
                # Display media references if available
                if media_refs:
                    print("\n=== Referenced Media ===")
                    for ref in media_refs:
                        print(f"- {ref.get('id')}: {ref.get('description')}")
                        if 'static_path' in ref and ref['static_path']:
                            print(f"  Path: {ref['static_path']}")

if __name__ == "__main__":
    main()