import os
import json
import numpy as np
import anthropic
import voyageai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import logging
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.weaviate_client import WeaviateClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextualEmbeddings:
    """
    Generate contextual embeddings for document chunks and store them in Weaviate.
    """
    def __init__(self, weaviate_url="http://localhost:8081", class_name="ModalAIDocument", 
                 voyage_api_key=None, anthropic_api_key=None):
        """
        Initialize the contextual embeddings generator.
        
        Args:
            weaviate_url: URL of the Weaviate instance
            class_name: Name of the Weaviate class to use
            voyage_api_key: API key for Voyage AI
            anthropic_api_key: API key for Anthropic
        """
        # Load API keys from environment if not provided
        if voyage_api_key is None:
            voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Initialize clients
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Connect to Weaviate
        self.weaviate_db = WeaviateClient(class_name=class_name, url=weaviate_url)
        
        # For caching query embeddings
        self.query_cache = {}
        
        # Track token usage
        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }
        self.token_lock = threading.Lock()

    def situate_context(self, doc, chunk):
        """
        Generate contextual information for a chunk using Claude.
        
        Args:
            doc: Full document content
            chunk: Content of the chunk
            
        Returns:
            Tuple of (context text, usage statistics)
        """
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        try:
            response = self.anthropic_client.beta.prompt_caching.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.0,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                                "cache_control": {"type": "ephemeral"}
                            },
                            {
                                "type": "text",
                                "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                            },
                        ]
                    },
                ],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )
            
            return response.content[0].text, response.usage
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return "", None

    def process_and_store(self, dataset_path="data/modalai_chunks.json", parallel_threads=5):
        """
        Process documents with contextual embeddings and store in Weaviate.
        
        Args:
            dataset_path: Path to the dataset file
            parallel_threads: Number of threads for parallel processing
        """
        # Check if we already have data in Weaviate
        count = self.weaviate_db.count_objects()
        if count > 0:
            logger.info(f"Found {count} documents in Weaviate. Skipping processing.")
            return
        
        # Load dataset
        try:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
                logger.info(f"Loaded {len(dataset)} documents from {dataset_path}")
        except Exception as e:
            logger.error(f"Error loading dataset from {dataset_path}: {e}")
            return
        
        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc['chunks']) for doc in dataset)

        def process_chunk(doc, chunk):
            # Generate contextual information for this chunk
            contextualized_text, usage = self.situate_context(doc['content'], chunk['content'])
            
            if usage:
                with self.token_lock:
                    self.token_counts['input'] += usage.input_tokens
                    self.token_counts['output'] += usage.output_tokens
                    self.token_counts['cache_read'] += usage.cache_read_input_tokens
                    self.token_counts['cache_creation'] += usage.cache_creation_input_tokens
            
            return {
                # Combine original content with contextual information
                'text_to_embed': f"{chunk['content']}\n\n{contextualized_text}",
                'metadata': {
                    'doc_id': doc['doc_id'],
                    'original_uuid': doc['original_uuid'],
                    'chunk_id': chunk['chunk_id'],
                    'original_index': chunk['original_index'],
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'original_content': chunk['content'],
                    'contextualized_content': contextualized_text
                }
            }

        logger.info(f"Processing {total_chunks} chunks with {parallel_threads} threads")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for doc in dataset:
                for chunk in doc['chunks']:
                    futures.append(executor.submit(process_chunk, doc, chunk))
            
            for future in tqdm(as_completed(futures), total=total_chunks, desc="Processing chunks"):
                result = future.result()
                texts_to_embed.append(result['text_to_embed'])
                metadata.append(result['metadata'])

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self._create_embeddings(texts_to_embed)
        
        # Store in Weaviate
        logger.info("Storing embeddings in Weaviate...")
        self.weaviate_db.store_embeddings(texts_to_embed, embeddings, metadata)

        # Log token usage statistics
        logger.info(f"Contextual embeddings processed and stored. Total chunks: {len(texts_to_embed)}")
        logger.info(f"Total input tokens without caching: {self.token_counts['input']}")
        logger.info(f"Total output tokens: {self.token_counts['output']}")
        logger.info(f"Total input tokens written to cache: {self.token_counts['cache_creation']}")
        logger.info(f"Total input tokens read from cache: {self.token_counts['cache_read']}")
        
        total_tokens = self.token_counts['input'] + self.token_counts['cache_read'] + self.token_counts['cache_creation']
        savings_percentage = (self.token_counts['cache_read'] / total_tokens) * 100 if total_tokens > 0 else 0
        logger.info(f"Total input token savings from prompt caching: {savings_percentage:.2f}% of all input tokens used were read from cache.")

    def _create_embeddings(self, texts):
        """
        Generate embeddings for texts using VoyageAI.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        batch_size = 128
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            batch_result = self.voyage_client.embed(batch, model="voyage-2").embeddings
            all_embeddings.extend(batch_result)
            time.sleep(0.1)  # Rate limiting
            
        return all_embeddings

    def search(self, query, k=20):
        """
        Search for relevant documents using vector similarity.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of search results
        """
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.voyage_client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        # Search in Weaviate
        return self.weaviate_db.search(query_embedding, k=k)


def main():
    """Main entry point for contextual embeddings."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate contextual embeddings for Modal AI documentation")
    parser.add_argument("--input", default="data/modalai_chunks.json", help="Input JSON file with chunks")
    parser.add_argument("--threads", type=int, default=5, help="Number of threads for parallel processing")
    parser.add_argument("--weaviate-url", default="http://localhost:8081", help="Weaviate URL")
    parser.add_argument("--class-name", default="ModalAIDocument", help="Weaviate class name")
    
    args = parser.parse_args()
    
    processor = ContextualEmbeddings(
        weaviate_url=args.weaviate_url,
        class_name=args.class_name
    )
    processor.process_and_store(dataset_path=args.input, parallel_threads=args.threads)
    
    # Test search
    results = processor.search("VOXL Flight configuration", k=3)
    
    print("\nSearch Results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}: {result['metadata'].get('title', 'No title')}")
        print(f"URL: {result['metadata'].get('url', 'No URL')}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Content: {result['metadata']['original_content'][:200]}...")

if __name__ == "__main__":
    main()