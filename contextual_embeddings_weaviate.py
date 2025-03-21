import os
import json
import pickle
import numpy as np
import anthropic
import voyageai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
from weaviate_db import WeaviateVectorDB

class ContextualEmbeddings:
    def __init__(self, class_name="ModalAIDocument", voyage_api_key=None, anthropic_api_key=None):
        if voyage_api_key is None:
            voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Connect to Weaviate
        self.weaviate_db = WeaviateVectorDB(class_name=class_name)
        
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
        """Generate contextual information for a chunk using Claude."""
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
            print(f"Error generating context: {e}")
            return "", None

    def process_and_store(self, dataset, parallel_threads=5):
        """Process documents with contextual embeddings and store in Weaviate."""
        # Check if we already have data in Weaviate
        count = self.weaviate_db.count_objects()
        if count > 0:
            print(f"Found {count} documents in Weaviate. Skipping processing.")
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

        print(f"Processing {total_chunks} chunks with {parallel_threads} threads")
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
        embeddings = self._create_embeddings(texts_to_embed)
        
        # Store in Weaviate
        self.weaviate_db.store_embeddings(texts_to_embed, embeddings, metadata)

        # Log token usage statistics
        print(f"Contextual embeddings processed and stored. Total chunks: {len(texts_to_embed)}")
        print(f"Total input tokens without caching: {self.token_counts['input']}")
        print(f"Total output tokens: {self.token_counts['output']}")
        print(f"Total input tokens written to cache: {self.token_counts['cache_creation']}")
        print(f"Total input tokens read from cache: {self.token_counts['cache_read']}")
        
        total_tokens = self.token_counts['input'] + self.token_counts['cache_read'] + self.token_counts['cache_creation']
        savings_percentage = (self.token_counts['cache_read'] / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"Total input token savings from prompt caching: {savings_percentage:.2f}% of all input tokens used were read from cache.")

    def _create_embeddings(self, texts):
        """Generate embeddings for texts using VoyageAI."""
        batch_size = 128
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            batch_result = self.voyage_client.embed(batch, model="voyage-2").embeddings
            all_embeddings.extend(batch_result)
            time.sleep(0.1)  # Rate limiting
            
        return all_embeddings

    def search(self, query, k=20):
        """Search for relevant documents using vector similarity."""
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.voyage_client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        # Search in Weaviate
        return self.weaviate_db.search(query_embedding, k=k)


def main():
    # Load chunked documents
    with open("data/modalai_chunks.json", "r") as f:
        chunks = json.load(f)
    
    # Initialize and build the contextual vector database
    processor = ContextualEmbeddings()
    processor.process_and_store(chunks, parallel_threads=5)
    
    # Test search functionality
    results = processor.search("How to set up VOXL Flight?", k=3)
    
    print("\nSearch Results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}: {result['metadata'].get('title', 'No title')}")
        print(f"URL: {result['metadata'].get('url', 'No URL')}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Content: {result['metadata']['original_content'][:200]}...")

if __name__ == "__main__":
    main()