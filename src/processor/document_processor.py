import os
import json
import uuid
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Process documents into chunks for embedding and retrieval.
    """
    def __init__(self, input_dir="data", output_dir="data"):
        """
        Initialize the document processor.
        
        Args:
            input_dir: Directory with input files
            output_dir: Directory for output files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def chunk_document(self, doc, chunk_size=800, overlap=100):
        """
        Split a document into chunks of specified size with overlap.
        
        Args:
            doc: Document dictionary with 'content' field
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        content = doc['content']
        chunks = []
        
        # Simple character-based chunking
        start = 0
        chunk_id = 0
        
        while start < len(content):
            # Calculate end position for this chunk
            end = min(start + chunk_size, len(content))
            
            # If this is not the last chunk, try to find a good break point
            if end < len(content):
                # Look for paragraph break
                paragraph_break = content.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break + 2  # Include the paragraph break
                else:
                    # Look for sentence break
                    sentence_break = max(
                        content.rfind('. ', start, end),
                        content.rfind('! ', start, end),
                        content.rfind('? ', start, end)
                    )
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2  # Include the period and space
            
            # Extract the chunk
            chunk_text = content[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_chunk_{chunk_id}",
                    "original_index": chunk_id,
                    "content": chunk_text
                })
                chunk_id += 1
            
            # Move start position for next chunk, accounting for overlap
            start = end - overlap if end < len(content) else len(content)
        
        return chunks
    
    def process_documents(self, input_file="modalai_docs.json", output_file="modalai_chunks.json", chunk_size=800, overlap=100):
        """
        Process documents from input file, chunk them, and save to output file.
        
        Args:
            input_file: Name of input JSON file with raw documents
            output_file: Name of output JSON file for processed documents
            chunk_size: Size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of processed document dictionaries
        """
        input_path = os.path.join(self.input_dir, input_file)
        output_path = os.path.join(self.output_dir, output_file)
        
        # Load documents
        try:
            with open(input_path, 'r') as f:
                docs = json.load(f)
                logger.info(f"Loaded {len(docs)} documents from {input_path}")
        except Exception as e:
            logger.error(f"Error loading documents from {input_path}: {e}")
            return []
        
        processed_docs = []
        
        for i, doc in enumerate(tqdm(docs, desc="Processing documents")):
            doc_id = f"doc_{i}"
            doc_uuid = str(uuid.uuid4())
            
            # Create document object
            processed_doc = {
                "doc_id": doc_id,
                "original_uuid": doc_uuid,
                "title": doc.get('title', ''),
                "url": doc.get('url', ''),
                "content": doc.get('content', ''),
                "chunks": self.chunk_document({**doc, 'doc_id': doc_id}, chunk_size, overlap)
            }
            
            processed_docs.append(processed_doc)
        
        # Save processed documents
        try:
            with open(output_path, 'w') as f:
                json.dump(processed_docs, f, indent=2)
                
            total_chunks = sum(len(doc['chunks']) for doc in processed_docs)
            logger.info(f"Processed {len(processed_docs)} documents with {total_chunks} chunks")
            logger.info(f"Saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed documents to {output_path}: {e}")
        
        return processed_docs


def main():
    """Main entry point for document processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Modal AI documentation into chunks")
    parser.add_argument("--input", default="modalai_docs.json", help="Input JSON file")
    parser.add_argument("--output", default="modalai_chunks.json", help="Output JSON file")
    parser.add_argument("--chunk-size", type=int, default=800, help="Size of each chunk in characters")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between chunks in characters")
    parser.add_argument("--input-dir", default="data", help="Input directory")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    
    args = parser.parse_args()
    
    processor = DocumentProcessor(input_dir=args.input_dir, output_dir=args.output_dir)
    processor.process_documents(
        input_file=args.input, 
        output_file=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )

if __name__ == "__main__":
    main()