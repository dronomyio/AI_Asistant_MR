import os
import json
import uuid
from tqdm import tqdm

def chunk_document(doc, chunk_size=800, overlap=100):
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

def process_documents(input_file, output_file):
    """
    Process documents from input file, chunk them, and save to output file.
    
    Args:
        input_file: Path to input JSON file with raw documents
        output_file: Path to output JSON file for processed documents
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load documents
    with open(input_file, 'r') as f:
        docs = json.load(f)
    
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
            "chunks": chunk_document({**doc, 'doc_id': doc_id})
        }
        
        processed_docs.append(processed_doc)
    
    # Save processed documents
    with open(output_file, 'w') as f:
        json.dump(processed_docs, f, indent=2)
    
    print(f"Processed {len(processed_docs)} documents with {sum(len(doc['chunks']) for doc in processed_docs)} chunks")
    return processed_docs

if __name__ == "__main__":
    process_documents("data/modalai_docs.json", "data/modalai_chunks.json")