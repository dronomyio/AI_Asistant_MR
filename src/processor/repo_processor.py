import os
import json
import logging
import re
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RepoProcessor:
    """
    Process local git repositories for documentation and code.
    """
    def __init__(self, repos_dir, output_dir="data"):
        """
        Initialize the repository processor.
        
        Args:
            repos_dir: Directory containing the cloned repositories
            output_dir: Directory to save processed data
        """
        self.repos_dir = Path(repos_dir)
        self.output_dir = Path(output_dir)
        self.media_dir = self.output_dir / "media"
        self.docs = []
        self.media_files = {}
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.media_dir.mkdir(exist_ok=True)
        
        # File extensions to process
        self.doc_extensions = ['.md', '.mdx', '.txt', '.rst', '.html']
        self.code_extensions = ['.c', '.cpp', '.h', '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.json', '.xml', '.yaml', '.yml']
        self.media_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.mp4', '.webm', '.webp']
        self.binary_extensions = ['.bin', '.so', '.dll', '.exe', '.zip', '.tar.gz']

    def clean_text(self, text):
        """Clean and normalize text content."""
        if not text:
            return ""
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Replace multiple newlines with a double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def process_markdown(self, file_path):
        """Process a markdown file and extract content."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Extract title from frontmatter or first heading
            title_match = re.search(r'^---\s*\n.*?title:\s*([^\n]+).*?\n---', content, re.DOTALL)
            if title_match:
                title = title_match.group(1).strip().strip('"\'')
            else:
                title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else os.path.basename(file_path)
            
            # Extract headings
            headings = []
            for match in re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append({"level": level, "text": text})
            
            # Find image references
            media_refs = []
            for match in re.finditer(r'!\[(.*?)\]\((.*?)\)', content):
                alt_text = match.group(1)
                image_path = match.group(2)
                
                # Resolve relative paths
                if not image_path.startswith(('http://', 'https://')):
                    full_path = os.path.normpath(os.path.join(os.path.dirname(file_path), image_path))
                    if os.path.exists(full_path):
                        rel_path = os.path.relpath(full_path, self.repos_dir)
                        media_refs.append({
                            "type": "image",
                            "path": rel_path,
                            "alt_text": alt_text,
                            "original_path": image_path
                        })
            
            return {
                "title": title,
                "content": self.clean_text(content),
                "file_path": str(file_path),
                "rel_path": os.path.relpath(file_path, self.repos_dir),
                "headings": headings,
                "media_references": media_refs
            }
        except Exception as e:
            logger.error(f"Error processing markdown file {file_path}: {e}")
            return None

    def process_code_file(self, file_path):
        """Process a code file and extract content."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Extract file name as title
            title = os.path.basename(file_path)
            
            # Extract comments as "documentation"
            comments = []
            
            # Handle different comment styles based on file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ['.py']:
                # Extract Python docstrings
                docstring_pattern = r'"""(.*?)"""'
                comments = re.findall(docstring_pattern, content, re.DOTALL)
                
                # Extract Python comments (single line)
                single_comments = re.findall(r'#\s*(.*?)$', content, re.MULTILINE)
                comments.extend(single_comments)
                
            elif ext in ['.c', '.cpp', '.h', '.java', '.js', '.ts', '.jsx', '.tsx']:
                # Extract C-style multi-line comments
                multi_comments = re.findall(r'/\*(.*?)\*/', content, re.DOTALL)
                comments.extend(multi_comments)
                
                # Extract C-style single-line comments
                single_comments = re.findall(r'//\s*(.*?)$', content, re.MULTILINE)
                comments.extend(single_comments)
            
            # Clean up comments
            comments = [self.clean_text(comment) for comment in comments if comment.strip()]
            comments_text = "\n\n".join(comments)
            
            return {
                "title": title,
                "content": content,
                "comments": comments_text,
                "file_path": str(file_path),
                "rel_path": os.path.relpath(file_path, self.repos_dir),
                "type": "code"
            }
        except Exception as e:
            logger.error(f"Error processing code file {file_path}: {e}")
            return None

    def copy_media_file(self, file_path):
        """
        Copy a media file to the media directory and return its info.
        """
        try:
            rel_path = os.path.relpath(file_path, self.repos_dir)
            dest_path = self.media_dir / rel_path
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Copy the file if it doesn't exist
            if not os.path.exists(dest_path):
                with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())
            
            # Get file size
            file_size = os.path.getsize(dest_path)
            
            # Get file extension
            _, ext = os.path.splitext(file_path)
            
            # Determine media type
            if ext.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                media_type = 'image'
            elif ext.lower() in ['.mp4', '.webm', '.mov']:
                media_type = 'video'
            elif ext.lower() == '.pdf':
                media_type = 'pdf'
            elif ext.lower() == '.svg':
                media_type = 'svg'
            else:
                media_type = 'unknown'
            
            return {
                "original_path": str(file_path),
                "rel_path": rel_path,
                "path": str(dest_path),
                "size": file_size,
                "type": media_type
            }
        except Exception as e:
            logger.error(f"Error copying media file {file_path}: {e}")
            return None

    def should_process_file(self, file_path):
        """Determine whether a file should be processed based on its path and extension."""
        # Skip hidden files and directories
        if os.path.basename(file_path).startswith('.'):
            return False
        
        # Skip node_modules, venv, and similar directories
        parts = Path(file_path).parts
        skip_dirs = ['node_modules', 'venv', '.git', '__pycache__', 'build', 'dist']
        if any(part in skip_dirs for part in parts):
            return False
        
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Skip binary files that aren't media
        if ext in self.binary_extensions:
            return False
        
        # Process documentation, code, and media files
        return (ext in self.doc_extensions or 
                ext in self.code_extensions or 
                ext in self.media_extensions)

    def process_repository(self, repo_path, max_files=None):
        """
        Process a repository and extract all relevant files.
        
        Args:
            repo_path: Path to the repository
            max_files: Maximum number of files to process
            
        Returns:
            List of processed documents
        """
        repo_dir = Path(repo_path)
        if not repo_dir.exists() or not repo_dir.is_dir():
            logger.error(f"Repository path {repo_dir} does not exist or is not a directory")
            return []
        
        # Find all relevant files
        all_files = []
        for root, _, files in os.walk(repo_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if self.should_process_file(file_path):
                    all_files.append(file_path)
        
        # Limit the number of files if needed
        if max_files and len(all_files) > max_files:
            all_files = all_files[:max_files]
        
        logger.info(f"Processing {len(all_files)} files from repository {repo_dir.name}")
        
        # Process files with multiple threads
        docs = []
        media_files = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {}
            
            # Submit file processing tasks
            for file_path in all_files:
                _, ext = os.path.splitext(file_path)
                ext = ext.lower()
                
                if ext in self.doc_extensions:
                    future = executor.submit(self.process_markdown, file_path)
                    future_to_file[future] = file_path
                elif ext in self.code_extensions:
                    future = executor.submit(self.process_code_file, file_path)
                    future_to_file[future] = file_path
                elif ext in self.media_extensions:
                    future = executor.submit(self.copy_media_file, file_path)
                    future_to_file[future] = file_path
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                              total=len(future_to_file), 
                              desc=f"Processing {repo_dir.name}"):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        _, ext = os.path.splitext(file_path)
                        ext = ext.lower()
                        
                        if ext in self.media_extensions:
                            # Store media file info
                            media_files[result["rel_path"]] = result
                        else:
                            # Store document or code file
                            docs.append(result)
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
        
        # Update instance variables
        self.docs.extend(docs)
        self.media_files.update(media_files)
        
        logger.info(f"Processed {len(docs)} documents and {len(media_files)} media files from repository {repo_dir.name}")
        return docs

    def process_all_repositories(self, max_files_per_repo=None):
        """
        Process all repositories in the repos directory.
        
        Args:
            max_files_per_repo: Maximum number of files to process per repository
            
        Returns:
            List of all processed documents
        """
        all_docs = []
        
        # Find all repositories
        for item in os.listdir(self.repos_dir):
            repo_path = self.repos_dir / item
            if repo_path.is_dir() and (repo_path / '.git').exists():
                logger.info(f"Found git repository: {repo_path.name}")
                docs = self.process_repository(repo_path, max_files_per_repo)
                all_docs.extend(docs)
        
        return all_docs

    def chunk_documents(self, chunk_size=800, overlap=100):
        """
        Split documents into smaller chunks for embedding.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        for doc in self.docs:
            content = doc.get("content", "")
            if not content:
                continue
            
            # Handle different chunking strategies based on document type
            doc_type = doc.get("type", "documentation")
            
            # For code files, split by function/class/method if possible
            if doc_type == "code":
                code_chunks = self._chunk_code_document(doc, chunk_size)
                if code_chunks and len(code_chunks) > 0:
                    chunks.extend(code_chunks)
                    continue
            
            # For markdown docs, try to split by heading sections
            if doc.get("rel_path", "").endswith(('.md', '.mdx')):
                md_chunks = self._chunk_markdown_by_section(doc, chunk_size)
                if md_chunks and len(md_chunks) > 0:
                    chunks.extend(md_chunks)
                    continue
            
            # For other files or if special chunking failed, use standard chunking
            # Chunks will have at minimum 100 characters
            if len(content) < 100:
                chunks.append({
                    "content": content,
                    "title": doc.get("title", ""),
                    "file_path": doc.get("file_path", ""),
                    "rel_path": doc.get("rel_path", ""),
                    "type": doc_type,
                    "doc_id": doc.get("rel_path", ""),
                    "original_index": 0,
                    "chunk_index": 0,
                    "media_references": doc.get("media_references", [])
                })
                continue
            
            # Split content into chunks
            content_length = len(content)
            chunk_index = 0
            
            for i in range(0, content_length, chunk_size - overlap):
                chunk_start = i
                chunk_end = min(i + chunk_size, content_length)
                
                # Don't create small chunks at the end
                if content_length - i < chunk_size / 2 and len(chunks) > 0 and i > 0:
                    # Extend the previous chunk instead
                    chunks[-1]["content"] += content[i:]
                    break
                
                chunk_text = content[chunk_start:chunk_end]
                
                # Find any media references relevant to this chunk
                chunk_media_refs = []
                for media_ref in doc.get("media_references", []):
                    # Include media references with smarter matching
                    # If we have position information, we could check if it falls within chunk
                    chunk_media_refs.append(media_ref)
                
                # Add context information to make chunks more self-contained
                context_prefix = f"Document: {doc.get('title', '')}\n"
                if doc_type == "code":
                    context_prefix += f"File: {doc.get('rel_path', '')}\n"
                
                # Create chunk with context
                chunks.append({
                    "content": context_prefix + chunk_text,
                    "title": doc.get("title", ""),
                    "file_path": doc.get("file_path", ""),
                    "rel_path": doc.get("rel_path", ""),
                    "type": doc_type,
                    "doc_id": doc.get("rel_path", ""),
                    "original_index": 0,
                    "chunk_index": chunk_index,
                    "media_references": chunk_media_refs
                })
                chunk_index += 1
        
        return chunks
    
    def _chunk_markdown_by_section(self, doc, max_size=1000):
        """Split markdown document by headings into logical sections."""
        content = doc.get("content", "")
        headings = doc.get("headings", [])
        
        if not headings or not content:
            return None
            
        # Find positions of all headings in the content
        heading_positions = []
        
        for heading in headings:
            heading_text = heading.get("text", "")
            level = heading.get("level", 1)
            heading_marker = "#" * level
            pattern = f"^{heading_marker}\\s+{re.escape(heading_text)}$"
            
            for match in re.finditer(pattern, content, re.MULTILINE):
                heading_positions.append({
                    "start": match.start(),
                    "text": heading_text,
                    "level": level
                })
        
        # Sort by position
        heading_positions.sort(key=lambda x: x["start"])
        
        # No headings found with regex (might be in frontmatter)
        if not heading_positions:
            return None
            
        # Create chunks based on headings
        chunks = []
        for i, heading in enumerate(heading_positions):
            # Section start is at the heading
            start = heading["start"]
            
            # Section end is the next heading or the end of the document
            end = heading_positions[i+1]["start"] if i < len(heading_positions) - 1 else len(content)
            
            section_text = content[start:end]
            
            # If section is too large, use regular chunking
            if len(section_text) > max_size * 1.5:
                # Use regular chunking for this section
                pass
            else:
                # Get media references for this section
                section_media_refs = []
                for media_ref in doc.get("media_references", []):
                    section_media_refs.append(media_ref)
                
                chunks.append({
                    "content": section_text,
                    "title": f"{doc.get('title', '')} - {heading['text']}",
                    "file_path": doc.get("file_path", ""),
                    "rel_path": doc.get("rel_path", ""),
                    "type": "documentation",
                    "doc_id": doc.get("rel_path", ""),
                    "original_index": 0,
                    "chunk_index": i,
                    "media_references": section_media_refs,
                    "heading": heading["text"],
                    "heading_level": heading["level"]
                })
        
        return chunks if chunks else None
    
    def _chunk_code_document(self, doc, max_size=1000):
        """Split code document by functions/classes/methods."""
        content = doc.get("content", "")
        file_path = doc.get("rel_path", "")
        
        if not content:
            return []
            
        # Different patterns based on file extension
        if file_path.endswith('.py'):
            # Python function and class patterns
            function_pattern = r'(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*(?:->.*?)?:(?:.|\n)*?(?=\n\S|\Z))'
            class_pattern = r'(class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\([^)]*\))?\s*:(?:.|\n)*?(?=\n\S|\Z))'
            patterns = [function_pattern, class_pattern]
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            # JavaScript/TypeScript patterns
            function_pattern = r'(function\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*{(?:.|\n)*?})'
            arrow_func_pattern = r'(const\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*(?:\([^)]*\)|[a-zA-Z_][a-zA-Z0-9_]*)\s*=>\s*{(?:.|\n)*?})'
            class_pattern = r'(class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)?\s*{(?:.|\n)*?})'
            patterns = [function_pattern, arrow_func_pattern, class_pattern]
        elif file_path.endswith(('.c', '.cpp', '.h')):
            # C/C++ patterns
            function_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^;{]*\)\s*{(?:.|\n)*?})'
            class_pattern = r'(class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:\s*(?:public|protected|private)\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*(?:public|protected|private)\s+[a-zA-Z_][a-zA-Z0-9_]*)*)?\s*{(?:.|\n)*?};)'
            patterns = [function_pattern, class_pattern]
        else:
            # For other file types, use standard chunking
            return []
        
        # Extract all code blocks
        code_blocks = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                block_text = match.group(1)
                code_blocks.append({
                    "text": block_text,
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Sort by position
        code_blocks.sort(key=lambda x: x["start"])
        
        # If no code blocks found, fall back to standard chunking
        if not code_blocks:
            return []
        
        # Create chunks for each code block
        chunks = []
        for i, block in enumerate(code_blocks):
            # Get comments before this block to include as context
            start_context = max(0, block["start"] - 200)  # Look up to 200 chars before
            context = content[start_context:block["start"]]
            
            # Extract comments from the context
            comment_lines = []
            if file_path.endswith('.py'):
                for line in context.split('\n'):
                    line = line.strip()
                    if line.startswith('#'):
                        comment_lines.append(line[1:].strip())
            else:  # C-style comments
                for line in context.split('\n'):
                    line = line.strip()
                    if line.startswith('//'):
                        comment_lines.append(line[2:].strip())
                # Also extract multi-line comments
                for match in re.finditer(r'/\*(.*?)\*/', context, re.DOTALL):
                    comment_lines.append(match.group(1).strip())
            
            comments_text = "\n".join(comment_lines)
            
            # Identify block type and name
            if file_path.endswith('.py'):
                if block["text"].startswith('def '):
                    name = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', block["text"]).group(1)
                    block_type = "function"
                elif block["text"].startswith('class '):
                    name = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', block["text"]).group(1)
                    block_type = "class"
                else:
                    name = f"code_block_{i}"
                    block_type = "code"
            elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                if re.match(r'function\s+', block["text"]):
                    name = re.search(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)', block["text"]).group(1)
                    block_type = "function"
                elif re.match(r'const\s+.*\s*=\s*', block["text"]):
                    name = re.search(r'const\s+([a-zA-Z_][a-zA-Z0-9_]*)', block["text"]).group(1)
                    block_type = "function"
                elif re.match(r'class\s+', block["text"]):
                    name = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', block["text"]).group(1)
                    block_type = "class"
                else:
                    name = f"code_block_{i}"
                    block_type = "code"
            else:
                name = f"code_block_{i}"
                block_type = "code"
            
            # Create enriched content with title and comments
            enriched_content = f"File: {file_path}\n"
            if block_type == "function":
                enriched_content += f"Function: {name}\n"
            elif block_type == "class":
                enriched_content += f"Class: {name}\n"
            
            if comments_text:
                enriched_content += f"Comments:\n{comments_text}\n\n"
            
            enriched_content += f"Code:\n{block['text']}"
            
            # Create the chunk
            chunks.append({
                "content": enriched_content,
                "title": f"{doc.get('title', '')} - {name}",
                "file_path": doc.get("file_path", ""),
                "rel_path": file_path,
                "type": "code",
                "doc_id": doc.get("rel_path", ""),
                "original_index": 0,
                "chunk_index": i,
                "code_type": block_type,
                "code_name": name
            })
        
        return chunks

    def save_output(self, chunks_file="repo_chunks.json", media_catalog_file="repo_media.json"):
        """
        Save processed data to output files.
        
        Args:
            chunks_file: Name of output file for document chunks
            media_catalog_file: Name of output file for media catalog
        """
        # Chunk the documents
        chunks = self.chunk_documents() or []
        
        # Save chunks
        chunks_path = self.output_dir / chunks_file
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")
        
        # Save media catalog
        media_path = self.output_dir / media_catalog_file
        with open(media_path, 'w', encoding='utf-8') as f:
            json.dump(self.media_files, f, indent=2)
        
        logger.info(f"Saved media catalog with {len(self.media_files)} entries to {media_path}")

def main():
    """Command-line interface for the repository processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process git repositories for RAG")
    parser.add_argument("--repos-dir", default="data/repos", help="Directory containing repositories")
    parser.add_argument("--output-dir", default="data", help="Directory to save output files")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum files to process per repository")
    parser.add_argument("--chunks-file", default="repo_chunks.json", help="Output file for document chunks")
    parser.add_argument("--media-catalog", default="repo_media.json", help="Output file for media catalog")
    
    args = parser.parse_args()
    
    processor = RepoProcessor(repos_dir=args.repos_dir, output_dir=args.output_dir)
    processor.process_all_repositories(max_files_per_repo=args.max_files)
    processor.save_output(chunks_file=args.chunks_file, media_catalog_file=args.media_catalog)

if __name__ == "__main__":
    main()