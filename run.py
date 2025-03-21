#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

def check_environment():
    """Check if necessary environment variables are set."""
    required_keys = ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "COHERE_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        logger.info("Please set the following environment variables:")
        for key in missing_keys:
            logger.info(f"  export {key}=your_{key.lower()}")
        return False
    
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = ["data", "app/static", "app/templates"]
    for dir_path in dirs:
        Path(BASE_DIR / dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created")

def run_scraper(args):
    """Run the web scraper to download Modal AI documentation."""
    from src.scraper.modal_ai_scraper import ModalAIScraper
    
    logger.info("Starting Modal AI documentation scraper...")
    scraper = ModalAIScraper(output_dir=str(BASE_DIR / "data"))
    scraper.crawl(max_pages=args.max_pages)
    scraper.save_docs()
    logger.info("Scraping completed")

def run_processor(args):
    """Process documents into chunks."""
    from src.processor.document_processor import DocumentProcessor
    
    logger.info("Processing documents into chunks...")
    processor = DocumentProcessor(
        input_dir=str(BASE_DIR / "data"),
        output_dir=str(BASE_DIR / "data")
    )
    processor.process_documents(
        input_file=args.input,
        output_file=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    logger.info("Document processing completed")

def run_embeddings(args):
    """Generate contextual embeddings and store in Weaviate."""
    from src.embeddings.contextual_embeddings import ContextualEmbeddings
    
    logger.info("Generating contextual embeddings...")
    embeddings = ContextualEmbeddings(
        weaviate_url=args.weaviate_url,
        class_name=args.class_name
    )
    embeddings.process_and_store(
        dataset_path=str(BASE_DIR / "data" / args.input),
        parallel_threads=args.threads
    )
    logger.info("Embedding generation completed")

def run_retrieval(args):
    """Run the retrieval system."""
    from src.retrieval.advanced_retrieval import main as retrieval_main
    
    logger.info("Starting retrieval system...")
    sys.argv = [
        sys.argv[0],
        "--weaviate-url", args.weaviate_url,
        "--elastic-url", args.elastic_url
    ]
    if args.query:
        sys.argv.extend(["--query", args.query])
    
    retrieval_main()

def run_chat(args):
    """Run the chat server."""
    logger.info("Starting chat server...")
    os.chdir(BASE_DIR / "app")
    subprocess.run([sys.executable, "chat_server.py"])

def main():
    """Main entry point for the program."""
    parser = argparse.ArgumentParser(description="Modal AI Documentation Retrieval System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Scraper command
    scraper_parser = subparsers.add_parser("scrape", help="Scrape Modal AI documentation")
    scraper_parser.add_argument("--max-pages", type=int, default=200, help="Maximum number of pages to scrape")
    
    # Processor command
    processor_parser = subparsers.add_parser("process", help="Process documents into chunks")
    processor_parser.add_argument("--input", default="modalai_docs.json", help="Input JSON file")
    processor_parser.add_argument("--output", default="modalai_chunks.json", help="Output JSON file")
    processor_parser.add_argument("--chunk-size", type=int, default=800, help="Size of each chunk in characters")
    processor_parser.add_argument("--overlap", type=int, default=100, help="Overlap between chunks in characters")
    
    # Embeddings command
    embeddings_parser = subparsers.add_parser("embed", help="Generate contextual embeddings")
    embeddings_parser.add_argument("--input", default="modalai_chunks.json", help="Input chunks file")
    embeddings_parser.add_argument("--threads", type=int, default=5, help="Number of parallel threads")
    embeddings_parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate URL")
    embeddings_parser.add_argument("--class-name", default="ModalAIDocument", help="Weaviate class name")
    
    # Retrieval command
    retrieval_parser = subparsers.add_parser("retrieve", help="Run retrieval system")
    retrieval_parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate URL")
    retrieval_parser.add_argument("--elastic-url", default="http://localhost:9200", help="Elasticsearch URL")
    retrieval_parser.add_argument("--query", help="Optional query to run (otherwise interactive mode)")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start chat server")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full pipeline")
    pipeline_parser.add_argument("--max-pages", type=int, default=200, help="Maximum number of pages to scrape")
    pipeline_parser.add_argument("--chunk-size", type=int, default=800, help="Size of each chunk in characters")
    pipeline_parser.add_argument("--overlap", type=int, default=100, help="Overlap between chunks in characters")
    pipeline_parser.add_argument("--threads", type=int, default=5, help="Number of parallel threads for embedding")
    pipeline_parser.add_argument("--weaviate-url", default="http://localhost:8080", help="Weaviate URL")
    pipeline_parser.add_argument("--elastic-url", default="http://localhost:9200", help="Elasticsearch URL")
    
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        return 1
    
    # Create directories
    create_directories()
    
    # Run the specified command
    if args.command == "scrape":
        run_scraper(args)
    elif args.command == "process":
        run_processor(args)
    elif args.command == "embed":
        run_embeddings(args)
    elif args.command == "retrieve":
        run_retrieval(args)
    elif args.command == "chat":
        run_chat(args)
    elif args.command == "pipeline":
        # Run the full pipeline
        scrape_args = argparse.Namespace(max_pages=args.max_pages)
        process_args = argparse.Namespace(
            input="modalai_docs.json",
            output="modalai_chunks.json",
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        embed_args = argparse.Namespace(
            input="modalai_chunks.json",
            threads=args.threads,
            weaviate_url=args.weaviate_url,
            class_name="ModalAIDocument"
        )
        
        run_scraper(scrape_args)
        run_processor(process_args)
        run_embeddings(embed_args)
        run_chat(args)
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())