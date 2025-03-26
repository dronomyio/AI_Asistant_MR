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
    dirs = ["data", "data/media", "app/static", "app/templates"]
    for dir_path in dirs:
        Path(BASE_DIR / dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created")

def run_scraper(args):
    """Run the web scraper to download Modal AI documentation."""
    from src.scraper.modal_ai_scraper import ModalAIScraper
    
    logger.info("Starting Modal AI documentation scraper...")
    scraper = ModalAIScraper(output_dir=str(BASE_DIR / "data"))
    
    # Configure media downloading if requested
    if args.download_media:
        logger.info("Media downloading enabled")
        # File extensions are already set in the scraper
    else:
        # Disable media downloading
        scraper.file_extensions = []
    
    # Start crawling with specified parameters
    scraper.crawl(
        max_pages=args.max_pages,
        max_workers=args.max_workers
    )
    
    # Save the results
    scraper.save_docs(
        filename=args.docs_file,
        media_catalog_file=args.media_catalog
    )
    
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
        media_catalog_file=args.media_catalog,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    logger.info("Document processing completed")

def run_media_processor(args):
    """Process media files using Unstructured.io."""
    from src.processor.media_processor import MediaProcessor
    
    # Check if media catalog exists
    media_catalog_path = BASE_DIR / "data" / args.catalog
    if not os.path.exists(media_catalog_path):
        logger.error(f"Media catalog not found at {media_catalog_path}")
        logger.info("Please run the scraper with --download-media first")
        return False
    
    logger.info("Processing media files with Unstructured.io...")
    processor = MediaProcessor(
        media_dir=str(BASE_DIR / "data" / "media"),
        output_dir=str(BASE_DIR / "data")
    )
    
    processor.process_and_save(
        catalog_file=args.catalog,
        output_file=args.output
    )
    
    logger.info("Media processing completed")
    return True

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
    scraper_parser.add_argument("--download-media", action="store_true", help="Download media files (images, videos, documents)")
    scraper_parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of concurrent download workers")
    scraper_parser.add_argument("--docs-file", default="modalai_docs.json", help="Output JSON file for documents")
    scraper_parser.add_argument("--media-catalog", default="modalai_media.json", help="Output JSON file for media catalog")
    
    # Processor command
    processor_parser = subparsers.add_parser("process", help="Process documents into chunks")
    processor_parser.add_argument("--input", default="modalai_docs.json", help="Input JSON file")
    processor_parser.add_argument("--output", default="modalai_chunks.json", help="Output JSON file")
    processor_parser.add_argument("--media-catalog", default="modalai_media.json", help="Media catalog JSON file")
    processor_parser.add_argument("--chunk-size", type=int, default=800, help="Size of each chunk in characters")
    processor_parser.add_argument("--overlap", type=int, default=100, help="Overlap between chunks in characters")
    
    # Media processor command
    media_parser = subparsers.add_parser("process-media", help="Process media files using Unstructured.io")
    media_parser.add_argument("--catalog", default="modalai_media.json", help="Media catalog JSON file")
    media_parser.add_argument("--output", default="modalai_processed_media.json", help="Output JSON file for processed media")
    
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
    pipeline_parser.add_argument("--download-media", action="store_true", help="Download and process media files")
    pipeline_parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of concurrent download workers")
    pipeline_parser.add_argument("--process-media", action="store_true", help="Process media files with Unstructured.io")
    
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
    elif args.command == "process-media":
        run_media_processor(args)
    elif args.command == "embed":
        run_embeddings(args)
    elif args.command == "retrieve":
        run_retrieval(args)
    elif args.command == "chat":
        run_chat(args)
    elif args.command == "pipeline":
        # Run the full pipeline
        scrape_args = argparse.Namespace(
            max_pages=args.max_pages,
            download_media=args.download_media,
            max_workers=args.max_workers,
            docs_file="modalai_docs.json",
            media_catalog="modalai_media.json"
        )
        
        process_args = argparse.Namespace(
            input="modalai_docs.json",
            output="modalai_chunks.json",
            media_catalog="modalai_media.json",
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        
        media_args = argparse.Namespace(
            catalog="modalai_media.json",
            output="modalai_processed_media.json"
        )
        
        embed_args = argparse.Namespace(
            input="modalai_chunks.json",
            threads=args.threads,
            weaviate_url=args.weaviate_url,
            class_name="ModalAIDocument"
        )
        
        # Run the pipeline steps
        run_scraper(scrape_args)
        run_processor(process_args)
        
        # Process media files if requested
        if args.download_media and args.process_media:
            run_media_processor(media_args)
        
        run_embeddings(embed_args)
        run_chat(args)
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())