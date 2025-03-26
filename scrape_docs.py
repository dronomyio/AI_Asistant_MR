import os
import sys
import argparse
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the enhanced scraper
from src.scraper.modal_ai_scraper import ModalAIScraper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the scraper.
    """
    parser = argparse.ArgumentParser(description="Scrape Modal AI documentation")
    parser.add_argument("--url", default="https://docs.modalai.com", help="Base URL to start scraping")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum number of pages to scrape")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of concurrent download workers")
    parser.add_argument("--download-media", action="store_true", help="Download media files (images, videos, documents)")
    parser.add_argument("--docs-file", default="modalai_docs.json", help="Output JSON file for documents")
    parser.add_argument("--media-catalog", default="modalai_media.json", help="Output JSON file for media catalog")
    
    args = parser.parse_args()
    
    logger.info(f"Starting scraper with base URL: {args.url}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Media downloading: {'enabled' if args.download_media else 'disabled'}")
    
    # Create scraper
    scraper = ModalAIScraper(base_url=args.url, output_dir=args.output)
    
    if not args.download_media:
        # If not downloading media, empty the file extensions list
        scraper.file_extensions = []
    
    # Start crawling
    logger.info("Starting crawl...")
    scraper.crawl(max_pages=args.max_pages, max_workers=args.max_workers)
    
    # Save results
    logger.info("Saving results...")
    scraper.save_docs(filename=args.docs_file, media_catalog_file=args.media_catalog)
    
    logger.info("Scraping completed successfully!")

if __name__ == "__main__":
    main()