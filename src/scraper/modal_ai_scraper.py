import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModalAIScraper:
    """
    Scraper for Modal AI documentation website.
    """
    def __init__(self, base_url="https://docs.modalai.com", output_dir="data"):
        """
        Initialize the scraper.
        
        Args:
            base_url: Base URL of the documentation site
            output_dir: Directory to save scraped data
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.visited_urls = set()
        self.docs = []
        self.max_retries = 3
        self.delay = 1  # seconds between requests
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def is_valid_url(self, url):
        """
        Check if URL is within the documentation domain.
        
        Args:
            url: URL to check
            
        Returns:
            Boolean indicating if URL is valid
        """
        parsed_url = urlparse(url)
        return parsed_url.netloc == urlparse(self.base_url).netloc and "docs.modalai.com" in url

    def clean_text(self, text):
        """
        Clean up text by removing excessive whitespace.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        return " ".join(text.split())

    def extract_content(self, soup, url):
        """
        Extract the main content from the page.
        
        Args:
            soup: BeautifulSoup object for the page
            url: URL of the page
            
        Returns:
            Dictionary with extracted content
        """
        # Target the main content area - the "main-content" div is used in the ModalAI docs
        main_content = soup.find("div", id="main-content") or soup.find("div", class_="main-content")
        
        if not main_content:
            # Fallback to other common content selectors
            main_content = soup.find("div", class_="content") or soup.find("main") or soup.find("article")
            
        if not main_content:
            logger.warning(f"No main content found at {url}")
            return {"title": "", "content": "", "headings": []}
        
        # Extract title
        title_elem = main_content.find("h1") or soup.find("h1")
        title = self.clean_text(title_elem.text) if title_elem else ""
        
        # Extract all headings for structure
        headings = []
        for h_tag in main_content.find_all(["h1", "h2", "h3", "h4"]):
            headings.append({
                "level": int(h_tag.name[1]),
                "text": self.clean_text(h_tag.text)
            })
        
        # Extract paragraphs and code blocks
        paragraphs = []
        for p in main_content.find_all(["p", "pre", "code", "ul", "ol", "div.language-bash"]):
            paragraphs.append(self.clean_text(p.text))
        
        content = "\n\n".join(paragraphs)
        
        return {
            "title": title,
            "content": content,
            "headings": headings
        }

    def get_page(self, url):
        """
        Fetch a page with retries.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content of the page or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))  # Exponential backoff
                else:
                    return None

    def extract_links(self, soup, current_url):
        """
        Extract all links from the page.
        
        Args:
            soup: BeautifulSoup object for the page
            current_url: URL of the current page
            
        Returns:
            List of extracted links
        """
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            absolute_url = urljoin(current_url, href)
            
            # Filter out external links, anchors, etc.
            if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                links.append(absolute_url)
        
        return links

    def crawl(self, start_url=None, max_pages=None):
        """
        Crawl the documentation site starting from the given URL.
        
        Args:
            start_url: Starting URL (defaults to base_url)
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of crawled documents
        """
        if start_url is None:
            start_url = self.base_url
        
        queue = [start_url]
        page_count = 0
        
        with tqdm(desc="Crawling pages", unit="page") as pbar:
            while queue and (max_pages is None or page_count < max_pages):
                current_url = queue.pop(0)
                
                if current_url in self.visited_urls:
                    continue
                
                self.visited_urls.add(current_url)
                
                # Fetch the page
                html = self.get_page(current_url)
                if not html:
                    continue
                
                # Parse HTML
                soup = BeautifulSoup(html, "html.parser")
                
                # Extract content
                content_data = self.extract_content(soup, current_url)
                if content_data["content"]:
                    self.docs.append({
                        "url": current_url,
                        "title": content_data["title"],
                        "content": content_data["content"],
                        "headings": content_data["headings"]
                    })
                    page_count += 1
                    pbar.update(1)
                
                # Extract links and add to queue
                links = self.extract_links(soup, current_url)
                queue.extend(links)
                
                # Be nice to the server
                time.sleep(self.delay)
        
        logger.info(f"Crawled {len(self.docs)} pages.")
        return self.docs

    def save_docs(self, filename="modalai_docs.json"):
        """
        Save the crawled documents to JSON files.
        
        Args:
            filename: Name of the output file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Save all documents in a single file
        with open(output_path, "w") as f:
            json.dump(self.docs, f, indent=2)
        
        logger.info(f"Saved {len(self.docs)} documents to {output_path}")


def main():
    """Main entry point for the scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape Modal AI documentation")
    parser.add_argument("--url", default="https://docs.modalai.com", help="Base URL to start scraping")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum number of pages to scrape")
    
    args = parser.parse_args()
    
    scraper = ModalAIScraper(base_url=args.url, output_dir=args.output)
    scraper.crawl(max_pages=args.max_pages)
    scraper.save_docs()

if __name__ == "__main__":
    main()