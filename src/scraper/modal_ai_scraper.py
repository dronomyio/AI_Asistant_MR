import os
import json
import requests
import mimetypes
import hashlib
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from tqdm import tqdm
import time
import logging
import concurrent.futures
from pathlib import Path

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
        self.media_dir = os.path.join(output_dir, "media")
        self.visited_urls = set()
        self.docs = []
        self.max_retries = 3
        self.delay = 1  # seconds between requests
        self.downloaded_files = set()  # Track downloaded files to avoid duplicates
        
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.media_dir, exist_ok=True)
        
        # Supported file extensions to download
        self.file_extensions = [
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.tar', '.gz', '.stl', '.step', '.stp', '.dxf', '.igs'
        ]

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

    def generate_safe_filename(self, url):
        """
        Generate a safe filename from a URL.
        
        Args:
            url: URL to convert to a filename
            
        Returns:
            Safe filename with extension
        """
        # Extract the original filename from the URL
        path = urlparse(url).path
        orig_filename = os.path.basename(path)
        
        # Extract the extension
        name, ext = os.path.splitext(orig_filename)
        if not ext:
            # Try to guess extension from URL
            ext = mimetypes.guess_extension(mimetypes.guess_type(url)[0] or '') or ''
        
        # Create a SHA-256 hash of the URL to ensure uniqueness
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
        
        # Create a safe filename
        safe_name = re.sub(r'[^\w\-.]', '_', name)
        if not safe_name:
            safe_name = 'file'
            
        # Combine hash and original name
        if len(safe_name) > 50:
            safe_name = safe_name[:50]  # Truncate if too long
            
        safe_filename = f"{safe_name}_{url_hash}{ext}"
        return safe_filename
    
    def download_file_worker(self, url, subdir=""):
        """
        Worker function to download a file and save it.
        
        Args:
            url: URL of the file to download
            subdir: Subdirectory within media_dir to save file
            
        Returns:
            Dictionary with file info or None if download failed
        """
        if url in self.downloaded_files:
            # Already downloaded this file
            return None
            
        try:
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"Downloading {url}")
                    response = requests.get(url, timeout=30, stream=True)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    logger.error(f"Error downloading {url}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.delay * (attempt + 1))  # Exponential backoff
                    else:
                        return None
            
            # Create a safe filename
            filename = self.generate_safe_filename(url)
            
            # Create the destination directory if it doesn't exist
            dest_dir = os.path.join(self.media_dir, subdir)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Save the file
            file_path = os.path.join(dest_dir, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Mark as downloaded
            self.downloaded_files.add(url)
            
            # Get content type
            content_type = response.headers.get('Content-Type', '')
            
            # Return file info
            return {
                'original_url': url,
                'filename': filename,
                'path': os.path.join(subdir, filename).replace('\\', '/'),
                'size': file_size,
                'content_type': content_type
            }
            
        except Exception as e:
            logger.error(f"Error saving file from {url}: {e}")
            return None
    
    def download_file(self, url, subdir=""):
        """
        Schedule a file download using the thread pool executor.
        
        Args:
            url: URL of the file to download
            subdir: Subdirectory within media_dir to save file
            
        Returns:
            Dictionary with file info or None if download failed
        """
        if not hasattr(self, 'executor'):
            # If no executor exists (e.g., direct call), do synchronous download
            return self.download_file_worker(url, subdir)
        
        if url in self.downloaded_files:
            # Already downloaded this file
            return None
        
        # Track URL as downloaded to prevent duplicates
        self.downloaded_files.add(url)
        
        # Create a placeholder with basic info
        file_info = {
            'original_url': url,
            'path': f"pending_{hashlib.md5(url.encode()).hexdigest()}",
            'status': 'downloading'
        }
        
        # Schedule the download
        future = self.executor.submit(self.download_file_worker, url, subdir)
        self.download_futures.append(future)
        
        # Return the placeholder
        return file_info
    
    def download_images(self, soup, url, page_dir):
        """
        Download all images from a page.
        
        Args:
            soup: BeautifulSoup object for the page
            url: URL of the page
            page_dir: Directory to save images to
            
        Returns:
            List of image info dictionaries
        """
        images = []
        
        # Find all image tags
        for img in soup.find_all('img', src=True):
            img_url = urljoin(url, img['src'])
            
            # Skip data URIs
            if img_url.startswith('data:'):
                continue
                
            # Download image
            img_info = self.download_file(img_url, page_dir)
            if img_info:
                # Add additional information
                img_info['alt_text'] = img.get('alt', '')
                img_info['width'] = img.get('width', '')
                img_info['height'] = img.get('height', '')
                img_info['type'] = 'image'
                
                images.append(img_info)
        
        return images
    
    def download_videos(self, soup, url, page_dir):
        """
        Download video elements from a page.
        
        Args:
            soup: BeautifulSoup object for the page
            url: URL of the page
            page_dir: Directory to save videos to
            
        Returns:
            List of video info dictionaries
        """
        videos = []
        
        # Find all video tags
        for video in soup.find_all('video', src=True):
            video_url = urljoin(url, video['src'])
            video_info = self.download_file(video_url, page_dir)
            if video_info:
                video_info['type'] = 'video'
                videos.append(video_info)
        
        # Find all video source tags
        for source in soup.find_all('source', src=True):
            if source.parent.name == 'video':
                video_url = urljoin(url, source['src'])
                video_info = self.download_file(video_url, page_dir)
                if video_info:
                    video_info['type'] = 'video'
                    videos.append(video_info)
        
        return videos
    
    def extract_document_files(self, soup, url, page_dir):
        """
        Extract document files linked from a page (PDFs, etc.).
        
        Args:
            soup: BeautifulSoup object for the page
            url: URL of the page
            page_dir: Directory to save documents to
            
        Returns:
            List of document info dictionaries
        """
        documents = []
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link['href']
            link_url = urljoin(url, href)
            
            # Check if URL is a file we want to download
            path = urlparse(link_url).path.lower()
            if any(path.endswith(ext) for ext in self.file_extensions):
                doc_info = self.download_file(link_url, page_dir)
                if doc_info:
                    doc_info['type'] = 'document'
                    doc_info['link_text'] = self.clean_text(link.text)
                    documents.append(doc_info)
        
        return documents
    
    def create_page_subdir(self, url):
        """
        Create a subdirectory name for a page.
        
        Args:
            url: URL of the page
            
        Returns:
            Subdirectory name
        """
        # Extract path from URL
        path = urlparse(url).path
        
        # Clean up the path to use as a directory
        path = path.strip('/')
        if not path:
            path = 'homepage'
        else:
            path = re.sub(r'[^\w/\-.]', '_', path)
            
        # Replace slashes with underscores to create a valid subdirectory name
        subdir = path.replace('/', '_')
        
        # Truncate if too long
        if len(subdir) > 100:
            subdir = subdir[:100]
            
        return subdir
    
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
            return {"title": "", "content": "", "headings": [], "media": []}
        
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
        
        # Create a subdirectory for this page's media files
        page_dir = self.create_page_subdir(url)
        
        # Extract media content
        images = self.download_images(main_content, url, page_dir)
        videos = self.download_videos(main_content, url, page_dir)
        documents = self.extract_document_files(main_content, url, page_dir)
        
        # Combine all media
        media = images + videos + documents
        
        return {
            "title": title,
            "content": content,
            "headings": headings,
            "media": media
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

    def crawl(self, start_url=None, max_pages=None, max_workers=5):
        """
        Crawl the documentation site starting from the given URL.
        
        Args:
            start_url: Starting URL (defaults to base_url)
            max_pages: Maximum number of pages to crawl
            max_workers: Maximum number of concurrent download workers
            
        Returns:
            List of crawled documents
        """
        if start_url is None:
            start_url = self.base_url
        
        queue = [start_url]
        page_count = 0
        
        # Initialize a thread pool for concurrent downloads
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.download_futures = []
        
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
                        "headings": content_data["headings"],
                        "media": content_data["media"]
                    })
                    page_count += 1
                    pbar.update(1)
                
                # Extract links and add to queue
                links = self.extract_links(soup, current_url)
                queue.extend(links)
                
                # Be nice to the server
                time.sleep(self.delay)
        
        # Wait for all downloads to complete
        logger.info("Waiting for media downloads to complete...")
        concurrent.futures.wait(self.download_futures)
        self.executor.shutdown()
        
        # Count media files
        media_count = sum(len(doc.get("media", [])) for doc in self.docs)
        logger.info(f"Crawled {len(self.docs)} pages with {media_count} media files.")
        return self.docs

    def save_docs(self, filename="modalai_docs.json", media_catalog_file="modalai_media.json"):
        """
        Save the crawled documents to JSON files.
        
        Args:
            filename: Name of the output file for documents
            media_catalog_file: Name of the output file for media catalog
        """
        # Save all documents
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(self.docs, f, indent=2)
        
        logger.info(f"Saved {len(self.docs)} documents to {output_path}")
        
        # Create a media catalog for easier lookup
        media_catalog = {}
        for doc in self.docs:
            for media in doc.get("media", []):
                media_id = media["path"]
                media_catalog[media_id] = {
                    "document_url": doc["url"],
                    "document_title": doc["title"],
                    **media
                }
        
        # Save media catalog
        media_catalog_path = os.path.join(self.output_dir, media_catalog_file)
        with open(media_catalog_path, "w") as f:
            json.dump(media_catalog, f, indent=2)
        
        logger.info(f"Saved media catalog with {len(media_catalog)} entries to {media_catalog_path}")


def main():
    """Main entry point for the scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape Modal AI documentation")
    parser.add_argument("--url", default="https://docs.modalai.com", help="Base URL to start scraping")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum number of pages to scrape")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of concurrent download workers")
    parser.add_argument("--docs-file", default="modalai_docs.json", help="Output JSON file for documents")
    parser.add_argument("--media-catalog", default="modalai_media.json", help="Output JSON file for media catalog")
    parser.add_argument("--download-media", action="store_true", help="Download media files (images, videos, documents)")
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    logger.setLevel(logging.INFO)
    
    scraper = ModalAIScraper(base_url=args.url, output_dir=args.output)
    
    if not args.download_media:
        # If not downloading media, empty the file extensions list
        scraper.file_extensions = []
    
    # Start crawling
    logger.info(f"Starting crawl from {args.url}")
    scraper.crawl(max_pages=args.max_pages, max_workers=args.max_workers)
    
    # Save results
    scraper.save_docs(filename=args.docs_file, media_catalog_file=args.media_catalog)

if __name__ == "__main__":
    main()