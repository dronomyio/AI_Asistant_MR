import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import time

class ModalAIDocScraper:
    def __init__(self, base_url="https://docs.modalai.com"):
        self.base_url = base_url
        self.visited_urls = set()
        self.docs = []
        self.max_retries = 3
        self.delay = 1  # seconds between requests

    def is_valid_url(self, url):
        """Check if URL is within the documentation domain."""
        parsed_url = urlparse(url)
        return parsed_url.netloc == urlparse(self.base_url).netloc and "docs.modalai.com" in url

    def clean_text(self, text):
        """Clean up text by removing excessive whitespace."""
        return " ".join(text.split())

    def extract_content(self, soup):
        """Extract the main content from the page."""
        # Target the main content area - adjust selectors based on the site structure
        main_content = soup.find("div", class_="content") or soup.find("main") or soup.find("article")
        
        if not main_content:
            return {"title": "", "content": "", "headings": []}
        
        # Extract title
        title_elem = soup.find("h1")
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
        for p in main_content.find_all(["p", "pre", "code", "ul", "ol"]):
            paragraphs.append(self.clean_text(p.text))
        
        content = "\n\n".join(paragraphs)
        
        return {
            "title": title,
            "content": content,
            "headings": headings
        }

    def get_page(self, url):
        """Fetch a page with retries."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                print(f"Error fetching {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))  # Exponential backoff
                else:
                    return None

    def extract_links(self, soup, current_url):
        """Extract all links from the page."""
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            absolute_url = urljoin(current_url, href)
            
            # Filter out external links, anchors, etc.
            if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                links.append(absolute_url)
        
        return links

    def crawl(self, start_url=None, max_pages=None):
        """Crawl the documentation site starting from the given URL."""
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
                content_data = self.extract_content(soup)
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
        
        print(f"Crawled {len(self.docs)} pages.")
        return self.docs

    def save_docs(self, output_dir="data"):
        """Save the crawled documents to JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all documents in a single file
        with open(os.path.join(output_dir, "modalai_docs.json"), "w") as f:
            json.dump(self.docs, f, indent=2)
        
        print(f"Saved {len(self.docs)} documents to {output_dir}/modalai_docs.json")

def main():
    scraper = ModalAIDocScraper()
    scraper.crawl(max_pages=200)  # Limit to 200 pages for testing
    scraper.save_docs()

if __name__ == "__main__":
    main()