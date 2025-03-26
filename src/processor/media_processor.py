"""
Media processor module for handling multimedia content using Unstructured.io.

This module provides functionality to process various types of media files:
- Images: Extract text using OCR and generate captions
- PDFs: Extract text, tables, and structure
- Other documents: Process based on type
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

# Unstructured imports
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from unstructured.staging.base import elements_to_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MediaProcessor:
    """
    Process media files using Unstructured.io.
    """
    def __init__(self, media_dir: str = "data/media", output_dir: str = "data"):
        """
        Initialize the media processor.
        
        Args:
            media_dir: Directory containing media files
            output_dir: Directory for output files
        """
        self.media_dir = media_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image file using Unstructured.io.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with extracted content
        """
        try:
            logger.info(f"Processing image: {image_path}")
            # Use partition_image to extract text from the image
            elements = partition_image(
                filename=image_path,
                include_page_breaks=False,
                extract_images_in_pdf=False
            )
            
            # Convert elements to a dictionary
            content = {
                "type": "image",
                "path": image_path,
                "elements": elements_to_json(elements),
                "text": " ".join([str(element) for element in elements]),
                "error": None
            }
            
            return content
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "type": "image",
                "path": image_path,
                "elements": [],
                "text": "",
                "error": str(e)
            }
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF file using Unstructured.io.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted content
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            # Use partition_pdf to extract content from the PDF
            elements = partition_pdf(
                filename=pdf_path,
                include_page_breaks=True,
                extract_images_in_pdf=True,
                infer_table_structure=True
            )
            
            # Convert elements to a dictionary
            content = {
                "type": "pdf",
                "path": pdf_path,
                "elements": elements_to_json(elements),
                "text": " ".join([str(element) for element in elements]),
                "error": None
            }
            
            return content
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return {
                "type": "pdf",
                "path": pdf_path,
                "elements": [],
                "text": "",
                "error": str(e)
            }
    
    def process_document(self, doc_path: str) -> Dict[str, Any]:
        """
        Process a document file using Unstructured.io.
        
        Args:
            doc_path: Path to the document file
            
        Returns:
            Dictionary with extracted content
        """
        try:
            logger.info(f"Processing document: {doc_path}")
            
            # Use the auto partitioner to handle various document types
            elements = partition(
                filename=doc_path,
                include_page_breaks=True,
                extract_images_in_pdf=True,
                include_metadata=True
            )
            
            # Convert elements to a dictionary
            content = {
                "type": "document",
                "path": doc_path,
                "elements": elements_to_json(elements),
                "text": " ".join([str(element) for element in elements]),
                "error": None
            }
            
            return content
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            return {
                "type": "document",
                "path": doc_path,
                "elements": [],
                "text": "",
                "error": str(e)
            }
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a file based on its type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with extracted content
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Process based on file extension
        if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return self.process_image(file_path)
        elif file_ext == '.pdf':
            return self.process_pdf(file_path)
        elif file_ext in ['.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.txt', '.md', '.rtf']:
            return self.process_document(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_ext} for {file_path}")
            return {
                "type": "unknown",
                "path": file_path,
                "elements": [],
                "text": "",
                "error": "Unsupported file type"
            }
    
    def process_media_catalog(self, catalog_file: str = "modalai_media.json") -> List[Dict[str, Any]]:
        """
        Process all media files referenced in the media catalog.
        
        Args:
            catalog_file: Name of the media catalog JSON file
            
        Returns:
            List of processed media content
        """
        catalog_path = os.path.join(self.output_dir, catalog_file)
        
        if not os.path.exists(catalog_path):
            logger.error(f"Media catalog file not found: {catalog_path}")
            return []
        
        # Load media catalog
        try:
            with open(catalog_path, 'r') as f:
                media_catalog = json.load(f)
                logger.info(f"Loaded media catalog with {len(media_catalog)} entries from {catalog_path}")
        except Exception as e:
            logger.error(f"Error loading media catalog from {catalog_path}: {e}")
            return []
        
        processed_media = []
        
        # Process each media file in the catalog
        for media_id, media_info in tqdm(media_catalog.items(), desc="Processing media files"):
            media_path = os.path.join(self.media_dir, media_info.get('path', ''))
            
            if not os.path.exists(media_path):
                logger.warning(f"Media file not found: {media_path}")
                continue
            
            processed_content = self.process_file(media_path)
            
            # Add metadata from catalog
            processed_content.update({
                "media_id": media_id,
                "document_url": media_info.get('document_url', ''),
                "document_title": media_info.get('document_title', ''),
                "content_type": media_info.get('content_type', ''),
                "alt_text": media_info.get('alt_text', ''),
                "link_text": media_info.get('link_text', '')
            })
            
            processed_media.append(processed_content)
        
        return processed_media
    
    def save_processed_media(self, processed_media: List[Dict[str, Any]], output_file: str = "modalai_processed_media.json"):
        """
        Save processed media content to a JSON file.
        
        Args:
            processed_media: List of processed media content
            output_file: Name of the output JSON file
        """
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(processed_media, f, indent=2)
            
            logger.info(f"Saved {len(processed_media)} processed media entries to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed media to {output_path}: {e}")
    
    def process_and_save(self, catalog_file: str = "modalai_media.json", output_file: str = "modalai_processed_media.json"):
        """
        Process all media files and save the results.
        
        Args:
            catalog_file: Name of the media catalog JSON file
            output_file: Name of the output JSON file
        """
        processed_media = self.process_media_catalog(catalog_file)
        self.save_processed_media(processed_media, output_file)
        return processed_media


def main():
    """Main entry point for media processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process media files using Unstructured.io")
    parser.add_argument("--catalog", default="modalai_media.json", help="Media catalog JSON file")
    parser.add_argument("--output", default="modalai_processed_media.json", help="Output JSON file")
    parser.add_argument("--media-dir", default="data/media", help="Media directory")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    
    args = parser.parse_args()
    
    processor = MediaProcessor(media_dir=args.media_dir, output_dir=args.output_dir)
    processor.process_and_save(catalog_file=args.catalog, output_file=args.output)


if __name__ == "__main__":
    main()