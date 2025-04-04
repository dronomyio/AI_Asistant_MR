# Legacy Compatibility Module
#
# This is a compatibility module to avoid breaking existing code.
# New code should use src/embeddings/contextual_embeddings.py instead.

import os
import logging
from src.embeddings.contextual_embeddings import ContextualEmbeddings as NewContextualEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log a warning about using the legacy module
logger.warning("Using legacy contextual_embeddings module. Consider updating code to use src.embeddings.contextual_embeddings.ContextualEmbeddings directly.")

# Re-export class with legacy compatibility
ContextualVectorDB = NewContextualEmbeddings