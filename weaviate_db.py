# Legacy Compatibility Module
#
# This is a compatibility module to avoid breaking existing code.
# New code should use src/db/weaviate_client.py instead.

import os
import logging
from src.db.weaviate_client import WeaviateClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log a warning about using the legacy module
logger.warning("Using legacy weaviate_db module. Consider updating code to use src.db.weaviate_client.WeaviateClient directly.")

# Alias the new class
WeaviateDB = WeaviateClient