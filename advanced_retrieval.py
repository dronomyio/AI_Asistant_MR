# Legacy Compatibility Module
#
# This is a compatibility module to avoid breaking existing code.
# New code should use src/retrieval/advanced_retrieval.py instead.

import os
import logging
from src.retrieval.advanced_retrieval import (
    AdvancedRetrieval, 
    ElasticsearchBM25, 
    hybrid_search, 
    rerank_results, 
    create_rag_response,
    format_results
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log a warning about using the legacy module
logger.warning("Using legacy advanced_retrieval module. Consider updating code to use src.retrieval.advanced_retrieval directly.")