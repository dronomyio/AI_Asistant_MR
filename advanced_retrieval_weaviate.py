# Legacy Compatibility Module
#
# This is a compatibility module to avoid breaking existing code.
# New code should use src/retrieval/advanced_retrieval.py instead.

import os
import logging
from src.retrieval.advanced_retrieval import AdvancedRetrieval, ElasticsearchBM25
from src.retrieval.advanced_retrieval import hybrid_search as new_hybrid_search
from src.retrieval.advanced_retrieval import rerank_results as new_rerank_results
from src.retrieval.advanced_retrieval import create_rag_response as new_create_rag_response
from src.retrieval.advanced_retrieval import format_results as new_format_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log a warning about using the legacy module
logger.warning("Using legacy advanced_retrieval_weaviate module. Consider updating code to use src.retrieval.advanced_retrieval directly.")

# Re-export functions with legacy compatibility
hybrid_search = new_hybrid_search
rerank_results = new_rerank_results
create_rag_response = new_create_rag_response
format_results = new_format_results