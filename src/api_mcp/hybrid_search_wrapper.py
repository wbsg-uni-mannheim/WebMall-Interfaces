"""
Hybrid Search Engine Wrapper

This module provides a wrapper around the nlweb SearchEngine that simplifies
the initialization and usage for hybrid servers. It provides a unified interface
for all hybrid servers to access semantic search capabilities.
"""

import sys
import os
import logging
from typing import Dict, Any, Optional

# Add the nlweb_mcp directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nlweb_mcp'))

try:
    from elasticsearch_client import ElasticsearchClient
    from embedding_service import EmbeddingService
    from search_engine import SearchEngine
    from config import WEBMALL_SHOPS
except ImportError as e:
    print(f"Error importing nlweb_mcp components: {e}")
    print("Please ensure nlweb_mcp is properly configured and accessible.")
    raise

logger = logging.getLogger(__name__)

class HybridSearchWrapper:
    """
    Wrapper class that provides a simplified interface to the nlweb SearchEngine
    for hybrid servers. Handles initialization and provides format-specific search methods.
    """
    
    def __init__(self, shop_id: str):
        """
        Initialize the hybrid search wrapper for a specific shop.
        
        Args:
            shop_id: The shop identifier (e.g., 'webmall_1', 'webmall_2', etc.)
        """
        self.shop_id = shop_id
        self._search_engine = None
        self._initialized = False
        
        # Validate shop_id
        if shop_id not in WEBMALL_SHOPS:
            raise ValueError(f"Invalid shop_id: {shop_id}. Must be one of: {list(WEBMALL_SHOPS.keys())}")
        
        self.shop_config = WEBMALL_SHOPS[shop_id]
        self.index_name = self.shop_config["index_name"]
        
    async def initialize(self):
        """
        Asynchronously initialize the search engine components.
        This should be called during server startup.
        """
        if self._initialized:
            return
            
        try:
            logger.info(f"Initializing hybrid search for {self.shop_id}")
            
            # Initialize Elasticsearch client
            es_client = ElasticsearchClient()
            
            # Initialize embedding service
            embedding_service = EmbeddingService()
            
            # Test embedding service
            if not embedding_service.test_embedding_service():
                raise Exception("Embedding service test failed")
            
            # Initialize search engine
            self._search_engine = SearchEngine(
                es_client,
                embedding_service,
                self.index_name
            )
            
            # Perform health check
            health = await self.health_check()
            if health.get("status") not in ["healthy", "degraded"]:
                logger.warning(f"Search engine health check shows issues: {health}")
            
            self._initialized = True
            logger.info(f"Successfully initialized hybrid search for {self.shop_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid search for {self.shop_id}: {e}")
            raise
    
    def _ensure_initialized(self):
        """Ensure the search engine is initialized before use."""
        if not self._initialized or not self._search_engine:
            raise RuntimeError("HybridSearchWrapper not initialized. Call initialize() first.")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the search system.
        
        Returns:
            Dict containing health status information
        """
        try:
            self._ensure_initialized()
            health_status = self._search_engine.health_check()
            health_status["shop_id"] = self.shop_id
            health_status["index_name"] = self.index_name
            return health_status
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "shop_id": self.shop_id,
                "index_name": self.index_name
            }
    
    # Format-specific search methods
    
    def search_server_a_format(self, query: str, per_page: int = 10, page: int = 1) -> str:
        """Search using server_a (E-Store Athletes) format."""
        self._ensure_initialized()
        return self._search_engine.search_server_a_format(query, per_page, page)
    
    def search_server_b_format(self, query: str, limit: int = 5, page_num: int = 1, sort_by_price: str = "none") -> str:
        """Search using server_b (TechTalk) format."""
        self._ensure_initialized()
        return self._search_engine.search_server_b_format(query, limit, page_num, sort_by_price)
    
    def search_server_c_format(self, query: str, results_per_page: int = 10, page_number: int = 1, result_order: str = "relevance") -> str:
        """Search using server_c (CamelCases) format."""
        self._ensure_initialized()
        return self._search_engine.search_server_c_format(query, results_per_page, page_number, result_order)
    
    def search_server_d_format(self, query: str, results_limit: int = 10, page_number: int = 1, min_price: float = None, max_price: float = None) -> str:
        """Search using server_d (Hardware Cafe) format."""
        self._ensure_initialized()
        return self._search_engine.search_server_d_format(query, results_limit, page_number, min_price, max_price)
    
    # Generic search methods
    
    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Perform a generic semantic search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            Dict containing search results in nlweb format
        """
        self._ensure_initialized()
        return self._search_engine.search(query, top_k)
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific product by ID.
        
        Args:
            product_id: The product ID to retrieve
            
        Returns:
            Dict containing product information or None if not found
        """
        self._ensure_initialized()
        return self._search_engine.get_product_by_id(product_id)
    
    # Utility methods
    
    def get_shop_info(self) -> Dict[str, Any]:
        """
        Get information about the current shop configuration.
        
        Returns:
            Dict containing shop configuration details
        """
        return {
            "shop_id": self.shop_id,
            "index_name": self.index_name,
            "shop_config": self.shop_config,
            "initialized": self._initialized
        }
    
    def __repr__(self):
        return f"HybridSearchWrapper(shop_id='{self.shop_id}', initialized={self._initialized})"


# Factory functions for easy instantiation

def create_hybrid_search_wrapper(shop_id: str) -> HybridSearchWrapper:
    """
    Factory function to create a HybridSearchWrapper instance.
    
    Args:
        shop_id: The shop identifier
        
    Returns:
        HybridSearchWrapper instance
    """
    return HybridSearchWrapper(shop_id)

async def create_and_initialize_hybrid_search(shop_id: str) -> HybridSearchWrapper:
    """
    Factory function to create and initialize a HybridSearchWrapper instance.
    
    Args:
        shop_id: The shop identifier
        
    Returns:
        Initialized HybridSearchWrapper instance
    """
    wrapper = HybridSearchWrapper(shop_id)
    await wrapper.initialize()
    return wrapper

# Shop-specific convenience functions

async def create_server_a_search() -> HybridSearchWrapper:
    """Create hybrid search wrapper for server A (E-Store Athletes)."""
    return await create_and_initialize_hybrid_search("webmall_1")

async def create_server_b_search() -> HybridSearchWrapper:
    """Create hybrid search wrapper for server B (TechTalk).""" 
    return await create_and_initialize_hybrid_search("webmall_2")

async def create_server_c_search() -> HybridSearchWrapper:
    """Create hybrid search wrapper for server C (CamelCases)."""
    return await create_and_initialize_hybrid_search("webmall_3")

async def create_server_d_search() -> HybridSearchWrapper:
    """Create hybrid search wrapper for server D (Hardware Cafe)."""
    return await create_and_initialize_hybrid_search("webmall_4")