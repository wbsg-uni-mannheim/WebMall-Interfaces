import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv

load_dotenv("../../.env")

# Handle both relative and absolute imports
try:
    from .woocommerce_client import WooCommerceClient
    from .elasticsearch_client import ElasticsearchClient
    from .embedding_service import EmbeddingService
    from .config import WEBMALL_SHOPS
except ImportError:
    from woocommerce_client import WooCommerceClient
    from elasticsearch_client import ElasticsearchClient
    from embedding_service import EmbeddingService
    from config import WEBMALL_SHOPS

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    def __init__(self, elasticsearch_client: ElasticsearchClient, embedding_service: EmbeddingService):
        self.es_client = elasticsearch_client
        self.embedding_service = embedding_service
        self.shop_clients = {}
        
        # Initialize WooCommerce clients for each shop
        for shop_id, shop_config in WEBMALL_SHOPS.items():
            wc_client = WooCommerceClient(
                base_url=shop_config["url"],
                consumer_key=shop_config.get("consumer_key"),
                consumer_secret=shop_config.get("consumer_secret")
            )
            self.shop_clients[shop_id] = wc_client
    
    def ingest_shop_data(self, shop_id: str, force_recreate_index: bool = False) -> Dict[str, Any]:
        """Ingest data for a specific shop"""
        if shop_id not in WEBMALL_SHOPS:
            raise ValueError(f"Invalid shop_id: {shop_id}")
        
        shop_config = WEBMALL_SHOPS[shop_id]
        index_name = shop_config["index_name"]
        
        logger.info(f"Starting data ingestion for {shop_id}")
        
        # Create or recreate the index
        self.es_client.create_index(index_name, force_recreate=force_recreate_index)
        
        # Get WooCommerce client
        wc_client = self.shop_clients[shop_id]
        
        # Test WooCommerce connection
        if not wc_client.test_connection():
            logger.error(f"Failed to connect to WooCommerce API for {shop_id}")
            return {"success": False, "error": "WooCommerce connection failed"}
        
        # Fetch all products from WooCommerce
        logger.info(f"Fetching products from {shop_id}")
        raw_products = wc_client.get_all_products()
        
        if not raw_products:
            logger.warning(f"No products found for {shop_id}")
            return {"success": True, "products_processed": 0, "message": "No products found"}
        
        # Process products in batches
        batch_size = 50
        total_processed = 0
        total_failed = 0
        
        for i in range(0, len(raw_products), batch_size):
            batch = raw_products[i:i + batch_size]
            processed, failed = self._process_product_batch(
                shop_id, batch, wc_client, index_name
            )
            total_processed += processed
            total_failed += failed
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(raw_products) + batch_size - 1)//batch_size} for {shop_id}")
        
        # Get final index statistics
        index_stats = self.es_client.get_index_stats(index_name)
        
        result = {
            "success": True,
            "shop_id": shop_id,
            "products_fetched": len(raw_products),
            "products_processed": total_processed,
            "products_failed": total_failed,
            "index_stats": index_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Completed data ingestion for {shop_id}: {result}")
        return result
    
    def _process_product_batch(self, shop_id: str, batch: List[Dict], wc_client: WooCommerceClient, 
                              index_name: str) -> tuple[int, int]:
        """Process a batch of products"""
        
        # Extract product data from WooCommerce format
        extracted_products = []
        for raw_product in batch:
            try:
                product_data = wc_client.extract_product_data(raw_product)
                extracted_products.append(product_data)
            except Exception as e:
                logger.error(f"Failed to extract product data for product {raw_product.get('id', 'unknown')}: {e}")
        
        if not extracted_products:
            return 0, len(batch)
        
        # Create separate embeddings for the batch
        try:
            embedding_results = self.embedding_service.create_separate_embeddings_batch(extracted_products)
        except Exception as e:
            logger.error(f"Failed to create separate embeddings for batch: {e}")
            return 0, len(batch)
        
        # Prepare documents for indexing
        documents = []
        for i, product in enumerate(extracted_products):
            try:
                embedding_data = embedding_results[i]
                
                # Create composite embedding using the elasticsearch client
                composite_embedding = self.es_client.create_composite_embedding(
                    embedding_data['title_embedding'],
                    embedding_data['content_embedding']
                )
                
                #print(f"Product: {product}")
                
                # Create document for Elasticsearch
                document = {
                    "product_id": product["product_id"],
                    "url": product["url"],
                    "title": product["title"],
                    "price": product["price"],
                    "description": product["description"],
                    "related_products": product["related_products"],
                    "category": product["category"],
                    "schema_org": product.get("schema_org", {}),  # Include schema.org data
                    "title_embedding": embedding_data['title_embedding'],
                    "content_embedding": embedding_data['content_embedding'],
                    "composite_embedding": composite_embedding,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                #print(f"Document: {document}")
                
                documents.append(document)
                
            except Exception as e:
                logger.error(f"Failed to prepare document for product {product.get('product_id', 'unknown')}: {e}")
        
        # Bulk index documents
        if documents:
            try:
                success, failed = self.es_client.bulk_index_products(index_name, documents)
                return success, len(failed) if failed else 0
            except Exception as e:
                logger.error(f"Failed to bulk index documents: {e}")
                return 0, len(documents)
        
        return 0, len(batch)
    
    def ingest_all_shops(self, force_recreate_indices: bool = False) -> Dict[str, Any]:
        """Ingest data for all shops"""
        results = {}
        
        logger.info("Starting data ingestion for all shops")
        
        for shop_id in WEBMALL_SHOPS.keys():
            try:
                result = self.ingest_shop_data(shop_id, force_recreate_indices)
                results[shop_id] = result
            except Exception as e:
                logger.error(f"Failed to ingest data for {shop_id}: {e}")
                results[shop_id] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Summary statistics
        total_processed = sum(r.get("products_processed", 0) for r in results.values())
        total_failed = sum(r.get("products_failed", 0) for r in results.values())
        successful_shops = sum(1 for r in results.values() if r.get("success", False))
        
        summary = {
            "total_shops": len(WEBMALL_SHOPS),
            "successful_shops": successful_shops,
            "total_products_processed": total_processed,
            "total_products_failed": total_failed,
            "timestamp": datetime.now().isoformat(),
            "shop_results": results
        }
        
        logger.info(f"Data ingestion completed: {summary}")
        return summary
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """Get status of all indices"""
        status = {}
        
        for shop_id, shop_config in WEBMALL_SHOPS.items():
            index_name = shop_config["index_name"]
            try:
                stats = self.es_client.get_index_stats(index_name)
                status[shop_id] = {
                    "index_name": index_name,
                    "exists": True,
                    "document_count": stats.get("document_count", 0),
                    "size_in_bytes": stats.get("size_in_bytes", 0)
                }
            except Exception as e:
                status[shop_id] = {
                    "index_name": index_name,
                    "exists": False,
                    "error": str(e)
                }
        
        return status