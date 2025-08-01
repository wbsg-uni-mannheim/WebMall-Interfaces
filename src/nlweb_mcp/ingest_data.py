#!/usr/bin/env python3
"""
Data ingestion script for NLWeb MCP implementation
"""

import logging
import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("../../.env")

# Handle both relative and absolute imports
try:
    from .elasticsearch_client import ElasticsearchClient
    from .embedding_service import EmbeddingService
    from .data_ingestion import DataIngestionPipeline
    from .config import WEBMALL_SHOPS
except ImportError:
    from elasticsearch_client import ElasticsearchClient
    from embedding_service import EmbeddingService
    from data_ingestion import DataIngestionPipeline
    from config import WEBMALL_SHOPS

def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Ingest data for NLWeb MCP servers')
    parser.add_argument('--shop', choices=list(WEBMALL_SHOPS.keys()) + ['all'], 
                       default='all', help='Shop to ingest data for')
    parser.add_argument('--force-recreate', action='store_true', 
                       help='Force recreate indices (will delete existing data)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        logger.info("Initializing Elasticsearch client...")
        es_client = ElasticsearchClient()
        
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService()
        
        # Test embedding service
        if not embedding_service.test_embedding_service():
            logger.error("Embedding service test failed")
            return 1
        
        # Initialize ingestion pipeline
        logger.info("Initializing data ingestion pipeline...")
        pipeline = DataIngestionPipeline(es_client, embedding_service)
        
        # Perform ingestion
        if args.shop == 'all':
            logger.info("Starting ingestion for all shops...")
            results = pipeline.ingest_all_shops(args.force_recreate)
        else:
            logger.info(f"Starting ingestion for {args.shop}...")
            results = pipeline.ingest_shop_data(args.shop, args.force_recreate)
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'ingestion_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Ingestion completed. Results saved to {results_file}")
        
        # Print summary
        if args.shop == 'all':
            print(f"\n=== Ingestion Summary ===")
            print(f"Total shops: {results.get('total_shops', 0)}")
            print(f"Successful shops: {results.get('successful_shops', 0)}")
            print(f"Total products processed: {results.get('total_products_processed', 0)}")
            print(f"Total products failed: {results.get('total_products_failed', 0)}")
            
            for shop_id, shop_result in results.get('shop_results', {}).items():
                status = "✓" if shop_result.get('success', False) else "✗"
                processed = shop_result.get('products_processed', 0)
                print(f"  {status} {shop_id}: {processed} products")
        else:
            status = "✓" if results.get('success', False) else "✗"
            processed = results.get('products_processed', 0)
            print(f"\n{status} {args.shop}: {processed} products processed")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())