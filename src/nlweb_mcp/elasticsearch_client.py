import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError, NotFoundError

# Handle both relative and absolute imports
try:
    from .config import ELASTICSEARCH_HOST, EMBEDDING_DIMENSIONS
except ImportError:
    from config import ELASTICSEARCH_HOST, EMBEDDING_DIMENSIONS

# Embedding weights for composite embedding
TITLE_WEIGHT = 0.6
CONTENT_WEIGHT = 0.4

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    def __init__(self, host: str = ELASTICSEARCH_HOST):
        # Configure client for ES 8.x with connection pooling and timeouts
        self.client = Elasticsearch(
            [host],
            # Disable SSL verification for local development
            verify_certs=False,
            ssl_show_warn=False,
            # Connection pool settings
            max_retries=3,
            retry_on_timeout=True,
            # Timeout settings
            timeout=30,
            # Connection pool configuration
            maxsize=25,
            # Refresh connections periodically
            sniff_on_start=False,
            sniff_on_connection_fail=False,
            sniff_timeout=10,
        )
        self.test_connection()
    
    def test_connection(self):
        """Test connection to Elasticsearch"""
        try:
            info = self.client.info()
            logger.info(f"Connected to Elasticsearch: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    def health_check(self):
        """Perform a health check and reconnect if necessary"""
        try:
            # Simple ping to test connection with timeout
            if not self.client.ping(timeout=10):
                logger.warning("Elasticsearch ping failed, attempting to reconnect...")
                try:
                    self.test_connection()
                    return False
                except Exception as reconnect_error:
                    logger.error(f"Reconnection failed: {reconnect_error}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False
    
    async def async_health_check(self):
        """Async version of health check"""
        import asyncio
        try:
            # Run health check in thread pool to avoid blocking
            result = await asyncio.wait_for(
                asyncio.to_thread(self.health_check),
                timeout=15.0  # 15 second timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Async health check timed out")
            return False
        except Exception as e:
            logger.error(f"Async health check failed: {e}")
            return False
    
    def create_index(self, index_name: str, force_recreate: bool = False):
        """Create an index with the appropriate mappings for NLWeb products with separate embeddings"""
        
        if force_recreate and self.client.indices.exists(index=index_name):
            logger.info(f"Deleting existing index: {index_name}")
            self.client.indices.delete(index=index_name)
        
        if self.client.indices.exists(index=index_name):
            logger.info(f"Index {index_name} already exists")
            return
        
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "product_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        },
                        "category_analyzer": {
                            "tokenizer": "path_hierarchy",
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "product_id": {
                        "type": "keyword"
                    },
                    "url": {
                        "type": "keyword"
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "product_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "price": {
                        "type": "float"
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "product_analyzer"
                    },
                    "related_products": {
                        "type": "keyword"
                    },
                    "category": {
                        "type": "text",
                        "analyzer": "product_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            },
                            "hierarchy": {
                                "type": "text",
                                "analyzer": "category_analyzer"
                            }
                        }
                    },
                    "title_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMENSIONS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "content_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMENSIONS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "composite_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMENSIONS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "schema_org": {
                        "type": "object",
                        "enabled": True
                    },
                    "created_at": {
                        "type": "date"
                    },
                    "updated_at": {
                        "type": "date"
                    }
                }
            }
        }
        
        try:
            # Use the correct API for ES 8.x
            self.client.indices.create(index=index_name, **mapping)
            logger.info(f"Created index: {index_name}")
        except RequestError as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            raise
    
    def create_composite_embedding(self, title_embedding: List[float], content_embedding: List[float]) -> List[float]:
        """Create weighted composite embedding from title and content embeddings"""
        try:
            # Convert to numpy arrays
            title_vec = np.array(title_embedding)
            content_vec = np.array(content_embedding)
            
            # Apply weights and combine
            composite = (TITLE_WEIGHT * title_vec + CONTENT_WEIGHT * content_vec)
            
            # Normalize the composite vector
            norm = np.linalg.norm(composite)
            if norm > 0:
                composite = composite / norm
            
            return composite.tolist()
        except Exception as e:
            logger.error(f"Failed to create composite embedding: {e}")
            # Return average if calculation fails
            return [(t + c) / 2 for t, c in zip(title_embedding, content_embedding)]
    
    def index_product(self, index_name: str, product_data: Dict[str, Any]):
        """Index a single product document"""
        try:
            product_id = product_data["product_id"]
            logger.debug(f"ElasticsearchClient.index_product: Indexing product with _id='{product_id}' (type: {type(product_id)}, repr: {repr(product_id)}) in index '{index_name}'")
            
            response = self.client.index(
                index=index_name,
                id=product_id,
                document=product_data
            )
            logger.debug(f"ElasticsearchClient.index_product: Successfully indexed product '{product_data.get('title', 'Unknown')}' with _id='{product_id}'")
            return response
        except Exception as e:
            logger.error(f"Failed to index product {product_data.get('product_id', 'unknown')}: {e}")
            raise
    
    def bulk_index_products(self, index_name: str, products: List[Dict[str, Any]]):
        """Bulk index multiple products"""
        from elasticsearch.helpers import bulk
        
        actions = []
        for i, product in enumerate(products):
            product_id = product["product_id"]
            if i < 3:  # Log first 3 products for debugging
                logger.debug(f"ElasticsearchClient.bulk_index_products: Product {i+1} - _id='{product_id}' (type: {type(product_id)}, repr: {repr(product_id)})")
            
            actions.append({
                "_index": index_name,
                "_id": product_id,
                "_source": product
            })
        
        try:
            success, failed = bulk(self.client, actions, chunk_size=100)
            logger.info(f"Bulk indexed {success} products to {index_name}")
            if failed:
                logger.warning(f"Failed to index {len(failed)} products")
                for fail in failed[:3]:  # Log first 3 failures for debugging
                    logger.debug(f"Failed to index: {fail}")
            return success, failed
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            raise
    
    def semantic_search(self, index_name: str, query_embedding: List[float], top_k: int = 10, 
                      category_filter: Optional[str] = None, price_range: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using composite embedding with optional filters"""
        
        # Build filter conditions
        filters = []
        if category_filter:
            filters.append({"match": {"category": category_filter}})
        
        if price_range:
            range_filter = {"range": {"price": {}}}
            if "min" in price_range:
                range_filter["range"]["price"]["gte"] = price_range["min"]
            if "max" in price_range:
                range_filter["range"]["price"]["lte"] = price_range["max"]
            filters.append(range_filter)
        
        # Use the modern kNN query format with the knn parameter at the root level
        search_query = {
            "knn": {
                "field": "composite_embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 2  # Common practice to set higher for better recall
            },
            "_source": ["product_id", "url", "title", "price", "description", "category", "schema_org"],
            "size": top_k
        }
        
        # Add filters if any
        if filters:
            search_query["knn"]["filter"] = {
                "bool": {
                    "must": filters
                }
            }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Health check before search with timeout
                if attempt > 0:
                    if not self.health_check():
                        logger.warning(f"Health check failed on retry {attempt}")
                
                # Add timeout to search operation
                response = self.client.search(
                    index=index_name, 
                    body=search_query,
                    timeout="30s"  # 30 second search timeout
                )
                hits = response["hits"]["hits"]
                
                results = []
                for hit in hits:
                    result = hit["_source"]
                    result["_score"] = hit["_score"]
                    results.append(result)
                
                return results
            except NotFoundError:
                logger.warning(f"Index {index_name} not found")
                return []
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Search attempt {attempt + 1} failed: {e}, retrying...")
                    import time
                    time.sleep(min(1 * (attempt + 1), 5))  # Progressive delay, max 5 seconds
                else:
                    logger.error(f"Search failed after {max_retries} attempts: {e}")
                    raise
    
    def hybrid_search(self, index_name: str, query: str, query_embedding: List[float], top_k: int = 10,
                     category_filter: Optional[str] = None, price_range: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search with filters"""
        
        # Build filter conditions
        filters = []
        if category_filter:
            filters.append({"match": {"category": category_filter}})
        
        if price_range:
            range_filter = {"range": {"price": {}}}
            if "min" in price_range:
                range_filter["range"]["price"]["gte"] = price_range["min"]
            if "max" in price_range:
                range_filter["range"]["price"]["lte"] = price_range["max"]
            filters.append(range_filter)
        
        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # Semantic search using composite embedding
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'composite_embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                },
                                "boost": 1.0
                            }
                        },
                        # Keyword search on title with high boost
                        {
                            "match": {
                                "title": {
                                    "query": query,
                                    "boost": 3.0
                                }
                            }
                        },
                        # Keyword search on description
                        {
                            "match": {
                                "description": {
                                    "query": query,
                                    "boost": 1.0
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["product_id", "url", "title", "price", "description", "category", "schema_org"]
        }
        
        # Add filters if any
        if filters:
            search_query["query"]["bool"]["filter"] = filters
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.search(
                    index=index_name,
                    body=search_query,
                    timeout="30s"
                )
                hits = response["hits"]["hits"]
                
                results = []
                for hit in hits:
                    result = hit["_source"]
                    result["_score"] = hit["_score"]
                    results.append(result)
                
                return results
            except NotFoundError:
                logger.warning(f"Index {index_name} not found")
                return []
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Hybrid search attempt {attempt + 1} failed: {e}, retrying...")
                    import time
                    time.sleep(min(1 * (attempt + 1), 5))
                else:
                    logger.error(f"Hybrid search failed after {max_retries} attempts: {e}")
                    raise
    
    def get_product_by_id(self, index_name: str, product_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific product by its ID"""
        try:
            # Enhanced debugging to catch ID format issues
            logger.debug(f"ElasticsearchClient.get_product_by_id: Querying index='{index_name}' for product_id='{product_id}' (type: {type(product_id)}, len: {len(str(product_id))}, repr: {repr(product_id)})")
            
            # Normalize product_id to ensure consistent format
            normalized_id = str(product_id).strip()
            if normalized_id != product_id:
                logger.debug(f"ElasticsearchClient.get_product_by_id: Normalized product_id from '{repr(product_id)}' to '{repr(normalized_id)}'")
            
            response = self.client.get(index=index_name, id=normalized_id)
            logger.debug(f"ElasticsearchClient.get_product_by_id: Successfully found product '{response['_source'].get('title', 'Unknown')}' with _id='{response['_id']}' in index='{index_name}'")
            return response["_source"]
        except NotFoundError:
            logger.warning(f"ElasticsearchClient.get_product_by_id: Product '{repr(product_id)}' (normalized: '{repr(normalized_id)}') not found in index '{index_name}' (NotFoundError)")
            
            # Try to find similar IDs for debugging
            try:
                logger.debug(f"ElasticsearchClient.get_product_by_id: Searching for similar product IDs in index '{index_name}'...")
                search_query = {
                    "query": {
                        "wildcard": {
                            "product_id": f"*{normalized_id}*"
                        }
                    },
                    "_source": ["product_id", "title"],
                    "size": 5
                }
                search_response = self.client.search(index=index_name, body=search_query, timeout="10s")
                
                hits = search_response.get("hits", {}).get("hits", [])
                if hits:
                    logger.debug(f"ElasticsearchClient.get_product_by_id: Found {len(hits)} products with similar IDs:")
                    for hit in hits:
                        source = hit["_source"]
                        logger.debug(f"  - Document _id: '{hit['_id']}', product_id field: '{source.get('product_id')}', title: '{source.get('title', 'Unknown')[:50]}...'")
                else:
                    logger.debug(f"ElasticsearchClient.get_product_by_id: No products found with similar IDs to '{normalized_id}'")
                    
            except Exception as search_error:
                logger.debug(f"ElasticsearchClient.get_product_by_id: Failed to search for similar IDs: {search_error}")
            
            return None
        except Exception as e:
            logger.error(f"ElasticsearchClient.get_product_by_id: Failed to get product {repr(product_id)} from index '{index_name}': {e}")
            raise
    
    def get_products_by_urls(self, index_name: str, urls: List[str]) -> List[Dict[str, Any]]:
        """Get products by their URLs with intelligent URL matching"""
        products = []
        
        for url in urls:
            try:
                logger.debug(f"ElasticsearchClient.get_products_by_urls: Searching for URL: {url}")
                
                # Try both URL formats - with and without trailing slash
                url_variations = [url.strip()]
                if url.endswith('/'):
                    url_variations.append(url.rstrip('/'))
                else:
                    url_variations.append(url + '/')
                
                found = False
                for url_variant in url_variations:
                    # Search for product by URL using term query for exact match
                    search_query = {
                        "query": {
                            "term": {
                                "url": url_variant  # Try without .keyword first
                            }
                        },
                        "_source": ["product_id", "url", "title", "price", "description", "category", "schema_org"],
                        "size": 1
                    }
                    
                    try:
                        response = self.client.search(
                            index=index_name,
                            body=search_query,
                            timeout="10s"
                        )
                        
                        hits = response.get("hits", {}).get("hits", [])
                        if hits:
                            logger.debug(f"Found product with URL variant: {url_variant}")
                            products.append(hits[0]["_source"])
                            found = True
                            break
                        else:
                            # Try with .keyword field for exact match
                            search_query["query"]["term"] = {"url.keyword": url_variant}
                            response = self.client.search(
                                index=index_name,
                                body=search_query,
                                timeout="10s"
                            )
                            
                            hits = response.get("hits", {}).get("hits", [])
                            if hits:
                                logger.debug(f"Found product with URL.keyword variant: {url_variant}")
                                products.append(hits[0]["_source"])
                                found = True
                                break
                    
                    except Exception as search_error:
                        logger.error(f"Search failed for URL variant {url_variant}: {search_error}")
                        continue
                
                if not found:
                    logger.warning(f"Product not found for any URL variant of: {url}")
                    products.append({
                        "url": url,
                        "error": "Product not found"
                    })
                    
            except Exception as e:
                logger.error(f"Failed to process URL {url}: {e}")
                products.append({
                    "url": url,
                    "error": f"Processing failed: {str(e)}"
                })
        
        return products
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics about an index"""
        try:
            stats = self.client.indices.stats(index=index_name)
            return {
                "document_count": stats["indices"][index_name]["total"]["docs"]["count"],
                "size_in_bytes": stats["indices"][index_name]["total"]["store"]["size_in_bytes"],
                "index_name": index_name
            }
        except NotFoundError:
            return {"error": f"Index {index_name} not found"}
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise