from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import json
import logging
import argparse
import os
import uuid
import threading
import time
from typing import Dict, Any

# Handle both relative and absolute imports
try:
    from ..elasticsearch_client import ElasticsearchClient
    from ..embedding_service import EmbeddingService
    from ..search_engine import SearchEngine
    from ..config import DEFAULT_TOP_K, WEBMALL_SHOPS
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from elasticsearch_client import ElasticsearchClient
    from embedding_service import EmbeddingService
    from search_engine import SearchEngine
    from config import DEFAULT_TOP_K, WEBMALL_SHOPS

load_dotenv(dotenv_path="/Users/aaronsteiner/Documents/GitHub/webmall-alternative-interfaces/.env")
logger = logging.getLogger(__name__)

# Global cart storage for persistence across HTTP requests
class GlobalCartStore:
    """Thread-safe global cart storage for HTTP/SSE transport persistence."""
    
    def __init__(self):
        self._carts: Dict[str, Dict[str, Any]] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = 3600  # 1 hour
        self._session_timeout = 7200   # 2 hours
        
    def _get_session_id(self, shop_id: str, client_info: str = "default") -> str:
        """Generate a session ID for cart tracking."""
        # For now, use shop_id as session ID since we want one cart per shop
        # In a real implementation, you'd use client connection info
        return f"{shop_id}_{client_info}"
    
    def get_cart(self, shop_id: str, client_info: str = "default") -> Dict[str, Any]:
        """Get cart for a session, creating if needed."""
        session_id = self._get_session_id(shop_id, client_info)
        
        with self._lock:
            self._last_access[session_id] = time.time()
            if session_id not in self._carts:
                self._carts[session_id] = {}
                logger.debug(f"Created new cart for session {session_id}")
            return self._carts[session_id]
    
    def clear_cart(self, shop_id: str, client_info: str = "default") -> None:
        """Clear cart for a session."""
        session_id = self._get_session_id(shop_id, client_info)
        
        with self._lock:
            if session_id in self._carts:
                self._carts[session_id].clear()
                self._last_access[session_id] = time.time()
                logger.debug(f"Cleared cart for session {session_id}")
    
    def cleanup_expired_carts(self) -> None:
        """Remove expired cart sessions."""
        current_time = time.time()
        expired_sessions = []
        
        with self._lock:
            for session_id, last_access in self._last_access.items():
                if current_time - last_access > self._session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self._carts.pop(session_id, None)
                self._last_access.pop(session_id, None)
                logger.debug(f"Cleaned up expired cart session {session_id}")

# Global cart store instance
_global_cart_store = GlobalCartStore()

@dataclass
class NLWebContext:
    """Context for the NLWeb MCP server."""
    search_engine: SearchEngine
    shop_id: str
    index_name: str
    cart: dict  # In-memory cart storage

def create_nlweb_server(shop_id: str, index_name: str, port: int = 8000) -> FastMCP:
    """Create a FastMCP server for NLWeb semantic search"""
    
    # Create lifespan function with shop-specific parameters
    @asynccontextmanager
    async def shop_lifespan(server: FastMCP) -> AsyncIterator[NLWebContext]:
        """
        Manages the NLWeb search engine lifecycle for this specific shop.
        """
        logger.info(f"Initializing NLWeb components for {shop_id}")
        
        try:
            # Initialize Elasticsearch client
            logger.debug(f"Initializing Elasticsearch client for {shop_id}")
            es_client = ElasticsearchClient()
            
            # Initialize embedding service
            logger.debug(f"Initializing embedding service for {shop_id}")
            embedding_service = EmbeddingService()
            
            # Test embedding service
            logger.debug(f"Testing embedding service for {shop_id}")
            if not embedding_service.test_embedding_service():
                raise Exception("Embedding service test failed")
            
            # Initialize search engine
            logger.debug(f"Initializing search engine for {shop_id}")
            search_engine = SearchEngine(
                es_client,
                embedding_service,
                index_name
            )
            
            # Perform health check
            logger.debug(f"Performing initial health check for {shop_id}")
            health = search_engine.health_check()
            if health.get("status") != "healthy":
                logger.warning(f"Server health check shows issues: {health}")
            else:
                logger.debug(f"Initial health check passed for {shop_id}")
            
            logger.info(f"Successfully initialized NLWeb MCP server for {shop_id}")
            
            # Start background health check task
            async def health_check_task():
                consecutive_failures = 0
                max_failures = 3
                base_delay = 300  # 5 minutes
                max_delay = 1800  # 30 minutes
                
                while True:
                    try:
                        # Use exponential backoff on failures
                        if consecutive_failures > 0:
                            delay = min(base_delay * (2 ** consecutive_failures), max_delay)
                        else:
                            delay = base_delay
                        
                        await asyncio.sleep(delay)
                        
                        # Use asyncio.wait_for to add timeout to health check
                        try:
                            health_result = await asyncio.wait_for(
                                asyncio.to_thread(es_client.health_check),
                                timeout=30.0  # 30 second timeout
                            )
                            
                            if health_result:
                                consecutive_failures = 0
                                logger.debug(f"Health check passed for {shop_id}")
                            else:
                                consecutive_failures += 1
                                logger.warning(f"Health check failed for {shop_id} (attempt {consecutive_failures}/{max_failures})")
                                
                                if consecutive_failures >= max_failures:
                                    logger.error(f"Health check failed {max_failures} times for {shop_id}, but continuing...")
                                    
                        except asyncio.TimeoutError:
                            consecutive_failures += 1
                            logger.warning(f"Health check timeout for {shop_id} (attempt {consecutive_failures}/{max_failures})")
                            
                    except asyncio.CancelledError:
                        logger.info(f"Health check task cancelled for {shop_id}")
                        break
                    except Exception as e:
                        consecutive_failures += 1
                        logger.error(f"Health check task error for {shop_id}: {e} (attempt {consecutive_failures}/{max_failures})")
            
            # Start background cart cleanup task
            async def cart_cleanup_task():
                cleanup_interval = 3600  # 1 hour
                while True:
                    try:
                        await asyncio.sleep(cleanup_interval)
                        _global_cart_store.cleanup_expired_carts()
                        logger.debug(f"Cart cleanup completed for {shop_id}")
                    except asyncio.CancelledError:
                        logger.info(f"Cart cleanup task cancelled for {shop_id}")
                        break
                    except Exception as e:
                        logger.error(f"Cart cleanup task error for {shop_id}: {e}")
            
            health_task = asyncio.create_task(health_check_task())
            cleanup_task = asyncio.create_task(cart_cleanup_task())
            
            try:
                yield NLWebContext(
                    search_engine=search_engine,
                    shop_id=shop_id,
                    index_name=index_name,
                    cart={}  # Initialize empty cart
                )
            finally:
                # Cancel background tasks on shutdown
                health_task.cancel()
                cleanup_task.cancel()
                try:
                    await asyncio.wait_for(health_task, timeout=5.0)
                    await asyncio.wait_for(cleanup_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.info(f"Background tasks shutdown completed for {shop_id}")
                    pass
        except Exception as e:
            logger.error(f"Failed to initialize NLWeb components: {e}")
            raise
        finally:
            # Cleanup resources
            try:
                if 'es_client' in locals():
                    # Close Elasticsearch client connections
                    if hasattr(es_client.client, 'close'):
                        es_client.client.close()
                        logger.debug(f"Closed Elasticsearch client for {shop_id}")
                
                if 'embedding_service' in locals():
                    # Close async client if it exists
                    if hasattr(embedding_service, 'async_client') and hasattr(embedding_service.async_client, 'close'):
                        await embedding_service.async_client.close()
                        logger.debug(f"Closed async OpenAI client for {shop_id}")
                    
                    # Close sync client if it has close method
                    if hasattr(embedding_service.client, 'close'):
                        embedding_service.client.close()
                        logger.debug(f"Closed sync OpenAI client for {shop_id}")
                        
            except Exception as cleanup_error:
                logger.warning(f"Error during resource cleanup for {shop_id}: {cleanup_error}")
            finally:
                logger.info(f"Shutting down NLWeb server for {shop_id}")
    
    # Initialize FastMCP server
    mcp = FastMCP(
        f"nlweb-{shop_id}",
        description=f"NLWeb semantic search server for {shop_id}",
        lifespan=shop_lifespan,
        host=os.getenv("HOST", "0.0.0.0"),
        port=str(port)
    )

    @mcp.tool(name=f"ask_{shop_id}")
    async def ask(ctx: Context, question: str, top_k: int = DEFAULT_TOP_K) -> str:
        f"""Search for products in {shop_id.replace('_', '-').upper()} using natural language queries.

        This tool performs semantic search across the {shop_id.replace('_', '-').upper()} product catalog using 
        OpenAI embeddings and returns schema.org formatted product information.

        Args:
            ctx: The MCP server provided context which includes the search engine
            question: The search query or question about products
            top_k: Number of results to return (default: 10, max: 50)

        Returns:
            JSON formatted response with product information from semantic search
        """
        try:
            search_engine = ctx.request_context.lifespan_context.search_engine
            
            # Validate parameters
            if not question or not question.strip():
                return json.dumps({"error": "Question parameter is required and cannot be empty"})
            
            top_k = min(max(1, top_k), 50)  # Clamp between 1 and 50
            
            # Perform semantic search with timeout (exclude descriptions to save tokens)
            try:
                results = await asyncio.wait_for(
                    asyncio.to_thread(search_engine.search, question.strip(), top_k, True),
                    timeout=45.0  # 45 second timeout for search
                )
                # Save results to file
                with open(f"results_{shop_id}.json", "w") as f:
                    json.dump(results, f, indent=2)
                return json.dumps(results, indent=2)
            except asyncio.TimeoutError:
                logger.warning(f"Search timeout for question: {question[:100]}...")
                return json.dumps({
                    "error": "Search request timed out",
                    "question": question,
                    "timeout": "45s"
                })
            
        except Exception as e:
            logger.error(f"Search failed for question '{question}': {e}")
            return json.dumps({
                "error": f"Search failed: {str(e)}",
                "question": question,
                "shop_id": ctx.request_context.lifespan_context.shop_id
            })

    @mcp.tool(name=f"get_product_{shop_id}")
    async def get_product(ctx: Context, product_id: str) -> str:
        f"""Get detailed information about a specific product by ID from {shop_id.replace('_', '-').upper()}.

        This tool retrieves comprehensive product information including 
        schema.org formatted data for a specific product in {shop_id.replace('_', '-').upper()}.

        Args:
            ctx: The MCP server provided context which includes the search engine
            product_id: The product ID to retrieve

        Returns:
            JSON formatted detailed product information or error message
        """
        try:
            search_engine = ctx.request_context.lifespan_context.search_engine
            
            if not product_id or not product_id.strip():
                return json.dumps({"error": "Product ID parameter is required and cannot be empty"})
            
            # Normalize product_id to ensure consistent format
            normalized_product_id = str(product_id).strip()
            logger.debug(f"get_product_{shop_id}: Normalized product_id from {repr(product_id)} to {repr(normalized_product_id)}")
            
            # Get product by ID with timeout
            try:
                product = await asyncio.wait_for(
                    asyncio.to_thread(search_engine.get_product_by_id, normalized_product_id),
                    timeout=30.0  # 30 second timeout
                )
                
                if product:
                    return json.dumps(product, indent=2)
                else:
                    return json.dumps({
                        "error": f"Product {normalized_product_id} not found",
                        "product_id": normalized_product_id,
                        "shop_id": ctx.request_context.lifespan_context.shop_id
                    })
                    
            except asyncio.TimeoutError:
                logger.warning(f"Get product timeout for ID: {normalized_product_id}")
                return json.dumps({
                    "error": "Get product request timed out",
                    "product_id": normalized_product_id,
                    "timeout": "30s"
                })
                
        except Exception as e:
            logger.error(f"Get product failed for ID '{normalized_product_id if 'normalized_product_id' in locals() else product_id}': {e}")
            return json.dumps({
                "error": f"Failed to get product: {str(e)}",
                "product_id": normalized_product_id if 'normalized_product_id' in locals() else product_id,
                "shop_id": ctx.request_context.lifespan_context.shop_id
            })

    @mcp.tool(name=f"get_products_by_urls_{shop_id}")
    async def get_products_by_urls(ctx: Context, urls: list) -> str:
        f"""Get detailed product information for multiple URLs from {shop_id.replace('_', '-').upper()}.

        This tool retrieves comprehensive product information including full descriptions
        and schema.org formatted data for a list of product URLs.

        Args:
            ctx: The MCP server provided context which includes the search engine
            urls: List of product URLs to retrieve detailed information for

        Returns:
            JSON formatted response with detailed product information for each URL
        """
        try:
            search_engine = ctx.request_context.lifespan_context.search_engine
            
            if not urls or not isinstance(urls, list):
                return json.dumps({"error": "URLs parameter must be a non-empty list"})
            
            # Validate and clean URLs
            valid_urls = []
            for url in urls:
                if isinstance(url, str) and url.strip():
                    # Don't force trailing slash - keep URL as provided
                    valid_urls.append(url.strip())
            
            if not valid_urls:
                return json.dumps({"error": "No valid URLs provided"})
            
            # Limit number of URLs to prevent abuse
            if len(valid_urls) > 20:
                return json.dumps({"error": "Maximum 20 URLs allowed per request"})
            
            # Get detailed product information for all URLs
            try:
                products = await asyncio.wait_for(
                    asyncio.to_thread(search_engine.get_products_by_urls, valid_urls),
                    timeout=30.0  # 30 second timeout
                )
                
                return json.dumps({
                    "shop_id": shop_id,
                    "requested_urls": len(valid_urls),
                    "products_found": len([p for p in products if "error" not in p]),
                    "products": products
                }, indent=2)
                
            except asyncio.TimeoutError:
                logger.warning(f"Get products by URLs timeout for {len(valid_urls)} URLs")
                return json.dumps({
                    "error": "Request timed out",
                    "requested_urls": len(valid_urls),
                    "timeout": "30s"
                })
                
        except Exception as e:
            logger.error(f"Get products by URLs failed: {e}")
            return json.dumps({
                "error": f"Failed to get products: {str(e)}",
                "shop_id": shop_id
            })

    @mcp.tool(name=f"health_check_{shop_id}")
    async def health_check(ctx: Context) -> str:
        """Check the health status of the search service.

        This tool provides information about the status of Elasticsearch,
        embedding service, and the search index.

        Args:
            ctx: The MCP server provided context which includes the search engine

        Returns:
            JSON formatted health status information
        """
        try:
            search_engine = ctx.request_context.lifespan_context.search_engine
            shop_id = ctx.request_context.lifespan_context.shop_id
            
            health_status = search_engine.health_check()
            health_status["shop_id"] = shop_id
            
            return json.dumps(health_status, indent=2)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return json.dumps({
                "status": "unhealthy", 
                "error": str(e),
                "shop_id": ctx.request_context.lifespan_context.shop_id
            })

    @mcp.tool(name=f"add_to_cart_{shop_id}")
    async def add_to_cart(ctx: Context, product_id: str, quantity: int = 1) -> str:
        f"""Add a product to the shopping cart for {shop_id.replace('_', '-').upper()}.

        This tool adds products to the shopping cart. The cart persists during the session.

        Args:
            ctx: The MCP server provided context
            product_id: The product ID to add to cart
            quantity: The quantity to add (default: 1)

        Returns:
            JSON formatted response with the current cart contents including product URLs
        """
        try:
            # Use global cart store instead of lifespan context cart
            cart = _global_cart_store.get_cart(shop_id)
            search_engine = ctx.request_context.lifespan_context.search_engine
            shop_config = WEBMALL_SHOPS.get(shop_id, {})
            shop_url = shop_config.get("url", "")
            
            logger.debug(f"add_to_cart_{shop_id}: Current cart has {len(cart)} items before adding")
            
            # Normalize product_id to ensure consistent format (same as other components)
            normalized_product_id = str(product_id).strip()
            logger.debug(f"add_to_cart_{shop_id}: Normalized product_id from {repr(product_id)} to {repr(normalized_product_id)}")
            
            # Get product details with timeout (same pattern as other functions)
            try:
                product = await asyncio.wait_for(
                    asyncio.to_thread(search_engine.get_product_by_id, normalized_product_id),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Get product timeout for ID: {normalized_product_id} in add_to_cart")
                return json.dumps({
                    "error": "Product lookup timed out",
                    "product_id": normalized_product_id,
                    "timeout": "30s"
                })
            
            if not product:
                return json.dumps({
                    "error": f"Product {normalized_product_id} not found",
                    "cart": list(cart.values())
                })
            
            # Extract product information
            product_name = product.get("name", "Unknown Product")
            product_price = product.get("offers", {}).get("price", 0)
            
            # Get the slug from the URL if available, otherwise from product data
            product_url = product.get("url", "")
            if not product_url and "slug" in product:
                product_url = f"{shop_url}/product/{product['slug']}"
            elif not product_url:
                # Try to extract from additional fields
                product_url = f"{shop_url}/product/{normalized_product_id}"
            
            # Add or update cart item (use normalized ID as key)
            if normalized_product_id in cart:
                cart[normalized_product_id]["quantity"] += quantity
            else:
                cart[normalized_product_id] = {
                    "product_id": normalized_product_id,
                    "name": product_name,
                    "price": product_price,
                    "quantity": quantity,
                    "url": product_url
                }
            
            logger.debug(f"add_to_cart_{shop_id}: Cart now has {len(cart)} items after adding")
            
            return json.dumps({
                "message": f"Added {quantity} x {product_name} to cart",
                "cart": list(cart.values())
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Add to cart failed for product '{normalized_product_id if 'normalized_product_id' in locals() else product_id}': {e}")
            return json.dumps({
                "error": f"Failed to add to cart: {str(e)}",
                "product_id": normalized_product_id if 'normalized_product_id' in locals() else product_id,
                "shop_id": shop_id
            })

    @mcp.tool(name=f"view_cart_{shop_id}")
    async def view_cart(ctx: Context) -> str:
        f"""View the current shopping cart contents for {shop_id.replace('_', '-').upper()}.

        This tool returns the current contents of the shopping cart.

        Args:
            ctx: The MCP server provided context

        Returns:
            JSON formatted response with the cart contents
        """
        try:
            # Use global cart store instead of lifespan context cart
            cart = _global_cart_store.get_cart(shop_id)
            
            logger.debug(f"view_cart_{shop_id}: Cart has {len(cart)} items")
            
            return json.dumps({
                "shop_id": shop_id,
                "cart": list(cart.values()),
                "total_items": sum(item["quantity"] for item in cart.values())
            }, indent=2)
            
        except Exception as e:
            logger.error(f"View cart failed: {e}")
            return json.dumps({
                "error": f"Failed to view cart: {str(e)}",
                "shop_id": shop_id
            })

    @mcp.tool(name=f"reset_cart_{shop_id}")
    async def reset_cart(ctx: Context) -> str:
        f"""Clear the shopping cart for {shop_id.replace('_', '-').upper()}.

        This tool empties the shopping cart.

        Args:
            ctx: The MCP server provided context

        Returns:
            JSON formatted response confirming the cart was cleared
        """
        try:
            # Use global cart store instead of lifespan context cart
            _global_cart_store.clear_cart(shop_id)
            
            logger.debug(f"reset_cart_{shop_id}: Cart has been cleared")
            
            return json.dumps({
                "message": "Cart has been cleared",
                "shop_id": shop_id,
                "cart": []
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Reset cart failed: {e}")
            return json.dumps({
                "error": f"Failed to reset cart: {str(e)}",
                "shop_id": shop_id
            })

    @mcp.tool(name=f"checkout_{shop_id}")
    async def checkout(ctx: Context, 
                      first_name: str,
                      last_name: str,
                      email: str,
                      phone: str,
                      address_1: str,
                      city: str,
                      state: str,
                      postcode: str,
                      country: str,
                      credit_card_number: str,
                      credit_card_expiry: str,
                      credit_card_cvc: str) -> str:
        f"""Complete checkout and create an order for items in the cart for {shop_id.replace('_', '-').upper()}.

        This tool creates an order using the items currently in the shopping cart.

        Args:
            ctx: The MCP server provided context
            first_name: Customer's first name
            last_name: Customer's last name
            email: Customer's email address
            phone: Customer's phone number
            address_1: Customer's street address
            city: Customer's city
            state: Customer's state/province
            postcode: Customer's postal/zip code
            country: Customer's country code
            credit_card_number: The credit card number
            credit_card_expiry: The credit card expiry date (MM/YY)
            credit_card_cvc: The credit card CVC code

        Returns:
            JSON formatted response with order confirmation and product URLs
        """
        try:
            # Use global cart store instead of lifespan context cart
            cart = _global_cart_store.get_cart(shop_id)
            
            logger.debug(f"checkout_{shop_id}: Cart has {len(cart)} items at checkout")
            
            if not cart:
                return json.dumps({
                    "error": "Cart is empty. Please add items to cart before checkout.",
                    "shop_id": shop_id
                })
            
            # Validate credit card (basic check)
            if not all([credit_card_number, credit_card_expiry, credit_card_cvc]):
                return json.dumps({
                    "error": "Credit card details are incomplete."
                })
            
            # Create order ID
            order_id = f"{shop_id}_order_{str(uuid.uuid4())[:8]}"
            
            # Calculate total
            total_amount = sum(item["price"] * item["quantity"] for item in cart.values())
            
            # Get product URLs from cart
            product_urls = [item["url"] for item in cart.values()]
            
            # Create order response
            order_response = {
                "message": "Order created successfully",
                "order_id": order_id,
                "shop_id": shop_id,
                "status": "processing",
                "customer": {
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email,
                    "phone": phone
                },
                "shipping_address": {
                    "address_1": address_1,
                    "city": city,
                    "state": state,
                    "postcode": postcode,
                    "country": country
                },
                "items": list(cart.values()),
                "product_urls": product_urls,
                "total": f"{total_amount:.2f}",
                "payment_method": "credit_card",
                "transaction_id": str(uuid.uuid4())
            }
            
            # Clear cart after successful checkout
            _global_cart_store.clear_cart(shop_id)
            logger.debug(f"checkout_{shop_id}: Cleared cart after successful checkout")
            
            return json.dumps(order_response, indent=2)
            
        except Exception as e:
            logger.error(f"Checkout failed: {e}")
            return json.dumps({
                "error": f"Checkout failed: {str(e)}",
                "shop_id": shop_id
            })

    return mcp

def run_server_cli(shop_id: str, index_name: str, port: int):
    """Run server with command line interface"""
    
    parser = argparse.ArgumentParser(description=f'NLWeb MCP Server for {shop_id}')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--transport', choices=['stdio', 'sse'], default='stdio', 
                       help='Transport type (default: stdio)')
    parser.add_argument('--log-file', help='Log file path (optional)')
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter(f'[{shop_id.upper()}] %(levelname)s - %(message)s')
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified or if debug is enabled)
    if args.log_file or args.debug:
        log_file = args.log_file or f"/tmp/nlweb_mcp_{shop_id}.log"
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            print(f"Debug logging enabled. Logs saved to: {log_file}")
        except Exception as e:
            print(f"Warning: Could not create log file {log_file}: {e}")
    
    print(f"Starting {shop_id} MCP server on port {port} with index {index_name}")
    
    async def main():
        # Create and run server
        mcp = create_nlweb_server(shop_id, index_name, port)
        
        if args.transport == 'sse':
            await mcp.run_sse_async()
        else:
            await mcp.run_stdio_async()
    
    asyncio.run(main())