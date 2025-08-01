from utils import get_woocommerce_client
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from typing import Dict, Any

# Add the nlweb_mcp directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nlweb_mcp'))

from config import WEBMALL_SHOPS
from search_engine import SearchEngine
from embedding_service import EmbeddingService
from elasticsearch_client import ElasticsearchClient

load_dotenv()
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
class HybridSearchContext:
    """Context for the Hybrid Search MCP server."""
    search_engine: SearchEngine
    wc_client: object
    shop_id: str


@asynccontextmanager
async def hybrid_search_lifespan(server: FastMCP) -> AsyncIterator[HybridSearchContext]:
    """
    Manages the hybrid search engine lifecycle.

    Args:
        server: The FastMCP server instance

    Yields:
        HybridSearchContext: The context containing the search engine
    """
    # Get shop configuration for webmall_1 (E-Store Athletes)
    shop_id = "webmall_1"
    shop_config = WEBMALL_SHOPS[shop_id]

    try:
        # Initialize Elasticsearch client
        es_client = ElasticsearchClient()

        # Initialize embedding service
        embedding_service = EmbeddingService()

        # Test embedding service
        if not embedding_service.test_embedding_service():
            raise Exception("Embedding service test failed")

        # Initialize search engine
        search_engine = SearchEngine(
            es_client,
            embedding_service,
            shop_config["index_name"]
        )

        # Initialize WooCommerce client for this shop
        wc_client = get_woocommerce_client(
            wc_url=os.getenv('WOO_STORE_URL_1'),
            wc_consumer_key=os.getenv('WOO_CONSUMER_KEY_1'),
            wc_consumer_secret=os.getenv('WOO_CONSUMER_SECRET_1')
        )

        yield HybridSearchContext(
            search_engine=search_engine,
            wc_client=wc_client,
            shop_id=shop_id
        )
    except Exception as e:
        print(f"Failed to initialize hybrid search components: {e}")
        raise
    finally:
        print(f"Shutting down hybrid search server for {shop_id}")


# Initialize FastMCP server with the hybrid search context
mcp = FastMCP(
    "mcp-woocommerce-hybrid",
    description="E-Store Athletes MCP server for the best shopping experience",
    lifespan=hybrid_search_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8060")
)


@mcp.tool()
async def search_products(ctx: Context, query: str, per_page: int = 10, page: int = 1, include_descriptions: bool = False) -> str:
    """Search for products in E-Store Athletes using semantic search.

    This tool searches for products based on a search query string using
    powerful semantic search but.
    It returns product information including name, price, description, and other details.

    Args:
        ctx: The MCP server provided context which includes the search engine
        query: Search query string to find products
        per_page: Number of products to return per page (default: 10, max: 100)
        page: Page number for pagination (default: 1)

    Returns:
        JSON formatted list of products matching the search query
    """
    try:
        search_engine = ctx.request_context.lifespan_context.search_engine

        # Use the enhanced search engine with server_a formatting
        return search_engine.search_server_a_format(query, per_page, page, include_descriptions)

    except Exception as e:
        return f"Error searching products: {str(e)}"


@mcp.tool()
async def get_product(ctx: Context, product_name: str, include_descriptions: bool = True) -> str:
    """Get detailed information about a specific product by name using semantic search.

    This tool searches for products by name and returns comprehensive information about the first matching product.

    Args:
        ctx: The MCP server provided context which includes the search engine
        product_name: The exact name of the product to search for and retrieve

    Returns:
        JSON formatted detailed product information
    """
    try:
        search_engine = ctx.request_context.lifespan_context.search_engine

        # Use the enhanced search engine with server_a formatting
        return search_engine.search_server_a_format(product_name, 1, 1, include_descriptions)

    except Exception as e:
        return f"Error searching products: {str(e)}"


@mcp.tool()
async def get_detailed_products(ctx: Context, product_ids: list[str]) -> str:
    """Get detailed information about specific products by their IDs.

    This tool retrieves comprehensive product information for a list of product IDs,
    returning all available details including descriptions, specifications, and metadata.

    Args:
        ctx: The MCP server provided context which includes the search engine
        product_ids: List of product IDs to retrieve detailed information for

    Returns:
        JSON formatted list of detailed product information in E-Store Athletes format
    """
    try:
        search_engine = ctx.request_context.lifespan_context.search_engine
        
        # Get detailed products by URLs (assuming IDs can be converted to URLs)
        detailed_products = []
        for product_id in product_ids:
            try:
                product = search_engine.get_product_by_id(product_id)
                if product:
                    # Format in server_a style with full details
                    schema_org = product.get("schema_object", {})
                    
                    # Extract price information with fallbacks
                    price_info = schema_org.get("offers", {})
                    if isinstance(price_info, list) and price_info:
                        price_info = price_info[0]
                    
                    current_price = price_info.get("price") or product.get("price", 0)
                    
                    detailed_product = {
                        "ID": product_id,
                        "label": product.get("name", ""),
                        "shortCode": product.get("url", "").split("/")[-1] if product.get("url") else "",
                        
                        "priceInfo": {
                            "current": str(current_price) if current_price else "",
                            "usual": price_info.get("highPrice", ""),
                            "dealPrice": str(current_price) if current_price else "",
                            "isOnDeal": bool(current_price),
                        },
                        
                        "desc": {
                            "longVersion": product.get("description", ""),
                        },
                        
                        "stock": {
                            "itemCode": str(product_id),
                            "status": "In stock",
                            "leftOverCount": schema_org.get("inventoryLevel", 0),
                        },
                        
                        "labels": {
                            "categories": [product.get("category", "")] if product.get("category") else [],
                            "tags": schema_org.get("keywords", "").split(",") if schema_org.get("keywords") else [],
                        },
                        
                        "snapshots": [schema_org.get("image", "")][:2] if schema_org.get("image") else [],
                        
                        "addresses": {
                            "selfLink": product.get("url", ""),
                            "shareLink": product.get("url", ""),
                        },
                        
                        "fullDetails": {
                            "schema_org": schema_org,
                            "metadata": {
                                "site": product.get("site", ""),
                                "last_updated": "N/A"
                            }
                        }
                    }
                    detailed_products.append(detailed_product)
                else:
                    # Add error entry for missing product
                    detailed_products.append({
                        "ID": product_id,
                        "error": f"Product {product_id} not found",
                        "status": "not_found"
                    })
            except Exception as e:
                detailed_products.append({
                    "ID": product_id,
                    "error": f"Error retrieving product {product_id}: {str(e)}",
                    "status": "error"
                })
        
        return json.dumps({
            "detailsRequest": {
                "requestedIds": product_ids,
                "foundCount": len([p for p in detailed_products if "error" not in p]),
                "errorCount": len([p for p in detailed_products if "error" in p])
            },
            "detailedProducts": detailed_products
        }, indent=2)
        
    except Exception as e:
        return f"Error retrieving detailed product information: {str(e)}"


@mcp.tool()
async def get_categories(ctx: Context, per_page: int = 50, page: int = 1, parent: int = None) -> str:
    """Get product categories.

    This tool retrieves product categories with their hierarchy and details.

    Args:
        ctx: The MCP server provided context which includes the search engine
        per_page: Number of categories to return per page (default: 50, max: 100)
        page: Page number for pagination (default: 1)
        parent: Parent category ID to get subcategories (optional, not used in semantic search)

    Returns:
        JSON formatted list of product categories
    """
    try:
        wc_client = ctx.request_context.lifespan_context.wc_client

        # Validate per_page parameter
        per_page = min(max(1, per_page), 100)

        # Build parameters for the API call
        params = {
            "per_page": per_page,
            "page": page,
            "hide_empty": False  # Include categories without products
        }

        # Add parent filter if specified
        if parent is not None:
            params["parent"] = parent

        # Get categories using WooCommerce API
        response = wc_client.get("products/categories", params=params)

        if response.status_code == 200:
            raw_categories = response.json()
            oddball_categories = []
            for cat in raw_categories:
                oddball_categories.append({
                    "CatID": cat.get("id"),
                    "title": cat.get("name"),
                    "shortHandle": cat.get("slug"),
                    "familyTree": cat.get("parent"),
                    "details": {
                        "blurb": cat.get("description"),
                        "showStyle": cat.get("display"),
                        "sortOrder": cat.get("menu_order"),
                        "itemCount": cat.get("count"),
                    },
                    "thumbnail": cat.get("image"),
                    "connections": {
                        "rawLinks": cat.get("_links", {})
                    }
                })

            return json.dumps({
                "pageTracker": {
                    "currentPage": page,
                    "perPageLimit": per_page
                },
                "parentFilter": parent,
                "totalCatsFound": len(oddball_categories),
                "categoriesList": oddball_categories
            }, indent=2)
        else:
            return f"Error retrieving categories: HTTP {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error retrieving categories: {str(e)}"


@mcp.tool()
async def add_to_cart(ctx: Context, product_id: str, quantity: int = 1) -> str:
    """Add a product to the shopping cart.

    This tool adds products to the shopping cart. The cart persists during the session.

    Args:
        ctx: The MCP server provided context
        product_id: The product ID to add to cart
        quantity: The quantity to add (default: 1)

    Returns:
        JSON formatted response with the current cart contents in E-Store Athletes "weird" format
    """
    try:
        shop_id = ctx.request_context.lifespan_context.shop_id
        search_engine = ctx.request_context.lifespan_context.search_engine
        shop_config = WEBMALL_SHOPS.get(shop_id, {})
        shop_url = shop_config.get("url", "")
        
        # Use global cart store
        cart = _global_cart_store.get_cart(shop_id)
        
        logger.debug(f"add_to_cart_{shop_id}: Current cart has {len(cart)} items before adding")
        
        # Normalize product_id
        normalized_product_id = str(product_id).strip()
        logger.debug(f"add_to_cart_{shop_id}: Normalized product_id from {repr(product_id)} to {repr(normalized_product_id)}")
        
        # Get product details
        try:
            product = await asyncio.wait_for(
                asyncio.to_thread(search_engine.get_product_by_id, normalized_product_id),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Get product timeout for ID: {normalized_product_id} in add_to_cart")
            return json.dumps({
                "error": "Product lookup timed out",
                "productReference": normalized_product_id,
                "timeout": "30s"
            })
        
        if not product:
            return json.dumps({
                "error": f"Product {normalized_product_id} not found",
                "basketContents": {"addresses": {"selfLink": "", "shareLink": ""}, "items": []}
            })
        
        # Extract product information
        product_name = product.get("name", "Unknown Product")
        product_price = product.get("offers", {}).get("price", 0)
        
        # Get the product URL
        product_url = product.get("url", "")
        if not product_url and "slug" in product:
            product_url = f"{shop_url}/product/{product['slug']}"
        elif not product_url:
            product_url = f"{shop_url}/product/{normalized_product_id}"
        
        # Add or update cart item
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
        
        # Format response in "weird" server A format
        cart_items = []
        for item in cart.values():
            cart_items.append({
                "itemReference": item["product_id"],
                "displayName": item["name"],
                "unitCost": item["price"],
                "quantitySelected": item["quantity"],
                "addresses": {
                    "selfLink": item["url"],
                    "shareLink": item["url"]
                }
            })
        
        return json.dumps({
            "message": f"Added {quantity} x {product_name} to basket",
            "basketContents": {
                "addresses": {
                    "selfLink": f"{shop_url}/cart",
                    "shareLink": f"{shop_url}/cart"
                },
                "items": cart_items,
                "totalItems": sum(item["quantity"] for item in cart.values())
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Add to cart failed for product '{normalized_product_id if 'normalized_product_id' in locals() else product_id}': {e}")
        return json.dumps({
            "error": f"Failed to add to basket: {str(e)}",
            "productReference": normalized_product_id if 'normalized_product_id' in locals() else product_id
        })


@mcp.tool()
async def view_cart(ctx: Context) -> str:
    """View the current shopping cart contents.

    This tool returns the current contents of the shopping cart in E-Store Athletes "weird" format.

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON formatted response with the cart contents
    """
    try:
        shop_id = ctx.request_context.lifespan_context.shop_id
        shop_config = WEBMALL_SHOPS.get(shop_id, {})
        shop_url = shop_config.get("url", "")
        
        # Use global cart store
        cart = _global_cart_store.get_cart(shop_id)
        
        logger.debug(f"view_cart_{shop_id}: Cart has {len(cart)} items")
        
        # Format cart in "weird" server A format
        cart_items = []
        for item in cart.values():
            cart_items.append({
                "itemReference": item["product_id"],
                "displayName": item["name"],
                "unitCost": item["price"],
                "quantitySelected": item["quantity"],
                "addresses": {
                    "selfLink": item["url"],
                    "shareLink": item["url"]
                }
            })
        
        return json.dumps({
            "basketContents": {
                "addresses": {
                    "selfLink": f"{shop_url}/cart",
                    "shareLink": f"{shop_url}/cart"
                },
                "items": cart_items,
                "totalItems": sum(item["quantity"] for item in cart.values()),
                "basketValue": sum(item["price"] * item["quantity"] for item in cart.values())
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"View cart failed: {e}")
        return json.dumps({
            "error": f"Failed to view basket: {str(e)}"
        })


@mcp.tool()
async def reset_cart(ctx: Context) -> str:
    """Clear the shopping cart.

    This tool empties the shopping cart.

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON formatted response confirming the cart was cleared
    """
    try:
        shop_id = ctx.request_context.lifespan_context.shop_id
        shop_config = WEBMALL_SHOPS.get(shop_id, {})
        shop_url = shop_config.get("url", "")
        
        # Use global cart store
        _global_cart_store.clear_cart(shop_id)
        
        logger.debug(f"reset_cart_{shop_id}: Cart has been cleared")
        
        return json.dumps({
            "message": "Basket has been cleared",
            "basketContents": {
                "addresses": {
                    "selfLink": f"{shop_url}/cart",
                    "shareLink": f"{shop_url}/cart"
                },
                "items": [],
                "totalItems": 0,
                "basketValue": 0.0
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Reset cart failed: {e}")
        return json.dumps({
            "error": f"Failed to reset basket: {str(e)}"
        })


@mcp.tool()
async def checkout(
    ctx: Context,
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
    credit_card_cvc: str
) -> str:
    """Complete checkout and create an order for items in the cart.

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
        JSON formatted response with order confirmation and product URLs in "weird" format
    """
    try:
        shop_id = ctx.request_context.lifespan_context.shop_id
        shop_config = WEBMALL_SHOPS.get(shop_id, {})
        shop_url = shop_config.get("url", "")
        
        # Use global cart store
        cart = _global_cart_store.get_cart(shop_id)
        
        logger.debug(f"checkout_{shop_id}: Cart has {len(cart)} items at checkout")
        
        if not cart:
            return json.dumps({
                "error": "Basket is empty. Please add items to basket before checkout.",
                "orderStatus": "failed"
            })
        
        # Validate credit card (basic check)
        if not all([credit_card_number, credit_card_expiry, credit_card_cvc]):
            return json.dumps({
                "error": "Credit card details are incomplete.",
                "orderStatus": "failed"
            })
        
        # Create order ID
        order_id = f"{shop_id}_order_{str(uuid.uuid4())[:8]}"
        
        # Calculate total
        total_amount = sum(item["price"] * item["quantity"] for item in cart.values())
        
        # Get product URLs from cart
        product_addresses = []
        for item in cart.values():
            product_addresses.append({
                "selfLink": item["url"],
                "shareLink": item["url"]
            })
        
        # Format order items in "weird" server A format
        order_items = []
        for item in cart.values():
            order_items.append({
                "itemReference": item["product_id"],
                "displayName": item["name"],
                "unitCost": item["price"],
                "quantityOrdered": item["quantity"],
                "addresses": {
                    "selfLink": item["url"],
                    "shareLink": item["url"]
                }
            })
        
        # Create order response in "weird" format
        order_response = {
            "message": "Order created and processed successfully",
            "orderDetails": {
                "orderReference": order_id,
                "orderStatus": "processing",
                "customerInfo": {
                    "fullName": f"{first_name} {last_name}",
                    "contactEmail": email,
                    "phoneNumber": phone
                },
                "shippingAddress": {
                    "streetAddress": address_1,
                    "cityName": city,
                    "stateRegion": state,
                    "postalCode": postcode,
                    "countryCode": country
                },
                "orderItems": order_items,
                "productAddresses": product_addresses,
                "orderValue": f"{total_amount:.2f}",
                "paymentMethod": "credit_card",
                "transactionReference": str(uuid.uuid4())
            },
            "addresses": {
                "orderSelfLink": f"{shop_url}/order/{order_id}",
                "orderShareLink": f"{shop_url}/order/{order_id}"
            }
        }
        
        # Clear cart after successful checkout
        _global_cart_store.clear_cart(shop_id)
        logger.debug(f"checkout_{shop_id}: Cleared cart after successful checkout")
        
        return json.dumps(order_response, indent=2)
        
    except Exception as e:
        logger.error(f"Checkout failed: {e}")
        return json.dumps({
            "error": f"Checkout failed: {str(e)}",
            "orderStatus": "failed"
        })


@mcp.tool()
async def checkout_products(
    ctx: Context,
    product_ids: list[int],
    quantities: list[int],
    first_name: str,
    last_name: str,
    address_1: str,
    city: str,
    state: str,
    postcode: str,
    country: str,
    email: str,
    phone: str,
    credit_card_number: str,
    credit_card_expiry: str,
    credit_card_cvc: str
) -> str:
    """Creates and pays for a new order with specified products (legacy direct checkout).

    This tool creates an order with the given products, customer details, 
    and credit card without using the cart.

    Args:
        ctx: The MCP server provided context
        product_ids: A list of product IDs to be included in the order
        quantities: A list of quantities corresponding to the product_ids
        first_name: Customer's first name
        last_name: Customer's last name  
        address_1: Customer's street address
        city: Customer's city
        state: Customer's state/province
        postcode: Customer's postal/zip code
        country: Customer's country code (e.g., 'US')
        email: Customer's email address
        phone: Customer's phone number
        credit_card_number: The credit card number for payment
        credit_card_expiry: The credit card expiry date (e.g., "MM/YY")
        credit_card_cvc: The credit card CVC code

    Returns:
        A JSON formatted string with the order confirmation details
    """
    if len(product_ids) != len(quantities):
        return json.dumps({
            "error": "The number of product IDs must match the number of quantities.",
            "orderStatus": "failed"
        })

    try:
        # Basic validation for credit card details (for simulation purposes)
        if not (credit_card_number and credit_card_expiry and credit_card_cvc):
            return json.dumps({
                "error": "Credit card details are incomplete.",
                "orderStatus": "failed"
            })

        # Simulate order creation
        order_id = str(uuid.uuid4())[:8]
        transaction_id = str(uuid.uuid4())

        # Calculate total (simulation)
        total_amount = sum(qty * 10.00 for qty in quantities)

        return json.dumps({
            "message": "Order created and paid successfully.",
            "orderDetails": {
                "orderReference": order_id,
                "orderStatus": "processing",
                "orderValue": f"{total_amount:.2f}",
                "transactionReference": transaction_id,
                "note": "This is a simulated direct order using hybrid semantic search."
            }
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Error creating order: {str(e)}",
            "orderStatus": "failed"
        })


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())
