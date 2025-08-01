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

from elasticsearch_client import ElasticsearchClient
from embedding_service import EmbeddingService
from search_engine import SearchEngine
from config import WEBMALL_SHOPS
from utils import get_woocommerce_client

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
    # Get shop configuration for webmall_2 (TechTalk)
    shop_id = "webmall_2"
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
            wc_url=os.getenv('WOO_STORE_URL_2'), 
            wc_consumer_key=os.getenv('WOO_CONSUMER_KEY_2'), 
            wc_consumer_secret=os.getenv('WOO_CONSUMER_SECRET_2')
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


mcp_product_catalog = FastMCP(
    "mcp-product-catalog-hybrid",
    description="TechTalk shopping server for AI agents",
    lifespan=hybrid_search_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT_PRODUCT_CATALOG", "8061")
)


@mcp_product_catalog.tool()
async def find_items_techtalk(ctx: Context, query: str, limit: int = 5, page_num: int = 1, sort_by_price: str = "none", include_descriptions: bool = False) -> str:
    """Search for items in the product catalog of TechTalk.

    This tool searches for items based on a search query string using
    powerful semantic search. It returns item information including name, price, description, and other details.
    Can optionally sort results by price.

    Args:
        ctx: The MCP server provided context which includes the search engine
        query: Search query string to find items
        limit: Number of items to return per page (default: 5, max: 100)
        page_num: Page number for pagination (default: 1)
        sort_by_price: Sort results by price - "asc" for low to high, "desc" for high to low, "none" for no sorting (default: "none")

    Returns:
        JSON formatted list of items matching the search query
    """
    try:
        search_engine = ctx.request_context.lifespan_context.search_engine

        # Use the enhanced search engine with server_b formatting
        return search_engine.search_server_b_format(query, limit, page_num, sort_by_price, include_descriptions)

    except Exception as e:
        return f"Error searching items: {str(e)}"


@mcp_product_catalog.tool()
async def get_detailed_items_techtalk(ctx: Context, product_ids: list[str]) -> str:
    """Get detailed information about specific items by their IDs.

    This tool retrieves comprehensive item information for a list of product IDs,
    returning all available details including descriptions, specifications, and metadata.

    Args:
        ctx: The MCP server provided context which includes the search engine
        product_ids: List of product IDs to retrieve detailed information for

    Returns:
        JSON formatted list of detailed item information in TechTalk format
    """
    try:
        search_engine = ctx.request_context.lifespan_context.search_engine
        
        # Get detailed products
        detailed_items = []
        for product_id in product_ids:
            try:
                product = search_engine.get_product_by_id(product_id)
                if product:
                    # Format in server_b style with full details
                    schema_org = product.get("schema_object", {})
                    
                    # Extract price information with fallbacks
                    price_info = schema_org.get("offers", {})
                    if isinstance(price_info, list) and price_info:
                        price_info = price_info[0]
                    
                    cost_amount = price_info.get("price") or product.get("price", 0)
                    
                    detailed_item = {
                        "catalog_entry_id": product_id,
                        "merchandise_title": product.get("name", ""),
                        
                        "financial_details": {
                            "cost_amount": str(cost_amount) if cost_amount else "",
                            "standard_rate": price_info.get("highPrice", ""),
                            "discount_rate": str(cost_amount) if cost_amount else "",
                            "bargain_active": bool(cost_amount),
                        },
                        
                        "content_sections": {
                            "detailed_info": product.get("description", ""),
                        },
                        
                        "inventory_tracking": {
                            "product_identifier": str(product_id),
                            "availability_state": "On the shelf",
                            "units_remaining": schema_org.get("inventoryLevel", 0),
                        },
                        
                        "classification_tags": [product.get("category", "")] if product.get("category") else [],
                        "visual_assets": [schema_org.get("image", "")][:2] if schema_org.get("image") else [],
                        "direct_link": product.get("url", ""),
                        
                        "comprehensive_metadata": {
                            "schema_org": schema_org,
                            "site_info": product.get("site", ""),
                            "last_indexed": "N/A"
                        }
                    }
                    detailed_items.append(detailed_item)
                else:
                    # Add error entry for missing product
                    detailed_items.append({
                        "catalog_entry_id": product_id,
                        "error": f"Item {product_id} not found",
                        "status": "not_found"
                    })
            except Exception as e:
                detailed_items.append({
                    "catalog_entry_id": product_id,
                    "error": f"Error retrieving item {product_id}: {str(e)}",
                    "status": "error"
                })
        
        return json.dumps({
            "lookup_summary": {
                "requested_ids": product_ids,
                "items_found": len([i for i in detailed_items if "error" not in i]),
                "lookup_errors": len([i for i in detailed_items if "error" in i])
            },
            "catalog_entries": detailed_items
        }, indent=2)
        
    except Exception as e:
        return f"Error retrieving detailed item information: {str(e)}"


@mcp_product_catalog.tool()
async def retrieve_item_details_techtalk(ctx: Context, item_identifier: int) -> str:
    """Get detailed information about a specific item using WooCommerce API.

    This tool retrieves comprehensive information about a single item by searching
    the WooCommerce API and returns data.

    Args:
        ctx: The MCP server provided context which includes the WooCommerce client
        item_identifier: The ID of the item to retrieve (product identifier)

    Returns:
        JSON formatted detailed item information
    """
    try:
        wc_client = ctx.request_context.lifespan_context.wc_client

        # Get product by ID using WooCommerce API
        response = wc_client.get(f"products/{item_identifier}")
        
        if response.status_code == 200:
            product = response.json()
        elif response.status_code == 404:
            return f"Item with ID {item_identifier} not found"
        else:
            return f"Error retrieving item: HTTP {response.status_code} - {response.text}"

        # Transform to server_b detailed format
        # Extract price information
        price_info = {
            "price": product.get("price"),
            "highPrice": product.get("regular_price"),
            "lowPrice": product.get("sale_price")
        }

        elaborate_product = {
            "lookup_results": {
                "requested_sku_code": item_identifier,
                "discovery_status": "found",
                "catalog_position": 1,
            },

            "core_identity": {
                "product_identifier": product.get("id"),
                "display_label": product.get("name", ""),
                "variant_type": product.get("type", "simple"),
                "publication_state": product.get("status", "publish"),
            },

            "pricing_structure": {
                "current_value": price_info.get("price", ""),
                "base_price": price_info.get("highPrice", ""),
                "promotional_price": price_info.get("lowPrice", ""),
                "deal_status": product.get("on_sale", False),
            },

            "narrative_content": {
                "full_description": product.get("description", ""),
                "elevator_pitch": product.get("short_description", ""),
            },

            "warehouse_data": {
                "sku_reference": product.get("sku", ""),
                "stock_condition": product.get("stock_status", "outofstock"),
                "inventory_count": product.get("stock_quantity", 0),
                "tracking_enabled": product.get("manage_stock", False),
            },

            "taxonomy_data": {
                "category_assignments": [{"name": cat.get("name", "")} for cat in product.get("categories", [])],
                "keyword_tags": [{"name": tag.get("name", "")} for tag in product.get("tags", [])],
            },

            "media_gallery": [{"src": img.get("src", "")} for img in product.get("images", [])],
            "storefront_url": product.get("permalink", ""),

            "metadata_extras": {
                "creation_timestamp": product.get("date_created", ""),
                "modification_timestamp": product.get("date_modified", ""),
                "menu_position": product.get("menu_order", 0),
            }
        }

        return json.dumps(elaborate_product, indent=2)

    except Exception as e:
        return f"Error retrieving item: {str(e)}"


@mcp_product_catalog.tool()
async def list_item_categories_techtalk(ctx: Context, items_per_page: int = 30, page_number: int = 1) -> str:
    """Get item categories using WooCommerce API.

    This tool retrieves item categories with their hierarchy and details
    from the WooCommerce API.

    Args:
        ctx: The MCP server provided context which includes the WooCommerce client
        items_per_page: Number of categories to return per page (default: 30, max: 100)
        page_number: Page number for pagination (default: 1)

    Returns:
        JSON formatted list of item categories
    """
    try:
        wc_client = ctx.request_context.lifespan_context.wc_client

        # Validate items_per_page parameter
        items_per_page = min(max(1, items_per_page), 100)

        # Build parameters for the API call
        params = {
            "per_page": items_per_page,
            "page": page_number,
            "hide_empty": False  # Include categories without products
        }

        # Get categories using WooCommerce API
        response = wc_client.get("products/categories", params=params)

        if response.status_code == 200:
            raw_categories = response.json()
            
            # Format as server_b style categories
            whimsical_categories = []
            for i, cat in enumerate(raw_categories):
                whimsical_categories.append({
                    "taxonomy_node_id": cat.get("id"),
                    "classification_name": cat.get("name", ""),
                    "url_fragment": cat.get("slug", ""),
                    "hierarchy_parent": cat.get("parent", 0),
                    "descriptive_text": cat.get("description", ""),
                    "merchandise_tally": cat.get("count", 0),
                    "visual_representation": cat.get("image", {}).get("src") if cat.get("image") else None,
                    "display_priority": cat.get("menu_order", i),
                })

            return json.dumps({
                "navigation_context": {
                    "current_position": page_number,
                    "results_limit": items_per_page,
                },
                "discovery_metrics": {
                    "categories_located": len(whimsical_categories),
                    "has_subcategories": any(cat.get("hierarchy_parent") != 0 for cat in whimsical_categories),
                },
                "taxonomy_structure": whimsical_categories
            }, indent=2)
        else:
            return f"Error retrieving categories: HTTP {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error retrieving categories: {str(e)}"


@mcp_product_catalog.tool()
async def add_to_shopping_cart_techtalk(ctx: Context, product_id: str, quantity: int = 1) -> str:
    """Add a product to the shopping cart in TechTalk.

    This tool adds products to the shopping cart. The cart persists during the session.

    Args:
        ctx: The MCP server provided context
        product_id: The product ID to add to cart
        quantity: The quantity to add (default: 1)

    Returns:
        JSON formatted response with the current cart contents in TechTalk "bizarre" format
    """
    try:
        shop_id = ctx.request_context.lifespan_context.shop_id
        search_engine = ctx.request_context.lifespan_context.search_engine
        shop_config = WEBMALL_SHOPS.get(shop_id, {})
        shop_url = shop_config.get("url", "")
        
        # Use global cart store
        cart = _global_cart_store.get_cart(shop_id)
        
        logger.debug(f"add_to_shopping_cart_techtalk_{shop_id}: Current cart has {len(cart)} items before adding")
        
        # Normalize product_id
        normalized_product_id = str(product_id).strip()
        logger.debug(f"add_to_shopping_cart_techtalk_{shop_id}: Normalized product_id from {repr(product_id)} to {repr(normalized_product_id)}")
        
        # Get product details
        try:
            product = await asyncio.wait_for(
                asyncio.to_thread(search_engine.get_product_by_id, normalized_product_id),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Get product timeout for ID: {normalized_product_id} in add_to_shopping_cart_techtalk")
            return json.dumps({
                "transaction_outcome": {
                    "completion_status": "failed", 
                    "error_description": "Product lookup timed out"
                },
                "product_reference": normalized_product_id,
                "timeout_duration": "30s"
            })
        
        if not product:
            return json.dumps({
                "transaction_outcome": {
                    "completion_status": "failed",
                    "error_description": f"Product {normalized_product_id} not found"
                },
                "shopping_container": {"catalog_entries": [], "total_line_items": 0}
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
        
        logger.debug(f"add_to_shopping_cart_techtalk_{shop_id}: Cart now has {len(cart)} items after adding")
        
        # Format response in "bizarre" server B format
        cart_entries = []
        for item in cart.values():
            cart_entries.append({
                "item_identifier": item["product_id"],
                "product_label": item["name"],
                "unit_pricing": item["price"],
                "quantity_selected": item["quantity"],
                "direct_link": item["url"],
                "line_total": item["price"] * item["quantity"]
            })
        
        return json.dumps({
            "transaction_outcome": {
                "completion_status": "successful",
                "action_description": f"Added {quantity} x {product_name} to shopping cart"
            },
            "shopping_container": {
                "catalog_entries": cart_entries,
                "total_line_items": sum(item["quantity"] for item in cart.values()),
                "container_value": sum(item["price"] * item["quantity"] for item in cart.values()),
                "cart_reference_url": f"{shop_url}/cart"
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Add to shopping cart failed for product '{normalized_product_id if 'normalized_product_id' in locals() else product_id}': {e}")
        return json.dumps({
            "transaction_outcome": {
                "completion_status": "failed",
                "error_description": f"Failed to add to shopping cart: {str(e)}"
            },
            "product_reference": normalized_product_id if 'normalized_product_id' in locals() else product_id
        })


@mcp_product_catalog.tool()
async def view_shopping_cart_techtalk(ctx: Context) -> str:
    """View the current shopping cart contents in TechTalk.

    This tool returns the current contents of the shopping cart in TechTalk "bizarre" format.

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
        
        logger.debug(f"view_shopping_cart_techtalk_{shop_id}: Cart has {len(cart)} items")
        
        # Format cart in "bizarre" server B format
        cart_entries = []
        for item in cart.values():
            cart_entries.append({
                "item_identifier": item["product_id"],
                "product_label": item["name"],
                "unit_pricing": item["price"],
                "quantity_selected": item["quantity"],
                "direct_link": item["url"],
                "line_total": item["price"] * item["quantity"]
            })
        
        return json.dumps({
            "shopping_container": {
                "catalog_entries": cart_entries,
                "total_line_items": sum(item["quantity"] for item in cart.values()),
                "container_value": sum(item["price"] * item["quantity"] for item in cart.values()),
                "cart_reference_url": f"{shop_url}/cart",
                "container_status": "active" if cart else "empty"
            },
            "navigation_context": {
                "checkout_available": len(cart) > 0,
                "modification_allowed": True
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"View shopping cart failed: {e}")
        return json.dumps({
            "transaction_outcome": {
                "completion_status": "failed",
                "error_description": f"Failed to view shopping cart: {str(e)}"
            }
        })


@mcp_product_catalog.tool()
async def clear_shopping_cart_techtalk(ctx: Context) -> str:
    """Clear the shopping cart in TechTalk.

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
        
        logger.debug(f"clear_shopping_cart_techtalk_{shop_id}: Cart has been cleared")
        
        return json.dumps({
            "transaction_outcome": {
                "completion_status": "successful",
                "action_description": "Shopping cart has been cleared"
            },
            "shopping_container": {
                "catalog_entries": [],
                "total_line_items": 0,
                "container_value": 0.0,
                "cart_reference_url": f"{shop_url}/cart",
                "container_status": "empty"
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Clear shopping cart failed: {e}")
        return json.dumps({
            "transaction_outcome": {
                "completion_status": "failed",
                "error_description": f"Failed to clear shopping cart: {str(e)}"
            }
        })


@mcp_product_catalog.tool()
async def checkout_cart_techtalk(
    ctx: Context,
    customer_first_name: str,
    customer_last_name: str,
    customer_email: str,
    customer_phone: str,
    shipping_street: str,
    shipping_city: str,
    shipping_state: str,
    shipping_zip: str,
    shipping_country_code: str,
    payment_card_number: str,
    card_expiration_date: str,
    card_security_code: str
) -> str:
    """Complete checkout and create an order for items in the cart in TechTalk.

    This tool creates an order using the items currently in the shopping cart.

    Args:
        ctx: The MCP server provided context
        customer_first_name: Customer's first name
        customer_last_name: Customer's last name
        customer_email: Customer's email address
        customer_phone: Customer's phone number
        shipping_street: Customer's street address
        shipping_city: Customer's city
        shipping_state: Customer's state/province
        shipping_zip: Customer's postal/zip code
        shipping_country_code: Customer's country code
        payment_card_number: The credit card number
        card_expiration_date: The credit card expiry date (MM/YY)
        card_security_code: The credit card CVC code

    Returns:
        JSON formatted response with order confirmation and product URLs in "bizarre" format
    """
    try:
        shop_id = ctx.request_context.lifespan_context.shop_id
        shop_config = WEBMALL_SHOPS.get(shop_id, {})
        shop_url = shop_config.get("url", "")
        
        # Use global cart store
        cart = _global_cart_store.get_cart(shop_id)
        
        logger.debug(f"checkout_cart_techtalk_{shop_id}: Cart has {len(cart)} items at checkout")
        
        if not cart:
            return json.dumps({
                "transaction_outcome": {
                    "completion_status": "failed",
                    "error_description": "Shopping cart is empty. Please add items to cart before checkout."
                }
            })
        
        # Validate credit card (basic check)
        if not all([payment_card_number, card_expiration_date, card_security_code]):
            return json.dumps({
                "transaction_outcome": {
                    "completion_status": "failed",
                    "error_description": "Credit card details are incomplete."
                }
            })
        
        # Create order ID
        order_id = f"{shop_id}_order_{str(uuid.uuid4())[:8]}"
        
        # Calculate total
        total_amount = sum(item["price"] * item["quantity"] for item in cart.values())
        
        # Get product URLs from cart
        direct_links = [item["url"] for item in cart.values()]
        
        # Format order items in "bizarre" server B format
        order_entries = []
        for item in cart.values():
            order_entries.append({
                "item_identifier": item["product_id"],
                "product_label": item["name"],
                "unit_pricing": item["price"],
                "quantity_ordered": item["quantity"],
                "direct_link": item["url"],
                "line_total": item["price"] * item["quantity"]
            })
        
        # Create order response in "bizarre" format
        order_response = {
            "transaction_outcome": {
                "completion_status": "successful",
                "celebration_message": "Your order has been successfully placed and processed"
            },
            "order_documentation": {
                "reference_number": order_id,
                "fulfillment_stage": "processing",
                "financial_charge": f"{total_amount:.2f}",
                "payment_tracking_id": str(uuid.uuid4()),
                "customer_record": {
                    "billing_name": f"{customer_first_name} {customer_last_name}",
                    "contact_email": customer_email,
                    "phone_contact": customer_phone,
                    "delivery_destination": f"{shipping_street}, {shipping_city}, {shipping_state} {shipping_zip}, {shipping_country_code}"
                }
            },
            "purchase_summary": {
                "catalog_entries": order_entries,
                "direct_links": direct_links,
                "total_line_items": len(cart),
                "container_value": total_amount
            }
        }
        
        # Clear cart after successful checkout
        _global_cart_store.clear_cart(shop_id)
        logger.debug(f"checkout_cart_techtalk_{shop_id}: Cleared cart after successful checkout")
        
        return json.dumps(order_response, indent=2)
        
    except Exception as e:
        logger.error(f"Checkout cart failed: {e}")
        return json.dumps({
            "transaction_outcome": {
                "completion_status": "failed",
                "error_description": f"Checkout failed: {str(e)}"
            }
        })


@mcp_product_catalog.tool()
async def process_customer_order(
    ctx: Context,
    item_codes: list[int],
    item_counts: list[int],
    customer_first_name: str,
    customer_last_name: str,
    shipping_street: str,
    shipping_city: str,
    shipping_state: str,
    shipping_zip: str,
    shipping_country_code: str,
    customer_email: str,
    customer_phone: str,
    payment_card_number: str,
    card_expiration_date: str,
    card_security_code: str
) -> str:
    """Processes a customer's order and makes payment in the TechTalk store.

    This tool handles the creation and payment for an order.

    Args:
        ctx: The MCP server context containing the search engine
        item_codes: A list of product identifiers (SKU codes) for the order
        item_counts: A list of quantities for each corresponding product
        customer_first_name: The customer's first name
        customer_last_name: The customer's last name
        shipping_street: The street address for shipping
        shipping_city: The city for shipping
        shipping_state: The state/province for shipping
        shipping_zip: The postal/zip code for shipping
        shipping_country_code: The two-letter country code for shipping (e.g., 'US')
        customer_email: The customer's email address
        customer_phone: The customer's phone number
        payment_card_number: The credit card number for payment
        card_expiration_date: The card's expiration date (e.g., "MM/YY")
        card_security_code: The card's CVC/CVV security code

    Returns:
        A JSON formatted string with order confirmation details or an error message
    """
    if len(item_codes) != len(item_counts):
        return "Error: The number of item codes must correspond to the number of item counts."

    try:
        if not (payment_card_number and card_expiration_date and card_security_code):
            return "Error: Payment card details are missing or incomplete."

        # Simulate order creation
        order_id = str(uuid.uuid4())[:8]
        transaction_id = str(uuid.uuid4())
        
        # Calculate total (simulation)
        total_amount = sum(qty * 15.00 for qty in item_counts)

        return json.dumps({
            "transaction_outcome": {
                "completion_status": "successful",
                "celebration_message": "Your order has been successfully placed and paid (SIMULATION).",
            },
            "order_documentation": {
                "reference_number": order_id,
                "fulfillment_stage": "processing",
                "financial_charge": f"{total_amount:.2f}",
                "payment_tracking_id": transaction_id,
            },
            "purchase_summary": {
                "sku_codes_processed": item_codes,
                "quantity_breakdown": item_counts,
                "total_line_items": len(item_codes),
            },
            "customer_record": {
                "billing_name": f"{customer_first_name} {customer_last_name}",
                "contact_email": customer_email,
                "delivery_destination": f"{shipping_street}, {shipping_city}, {shipping_state} {shipping_zip}",
            },
            "note": "This is a simulated order using hybrid semantic search. No actual purchase was made."
        }, indent=2)

    except Exception as e:
        return f"An unexpected error occurred during order processing: {str(e)}"


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        await mcp_product_catalog.run_sse_async()
    else:
        await mcp_product_catalog.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())