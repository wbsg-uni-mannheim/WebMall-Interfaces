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
    # Get shop configuration for webmall_3 (CamelCases)
    shop_id = "webmall_3"
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
            wc_url=os.getenv('WOO_STORE_URL_3'), 
            wc_consumer_key=os.getenv('WOO_CONSUMER_KEY_3'), 
            wc_consumer_secret=os.getenv('WOO_CONSUMER_SECRET_3')
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


mcp_store_inventory = FastMCP(
    "mcp-store-inventory-hybrid",
    description="AI agent server for CamelCases e-commerce store",
    lifespan=hybrid_search_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT_STORE_INVENTORY", "8062")
)


@mcp_store_inventory.tool()
async def query_stock(ctx: Context, product_name: str, results_per_page: int = 10, page_number: int = 1, result_order: str = "relevance", include_descriptions: bool = False) -> str:
    """Query stock levels for products in the store of CamelCases.

    This tool searches for products and returns their stock information.
    Can order results by different criteria.

    Args:
        ctx: The MCP server provided context which includes the search engine
        product_name: Name or part of the name of the product to query
        results_per_page: Number of results to return per page (default: 10, max: 100)
        page_number: Page number for pagination (default: 1)
        result_order: Order results by - "relevance", "price_low", "price_high", "name", "stock" (default: "relevance")

    Returns:
        JSON formatted list of products with stock details
    """
    try:
        search_engine = ctx.request_context.lifespan_context.search_engine

        # Use the enhanced search engine with server_c formatting
        return search_engine.search_server_c_format(product_name, results_per_page, page_number, result_order, include_descriptions)

    except Exception as e:
        return f"Error querying stock: {str(e)}"


@mcp_store_inventory.tool()
async def get_detailed_warehouse_products(ctx: Context, product_ids: list[str]) -> str:
    """Get detailed warehouse information about specific products by their IDs.

    This tool retrieves comprehensive product information for a list of product IDs,
    returning all available details including descriptions, stock levels, and metadata.

    Args:
        ctx: The MCP server provided context which includes the search engine
        product_ids: List of product IDs to retrieve detailed information for

    Returns:
        JSON formatted list of detailed product information in CamelCases format
    """
    try:
        search_engine = ctx.request_context.lifespan_context.search_engine
        
        # Get detailed products
        detailed_products = []
        for product_id in product_ids:
            try:
                product = search_engine.get_product_by_id(product_id)
                if product:
                    # Format in server_c style with full details
                    schema_org = product.get("schema_object", {})
                    
                    # Extract price information with fallbacks
                    price_info = schema_org.get("offers", {})
                    if isinstance(price_info, list) and price_info:
                        price_info = price_info[0]
                    
                    selling_price = price_info.get("price") or product.get("price", 0)
                    
                    detailed_product = {
                        "warehouse_sku_code": product_id,
                        "merchandise_designation": product.get("name", ""),
                        
                        "availability_metrics": {
                            "inventory_status": "instock",
                            "units_on_hand": schema_org.get("inventoryLevel", 0),
                            "stock_management": True,
                            "low_stock_threshold": 5,
                        },
                        
                        "commercial_data": {
                            "selling_price": str(selling_price) if selling_price else "",
                            "market_value": price_info.get("highPrice", ""),
                            "promotional_rate": str(selling_price) if selling_price else "",
                            "discount_active": bool(selling_price),
                        },
                        
                        "catalog_reference": {
                            "product_identifier": str(product_id),
                            "storefront_link": product.get("url", ""),
                            "type_classification": schema_org.get("@type", "Product"),
                        },
                        
                        "product_overview": {
                            "brief_description": product.get("description", ""),
                            "full_description": product.get("description", ""),
                        },
                        
                        "comprehensive_metadata": {
                            "schema_org": schema_org,
                            "site_info": product.get("site", ""),
                            "category": product.get("category", ""),
                            "last_indexed": "N/A"
                        }
                    }
                    detailed_products.append(detailed_product)
                else:
                    # Add error entry for missing product
                    detailed_products.append({
                        "warehouse_sku_code": product_id,
                        "error": f"Product {product_id} not found in warehouse",
                        "status": "not_found"
                    })
            except Exception as e:
                detailed_products.append({
                    "warehouse_sku_code": product_id,
                    "error": f"Error retrieving product {product_id}: {str(e)}",
                    "status": "error"
                })
        
        return json.dumps({
            "inventory_query": {
                "requested_sku_codes": product_ids,
                "products_located": len([p for p in detailed_products if "error" not in p]),
                "retrieval_errors": len([p for p in detailed_products if "error" in p])
            },
            "warehouse_catalog": detailed_products
        }, indent=2)
        
    except Exception as e:
        return f"Error retrieving detailed warehouse information: {str(e)}"


@mcp_store_inventory.tool()
async def get_product_info_by_id(ctx: Context, unique_product_id: int) -> str:
    """Retrieve comprehensive information for a product by its unique ID using WooCommerce API.

    This tool fetches all available details for a single product.

    Args:
        ctx: The MCP server provided context which includes the WooCommerce client
        unique_product_id: The unique identifier of the product (product identifier)

    Returns:
        JSON formatted detailed product information
    """
    try:
        wc_client = ctx.request_context.lifespan_context.wc_client

        # Get product by ID using WooCommerce API
        response = wc_client.get(f"products/{unique_product_id}")
        
        if response.status_code == 200:
            product = response.json()
        elif response.status_code == 404:
            return f"Product with ID {unique_product_id} not found"
        else:
            return f"Error retrieving product: HTTP {response.status_code} - {response.text}"

        # Transform to server_c detailed format
        # Extract price information
        price_info = {
            "price": product.get("price"),
            "highPrice": product.get("regular_price"),
            "lowPrice": product.get("sale_price")
        }

        comprehensive_product = {
            "retrieval_context": {
                "searched_sku_code": unique_product_id,
                "lookup_success": True,
                "data_source": "warehouse_catalog",
            },

            "entity_profile": {
                "product_identifier": product.get("id"),
                "brand_name": product.get("name", ""),
                "comprehensive_narrative": product.get("description", ""),
            },

            "commercial_metrics": {
                "current_market_price": price_info.get("price", ""),
                "standard_retail_price": price_info.get("highPrice", ""),
                "discounted_price": price_info.get("lowPrice", ""),
                "promotional_status": product.get("on_sale", False),
            },

            "inventory_intelligence": {
                "sku_reference_code": product.get("sku", ""),
                "warehouse_status": product.get("stock_status", "outofstock"),
                "available_quantity": product.get("stock_quantity", 0),
                "inventory_tracking": product.get("manage_stock", False),
                "reorder_point": product.get("low_stock_amount", 5),
            },

            "catalog_organization": {
                "category_memberships": [cat.get("name", "") for cat in product.get("categories", [])],
                "content_tags": [tag.get("name", "") for tag in product.get("tags", [])],
                "product_variant_type": product.get("type", "simple"),
            },

            "visual_media": [img.get("src", "") for img in product.get("images", [])],
            "customer_access_url": product.get("permalink", ""),

            "administrative_data": {
                "creation_date": product.get("date_created", ""),
                "last_modified": product.get("date_modified", ""),
                "publication_status": product.get("status", "publish"),
                "catalog_visibility": product.get("catalog_visibility", "visible"),
            }
        }

        return json.dumps(comprehensive_product, indent=2)

    except Exception as e:
        return f"Error retrieving product info: {str(e)}"


@mcp_store_inventory.tool()
async def fetch_product_types(ctx: Context, items_per_page: int = 50, page_num: int = 1, parent_id: int = None) -> str:
    """Fetch product types (categories) available in the store using WooCommerce API.

    This tool retrieves a list of product categories, including their hierarchical structure
    from the WooCommerce API.

    Args:
        ctx: The MCP server provided context which includes the WooCommerce client
        items_per_page: Number of product types to return per page (default: 50, max: 100)
        page_num: Page number for pagination (default: 1)
        parent_id: Parent category ID to filter sub-types (optional)

    Returns:
        JSON formatted list of product types
    """
    try:
        wc_client = ctx.request_context.lifespan_context.wc_client

        # Validate items_per_page parameter
        items_per_page = min(max(1, items_per_page), 100)

        # Build parameters for the API call
        params = {
            "per_page": items_per_page,
            "page": page_num,
            "hide_empty": False  # Include categories without products
        }

        # Add parent filter if specified
        if parent_id is not None:
            params["parent"] = parent_id

        # Get categories using WooCommerce API
        response = wc_client.get("products/categories", params=params)

        if response.status_code == 200:
            raw_categories = response.json()
            
            # Format as server_c style categories
            quirky_classifications = []
            for i, cat in enumerate(raw_categories):
                quirky_classifications.append({
                    "taxonomy_identifier": cat.get("id"),
                    "classification_label": cat.get("name", ""),
                    "url_token": cat.get("slug", ""),
                    "ancestry_link": cat.get("parent", 0),
                    "explanatory_content": cat.get("description", ""),
                    "inventory_tally": cat.get("count", 0),
                    "thumbnail_representation": cat.get("image", {}).get("src") if cat.get("image") else None,
                    "menu_sequence": cat.get("menu_order", i),
                })

            return json.dumps({
                "taxonomy_exploration": {
                    "current_page_number": page_num,
                    "results_per_page_limit": items_per_page,
                    "parent_type_filter_id": parent_id,
                    "browsing_depth": "root_level" if parent_id is None else "subcategory_level",
                },
                "classification_analytics": {
                    "total_product_types": len(quirky_classifications),
                    "has_nested_types": any(cat.get("ancestry_link") != 0 for cat in quirky_classifications),
                    "empty_categories_included": True,
                },
                "merchandise_taxonomy": quirky_classifications
            }, indent=2)
        else:
            return f"Error fetching product types: HTTP {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error fetching product types: {str(e)}"


@mcp_store_inventory.tool()
async def submit_order_for_payment(
    ctx: Context,
    product_identifiers: list[int],
    product_quantities: list[int],
    recipient_full_name: str,
    recipient_contact_email: str,
    delivery_address_line_1: str,
    delivery_city: str,
    delivery_state_or_province: str,
    delivery_postal_code: str,
    delivery_country_iso: str,
    recipient_phone_number: str,
    payment_card_num: str,
    payment_card_exp: str,
    payment_card_sec_code: str
) -> str:
    """Submits a complete order for payment and processing in the CamelCases store.

    This tool takes a list of products, customer details, and payment information to create a paid order.

    Args:
        ctx: The MCP server context
        product_identifiers: List of product IDs (SKU codes) to be ordered
        product_quantities: List of quantities for each product
        recipient_full_name: The full name of the customer
        recipient_contact_email: The customer's email address
        delivery_address_line_1: The primary line of the shipping address
        delivery_city: The city for shipping
        delivery_state_or_province: The state or province for shipping
        delivery_postal_code: The postal code for shipping
        delivery_country_iso: The two-letter ISO country code for shipping
        recipient_phone_number: The customer's phone number
        payment_card_num: The credit card number for payment
        payment_card_exp: The credit card's expiration date (MM/YY)
        payment_card_sec_code: The card's security code (CVC)

    Returns:
        A JSON string detailing the result of the order submission
    """
    if len(product_identifiers) != len(product_quantities):
        return json.dumps({"transaction_result": "failure", "error_explanation": "Mismatch between product identifiers and quantities."})

    try:
        if not all([payment_card_num, payment_card_exp, payment_card_sec_code]):
            return json.dumps({"transaction_result": "failure", "error_explanation": "Incomplete payment card information provided."})

        # Simple split of full name into first and last
        name_parts = recipient_full_name.split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ''

        # Simulate order creation
        order_id = str(uuid.uuid4())[:8]
        transaction_id = str(uuid.uuid4())
        
        # Calculate total (simulation)
        total_amount = sum(qty * 20.00 for qty in product_quantities) 

        return json.dumps({
            "transaction_result": "success",
            "completion_ceremony": {
                "victory_message": "Order processed successfully with payment confirmed! (SIMULATION)",
                "celebration_status": "order_fulfilled",
            },
            "order_documentation": {
                "order_number": order_id,
                "confirmation_code": transaction_id,
                "total_charged": f"{total_amount:.2f}",
                "order_status_updated": "processing",
            },
            "purchase_breakdown": {
                "sku_codes_ordered": product_identifiers,
                "quantities_per_item": product_quantities,
                "line_items_total": len(product_identifiers),
            },
            "customer_profile": {
                "recipient_identity": recipient_full_name,
                "communication_email": recipient_contact_email,
                "shipping_destination": f"{delivery_address_line_1}, {delivery_city}, {delivery_state_or_province} {delivery_postal_code}",
            },
            "note": "This is a simulated order using hybrid semantic search. No actual purchase was made."
        }, indent=2)

    except Exception as e:
        return json.dumps({"transaction_result": "failure", "error_explanation": "An unexpected server error occurred.", "error_message": str(e)})


@mcp_store_inventory.tool()
async def add_item_to_warehouse_cart(ctx: Context, product_id: str, quantity: int = 1) -> str:
    """Add an item to the warehouse cart in CamelCases.

    This tool adds products to the warehouse cart. The cart persists during the session.

    Args:
        ctx: The MCP server provided context
        product_id: The product ID to add to cart
        quantity: The quantity to add (default: 1)

    Returns:
        JSON formatted response with the current cart contents in CamelCases "eccentric" format
    """
    try:
        shop_id = ctx.request_context.lifespan_context.shop_id
        search_engine = ctx.request_context.lifespan_context.search_engine
        shop_config = WEBMALL_SHOPS.get(shop_id, {})
        shop_url = shop_config.get("url", "")
        
        # Use global cart store
        cart = _global_cart_store.get_cart(shop_id)
        
        logger.debug(f"add_item_to_warehouse_cart_{shop_id}: Current cart has {len(cart)} items before adding")
        
        # Normalize product_id
        normalized_product_id = str(product_id).strip()
        logger.debug(f"add_item_to_warehouse_cart_{shop_id}: Normalized product_id from {repr(product_id)} to {repr(normalized_product_id)}")
        
        # Get product details
        try:
            product = await asyncio.wait_for(
                asyncio.to_thread(search_engine.get_product_by_id, normalized_product_id),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Get product timeout for ID: {normalized_product_id} in add_item_to_warehouse_cart")
            return json.dumps({
                "transaction_result": "failure",
                "error_explanation": "Product lookup timed out",
                "product_reference_id": normalized_product_id,
                "timeout_duration": "30s"
            })
        
        if not product:
            return json.dumps({
                "transaction_result": "failure",
                "error_explanation": f"Product {normalized_product_id} not found in warehouse catalog",
                "warehouse_catalog": {"catalog_reference": {"storefront_link": ""}, "items": []}
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
        
        logger.debug(f"add_item_to_warehouse_cart_{shop_id}: Cart now has {len(cart)} items after adding")
        
        # Format response in "eccentric" server C format
        warehouse_items = []
        for item in cart.values():
            warehouse_items.append({
                "inventory_item_code": item["product_id"],
                "merchandise_name": item["name"],
                "unit_cost_value": item["price"],
                "quantity_in_cart": item["quantity"],
                "catalog_reference": {
                    "storefront_link": item["url"]
                },
                "subtotal_amount": item["price"] * item["quantity"]
            })
        
        return json.dumps({
            "transaction_result": "success",
            "action_summary": f"Added {quantity} x {product_name} to warehouse cart",
            "warehouse_catalog": {
                "catalog_reference": {
                    "storefront_link": f"{shop_url}/cart"
                },
                "items": warehouse_items,
                "total_item_count": sum(item["quantity"] for item in cart.values()),
                "total_cart_value": sum(item["price"] * item["quantity"] for item in cart.values())
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Add item to warehouse cart failed for product '{normalized_product_id if 'normalized_product_id' in locals() else product_id}': {e}")
        return json.dumps({
            "transaction_result": "failure",
            "error_explanation": f"Failed to add item to warehouse cart: {str(e)}",
            "product_reference_id": normalized_product_id if 'normalized_product_id' in locals() else product_id
        })


@mcp_store_inventory.tool()
async def view_warehouse_cart(ctx: Context) -> str:
    """View the current warehouse cart contents in CamelCases.

    This tool returns the current contents of the warehouse cart in CamelCases "eccentric" format.

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
        
        logger.debug(f"view_warehouse_cart_{shop_id}: Cart has {len(cart)} items")
        
        # Format cart in "eccentric" server C format
        warehouse_items = []
        for item in cart.values():
            warehouse_items.append({
                "inventory_item_code": item["product_id"],
                "merchandise_name": item["name"],
                "unit_cost_value": item["price"],
                "quantity_in_cart": item["quantity"],
                "catalog_reference": {
                    "storefront_link": item["url"]
                },
                "subtotal_amount": item["price"] * item["quantity"]
            })
        
        return json.dumps({
            "warehouse_catalog": {
                "catalog_reference": {
                    "storefront_link": f"{shop_url}/cart"
                },
                "items": warehouse_items,
                "total_item_count": sum(item["quantity"] for item in cart.values()),
                "total_cart_value": sum(item["price"] * item["quantity"] for item in cart.values()),
                "cart_status": "active" if cart else "empty"
            },
            "operation_context": {
                "checkout_enabled": len(cart) > 0,
                "modification_permitted": True
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"View warehouse cart failed: {e}")
        return json.dumps({
            "transaction_result": "failure",
            "error_explanation": f"Failed to view warehouse cart: {str(e)}"
        })


@mcp_store_inventory.tool()
async def empty_warehouse_cart(ctx: Context) -> str:
    """Empty the warehouse cart in CamelCases.

    This tool empties the warehouse cart.

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
        
        logger.debug(f"empty_warehouse_cart_{shop_id}: Cart has been cleared")
        
        return json.dumps({
            "transaction_result": "success",
            "action_summary": "Warehouse cart has been emptied",
            "warehouse_catalog": {
                "catalog_reference": {
                    "storefront_link": f"{shop_url}/cart"
                },
                "items": [],
                "total_item_count": 0,
                "total_cart_value": 0.0,
                "cart_status": "empty"
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Empty warehouse cart failed: {e}")
        return json.dumps({
            "transaction_result": "failure",
            "error_explanation": f"Failed to empty warehouse cart: {str(e)}"
        })


@mcp_store_inventory.tool()
async def process_warehouse_cart_checkout(
    ctx: Context,
    customer_first_name: str,
    customer_last_name: str,
    customer_email: str,
    customer_phone: str,
    delivery_address_line_1: str,
    delivery_city: str,
    delivery_state_or_province: str,
    delivery_postal_code: str,
    delivery_country_code: str,
    payment_card_number: str,
    card_expiration_date: str,
    card_security_code: str
) -> str:
    """Complete checkout and create an order for items in the warehouse cart in CamelCases.

    This tool creates an order using the items currently in the warehouse cart.

    Args:
        ctx: The MCP server provided context
        customer_first_name: Customer's first name
        customer_last_name: Customer's last name
        customer_email: Customer's email address
        customer_phone: Customer's phone number
        delivery_address_line_1: Customer's street address
        delivery_city: Customer's city
        delivery_state_or_province: Customer's state/province
        delivery_postal_code: Customer's postal/zip code
        delivery_country_code: Customer's country code
        payment_card_number: The credit card number
        card_expiration_date: The credit card expiry date (MM/YY)
        card_security_code: The credit card CVC code

    Returns:
        JSON formatted response with order confirmation and product URLs in "eccentric" format
    """
    try:
        shop_id = ctx.request_context.lifespan_context.shop_id
        shop_config = WEBMALL_SHOPS.get(shop_id, {})
        shop_url = shop_config.get("url", "")
        
        # Use global cart store
        cart = _global_cart_store.get_cart(shop_id)
        
        logger.debug(f"process_warehouse_cart_checkout_{shop_id}: Cart has {len(cart)} items at checkout")
        
        if not cart:
            return json.dumps({
                "transaction_result": "failure",
                "error_explanation": "Warehouse cart is empty. Please add items to cart before checkout."
            })
        
        # Validate credit card (basic check)
        if not all([payment_card_number, card_expiration_date, card_security_code]):
            return json.dumps({
                "transaction_result": "failure",
                "error_explanation": "Credit card details are incomplete."
            })
        
        # Create order ID
        order_id = f"{shop_id}_order_{str(uuid.uuid4())[:8]}"
        
        # Calculate total
        total_amount = sum(item["price"] * item["quantity"] for item in cart.values())
        
        # Get product URLs from cart
        storefront_links = []
        for item in cart.values():
            storefront_links.append({
                "storefront_link": item["url"]
            })
        
        # Format order items in "eccentric" server C format
        order_items = []
        for item in cart.values():
            order_items.append({
                "inventory_item_code": item["product_id"],
                "merchandise_name": item["name"],
                "unit_cost_value": item["price"],
                "quantity_ordered": item["quantity"],
                "catalog_reference": {
                    "storefront_link": item["url"]
                },
                "line_item_total": item["price"] * item["quantity"]
            })
        
        # Create order response in "eccentric" format
        order_response = {
            "transaction_result": "success",
            "completion_message": "Order has been successfully processed and confirmed",
            "order_documentation": {
                "order_reference_id": order_id,
                "processing_status": "confirmed",
                "total_order_amount": f"{total_amount:.2f}",
                "payment_transaction_id": str(uuid.uuid4()),
                "customer_information": {
                    "full_customer_name": f"{customer_first_name} {customer_last_name}",
                    "communication_email": customer_email,
                    "contact_phone": customer_phone,
                    "shipping_destination": f"{delivery_address_line_1}, {delivery_city}, {delivery_state_or_province} {delivery_postal_code}, {delivery_country_code}"
                }
            },
            "warehouse_catalog": {
                "items": order_items,
                "catalog_references": storefront_links,
                "total_item_count": len(cart),
                "total_order_value": total_amount
            }
        }
        
        # Clear cart after successful checkout
        _global_cart_store.clear_cart(shop_id)
        logger.debug(f"process_warehouse_cart_checkout_{shop_id}: Cleared cart after successful checkout")
        
        return json.dumps(order_response, indent=2)
        
    except Exception as e:
        logger.error(f"Process warehouse cart checkout failed: {e}")
        return json.dumps({
            "transaction_result": "failure",
            "error_explanation": f"Checkout failed: {str(e)}"
        })


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        await mcp_store_inventory.run_sse_async()
    else:
        await mcp_store_inventory.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())