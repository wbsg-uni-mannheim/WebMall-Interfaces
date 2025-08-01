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
    # Get shop configuration for webmall_4 (Hardware Cafe)
    shop_id = "webmall_4"
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
            wc_url=os.getenv('WOO_STORE_URL_4'), 
            wc_consumer_key=os.getenv('WOO_CONSUMER_KEY_4'), 
            wc_consumer_secret=os.getenv('WOO_CONSUMER_SECRET_4')
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


mcp_ecommerce_data = FastMCP(
    "mcp-ecommerce-data-hybrid",
    description="Hardware Cafe e-commerce store",
    lifespan=hybrid_search_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT_ECOMMERCE_DATA", "8063")
)


@mcp_ecommerce_data.tool()
async def get_items_by_keyword(ctx: Context, keyword: str, results_limit: int = 10, page_number: int = 1, min_price: float = None, max_price: float = None, include_descriptions: bool = False) -> str:
    """Retrieve e-commerce items based on a keyword search of Hardware Cafe.

    This tool searches for items using a keyword and provides basic item information.
    Can optionally filter by price range.

    Args:
        ctx: The MCP server provided context which includes the search engine
        keyword: Keyword to search for items
        results_limit: Maximum number of results to return per page (default: 10, max: 100)
        page_number: Page number for pagination (default: 1)
        min_price: Minimum price filter (optional, use "cheap" in keyword for budget items)
        max_price: Maximum price filter (optional)

    Returns:
        JSON formatted list of items matching the keyword
    """
    try:
        search_engine = ctx.request_context.lifespan_context.search_engine

        # Use the enhanced search engine with server_d formatting
        return search_engine.search_server_d_format(keyword, results_limit, page_number, min_price, max_price, include_descriptions)

    except Exception as e:
        return f"Error retrieving items by keyword: {str(e)}"


@mcp_ecommerce_data.tool()
async def find_cheap_items(ctx: Context, keyword: str = "", max_budget: float = 50.0, results_limit: int = 15, include_descriptions: bool = False) -> str:
    """Find budget-friendly items in the Hardware Cafe.

    This is an alternative tool for budget shopping, separate from price range parameters.
    Focuses specifically on affordable merchandise using semantic search.

    Args:
        ctx: The MCP server provided context which includes the search engine
        keyword: Optional keyword to search within budget items (default: "")
        max_budget: Maximum price threshold for budget items (default: 50.0)
        results_limit: Maximum number of budget items to return (default: 15, max: 50)

    Returns:
        JSON formatted list of affordable items
    """
    try:
        search_engine = ctx.request_context.lifespan_context.search_engine

        # Validate results_limit parameter
        results_limit = min(max(1, results_limit), 50)

        # Use semantic search with budget filtering
        search_query = f"cheap {keyword}" if keyword else "cheap budget affordable"
        search_results = search_engine.search(search_query, top_k=results_limit)
        
        # Filter results by budget and transform to server_d budget format
        thrifty_discoveries = []
        for result in search_results.get("results", []):
            schema_org = result.get("schema_object", {})
            
            # Extract price information
            price_info = schema_org.get("offers", {})
            if isinstance(price_info, list) and price_info:
                price_info = price_info[0]
            
            price = float(price_info.get("price", 0) or 0)
            
            # Only include items within budget
            if price <= max_budget and price > 0:
                regular_price = float(price_info.get("highPrice", price) or price)
                savings = regular_price - price if regular_price > price else 0
                discount_percentage = round((savings / regular_price) * 100, 1) if regular_price > 0 else 0
                
                thrifty_discoveries.append({
                    "bargain_sku_code": result.get("url", "").split("/")[-1] if result.get("url") else "",
                    "affordable_title": result.get("name", ""),

                    "budget_metrics": {
                        "wallet_friendly_price": f"{price:.2f}",
                        "savings_vs_regular": savings,
                        "discount_percentage": discount_percentage,
                    },

                    "product_essence": {
                        "quick_description": schema_org.get("description", "").strip()[:100] if include_descriptions else "",
                        "value_proposition": f"Great value at {price:.2f} EUR",
                    },

                    "classification_labels": [schema_org.get("category", "")],
                    "availability_info": "instock" if schema_org.get("availability") == "InStock" else "outofstock",
                    "marketplace_link": result.get("url", ""),
                })

        # Calculate summary metrics
        cheapest_item = min((float(item["budget_metrics"]["wallet_friendly_price"]) for item in thrifty_discoveries), default=0)
        avg_price = sum(float(item["budget_metrics"]["wallet_friendly_price"]) for item in thrifty_discoveries) / len(thrifty_discoveries) if thrifty_discoveries else 0
        total_savings = sum(item["budget_metrics"]["savings_vs_regular"] for item in thrifty_discoveries)

        return json.dumps({
            "budget_shopping_session": {
                "search_focus": "affordable_items",
                "keyword_filter": keyword if keyword else "all_categories",
                "price_ceiling": max_budget,
                "bargain_hunt_results": len(thrifty_discoveries),
            },
            "savings_summary": {
                "cheapest_item": cheapest_item,
                "average_budget_price": avg_price,
                "total_potential_savings": total_savings,
            },
            "thrifty_catalog": thrifty_discoveries
        }, indent=2)

    except Exception as e:
        return f"Error finding cheap items: {str(e)}"


@mcp_ecommerce_data.tool()
async def get_detailed_marketplace_items(ctx: Context, product_ids: list[str]) -> str:
    """Get detailed marketplace information about specific items by their IDs.

    This tool retrieves comprehensive item information for a list of product IDs,
    returning all available details including descriptions, specifications, and metadata.

    Args:
        ctx: The MCP server provided context which includes the search engine
        product_ids: List of product IDs to retrieve detailed information for

    Returns:
        JSON formatted list of detailed item information in Hardware Cafe format
    """
    try:
        search_engine = ctx.request_context.lifespan_context.search_engine
        
        # Get detailed products
        detailed_items = []
        for product_id in product_ids:
            try:
                product = search_engine.get_product_by_id(product_id)
                if product:
                    # Format in server_d style with full details
                    schema_org = product.get("schema_object", {})
                    
                    # Extract price information with fallbacks
                    price_info = schema_org.get("offers", {})
                    if isinstance(price_info, list) and price_info:
                        price_info = price_info[0]
                    
                    purchase_cost = price_info.get("price") or product.get("price", 0)
                    
                    detailed_item = {
                        "catalog_sku_code": product_id,
                        "item_designation": product.get("name", ""),
                        
                        "economic_data": {
                            "purchase_cost": str(purchase_cost) if purchase_cost else "",
                            "baseline_price": price_info.get("highPrice", ""),
                            "markdown_price": str(purchase_cost) if purchase_cost else "",
                            "promotion_flag": bool(purchase_cost),
                        },
                        
                        "descriptive_content": {
                            "extended_details": product.get("description", ""),
                        },
                        
                        "organizational_tags": [product.get("category", "")] if product.get("category") else [],
                        "storefront_reference": product.get("url", ""),
                        
                        "inventory_status": {
                            "product_identifier": str(product_id),
                            "availability": "InStock",
                            "units_available": schema_org.get("inventoryLevel", 0),
                        },
                        
                        "comprehensive_metadata": {
                            "schema_org": schema_org,
                            "site_info": product.get("site", ""),
                            "category": product.get("category", ""),
                            "last_indexed": "N/A"
                        }
                    }
                    detailed_items.append(detailed_item)
                else:
                    # Add error entry for missing product
                    detailed_items.append({
                        "catalog_sku_code": product_id,
                        "error": f"Item {product_id} not found in marketplace",
                        "status": "not_found"
                    })
            except Exception as e:
                detailed_items.append({
                    "catalog_sku_code": product_id,
                    "error": f"Error retrieving item {product_id}: {str(e)}",
                    "status": "error"
                })
        
        return json.dumps({
            "search_expedition": {
                "requested_sku_codes": product_ids,
                "items_discovered": len([i for i in detailed_items if "error" not in i]),
                "retrieval_failures": len([i for i in detailed_items if "error" in i])
            },
            "marketplace_inventory": detailed_items
        }, indent=2)
        
    except Exception as e:
        return f"Error retrieving detailed marketplace information: {str(e)}"


@mcp_ecommerce_data.tool()
async def get_item_details_by_id(ctx: Context, item_unique_id: int) -> str:
    """Retrieve comprehensive details for a specific e-commerce item by its unique ID using WooCommerce API.

    This tool provides all available information for a single item.

    Args:
        ctx: The MCP server provided context which includes the WooCommerce client
        item_unique_id: The unique ID of the item to retrieve details for (product identifier)

    Returns:
        JSON formatted detailed item information
    """
    try:
        wc_client = ctx.request_context.lifespan_context.wc_client

        # Get product by ID using WooCommerce API
        response = wc_client.get(f"products/{item_unique_id}")
        
        if response.status_code == 200:
            product = response.json()
        elif response.status_code == 404:
            return f"Item with ID {item_unique_id} not found"
        else:
            return f"Error retrieving item: HTTP {response.status_code} - {response.text}"

        # Transform to server_d detailed format
        # Extract price information
        price_info = {
            "price": product.get("price"),
            "highPrice": product.get("regular_price"),
            "lowPrice": product.get("sale_price")
        }

        elaborate_item_profile = {
            "discovery_journal": {
                "requested_sku_code": item_unique_id,
                "search_outcome": "item_located",
                "catalog_source": "hardware_cafe_database",
            },

            "item_identity": {
                "unique_id": product.get("id"),
                "full_name": product.get("name", ""),
                "product_identifier": product.get("sku", ""),
                "item_type": product.get("type", "simple"),
            },

            "commercial_intelligence": {
                "current_price": price_info.get("price", ""),
                "standard_price": price_info.get("highPrice", ""),
                "promotional_price": price_info.get("lowPrice", ""),
                "deal_status": product.get("on_sale", False),
                "pricing_history": {
                    "last_updated": product.get("date_modified", ""),
                    "creation_date": product.get("date_created", ""),
                },
            },

            "content_library": {
                "detailed_description": product.get("description", ""),
                "elevator_pitch": product.get("short_description", ""),
                "technical_specs": [attr.get("name", "") + ": " + ", ".join([opt.get("name", "") for opt in attr.get("options", [])]) for attr in product.get("attributes", [])],
            },

            "inventory_intelligence": {
                "stock_status_info": product.get("stock_status", "outofstock"),
                "quantity_in_stock": product.get("stock_quantity", 0),
                "stock_tracking": product.get("manage_stock", False),
                "low_stock_alert": product.get("low_stock_amount", 5),
            },

            "catalog_taxonomy": {
                "associated_categories": [cat.get("name", "") for cat in product.get("categories", [])],
                "keyword_tags": [tag.get("name", "") for tag in product.get("tags", [])],
            },

            "visual_showcase": {
                "main_image_url": product.get("images", [{}])[0].get("src", "") if product.get("images") else "",
                "gallery_images": [img.get("src", "") for img in product.get("images", [])],
            },

            "customer_access": {
                "item_url": product.get("permalink", ""),
                "external_link": product.get("external_url", ""),
            },

            "business_metadata": {
                "featured_status": product.get("featured", False),
                "catalog_visibility": product.get("catalog_visibility", "visible"),
                "purchase_note": product.get("purchase_note", ""),
            }
        }

        return json.dumps(elaborate_item_profile, indent=2)

    except Exception as e:
        return f"Error retrieving item details: {str(e)}"


@mcp_ecommerce_data.tool()
async def get_all_categories(ctx: Context, limit_per_page: int = 50, page_num: int = 1, parent_category_filter: int = None) -> str:
    """Retrieve all available e-commerce categories using WooCommerce API.

    This tool fetches a list of product categories, including their hierarchical relationships
    from the WooCommerce API.

    Args:
        ctx: The MCP server provided context which includes the WooCommerce client
        limit_per_page: Number of categories to return per page (default: 50, max: 100)
        page_num: Page number for pagination (default: 1)
        parent_category_filter: Parent category ID to filter subcategories (optional)

    Returns:
        JSON formatted list of e-commerce categories
    """
    try:
        wc_client = ctx.request_context.lifespan_context.wc_client

        # Validate limit_per_page parameter
        limit_per_page = min(max(1, limit_per_page), 100)

        # Build parameters for the API call
        params = {
            "per_page": limit_per_page,
            "page": page_num,
            "hide_empty": False  # Include categories without products
        }

        # Add parent filter if specified
        if parent_category_filter is not None:
            params["parent"] = parent_category_filter

        # Get categories using WooCommerce API
        response = wc_client.get("products/categories", params=params)

        if response.status_code == 200:
            raw_categories = response.json()
            
            # Format as server_d style categories
            imaginative_categories = []
            for i, cat in enumerate(raw_categories):
                imaginative_categories.append({
                    "classification_identifier": cat.get("id"),
                    "category_title": cat.get("name", ""),
                    "url_path": cat.get("slug", ""),
                    "parent_category_id": cat.get("parent", 0),
                    "merchandise_count": cat.get("count", 0),
                    "descriptive_text": cat.get("description", ""),
                    "visual_thumbnail": cat.get("image", {}).get("src") if cat.get("image") else None,
                    "menu_position": cat.get("menu_order", i),
                })

            categories_with_products = len([cat for cat in imaginative_categories if cat["merchandise_count"] > 0])
            empty_categories = len(imaginative_categories) - categories_with_products

            return json.dumps({
                "category_exploration": {
                    "current_page_number": page_num,
                    "results_limit": limit_per_page,
                    "parent_filter_applied": parent_category_filter,
                    "navigation_level": "top_level" if parent_category_filter is None else "subcategory_level",
                },
                "catalog_analytics": {
                    "total_categories_found": len(imaginative_categories),
                    "categories_with_products": categories_with_products,
                    "empty_categories": empty_categories,
                },
                "category_directory": imaginative_categories
            }, indent=2)
        else:
            return f"Error retrieving categories: HTTP {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error retrieving categories: {str(e)}"


@mcp_ecommerce_data.tool()
async def finalize_purchase(
    ctx: Context,
    sku_list: list[int],
    quantity_list: list[int],
    purchaser_name: str,
    purchaser_email: str,
    delivery_address: str,
    delivery_locality: str,
    delivery_region: str,
    delivery_postal_code: str,
    delivery_country_code: str,
    purchaser_phone: str,
    charge_card_number: str,
    charge_card_expiry: str,
    charge_card_cvc: str
) -> str:
    """Finalizes a purchase and processes a payment for the Hardware Cafe.

    This tool takes a shopping cart, customer data, and payment details to create a fully paid order
    using semantic search. Since this is a hybrid server, no actual order is created.

    Args:
        ctx: The MCP server context
        sku_list: A list of product SKUs (product identifiers) to be purchased
        quantity_list: A list of quantities corresponding to each SKU
        purchaser_name: The full name of the person making the purchase
        purchaser_email: The contact email of the purchaser
        delivery_address: The street address for delivery
        delivery_locality: The city for delivery
        delivery_region: The state/region for delivery
        delivery_postal_code: The postal code for delivery
        delivery_country_code: The ISO country code for delivery
        purchaser_phone: The contact phone number of the purchaser
        charge_card_number: The credit card number to charge
        charge_card_expiry: The expiration date of the credit card (MM/YY)
        charge_card_cvc: The CVC/CVV security code of the credit card

    Returns:
        A JSON string containing the transaction receipt or an error message
    """
    if len(sku_list) != len(quantity_list):
        return json.dumps({"purchase_outcome": "failed", "failure_reason": {"message": "SKU list and quantity list must be the same length."}})

    try:
        if not all([charge_card_number, charge_card_expiry, charge_card_cvc]):
            return json.dumps({"purchase_outcome": "failed", "failure_reason": {"message": "Invalid credit card details provided."}})

        name_parts = purchaser_name.split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ''

        # Simulate order creation
        order_id = str(uuid.uuid4())[:8]
        transaction_id = str(uuid.uuid4())
        
        # Calculate total (simulation)
        total_amount = sum(qty * 25.00 for qty in quantity_list)  

        return json.dumps({
            "purchase_outcome": "completed",
            "celebration_details": {
                "success_message": "Transaction completed successfully! Your order is confirmed. (SIMULATION)",
                "completion_timestamp": "",
            },
            "transaction_documentation": {
                "order_id": order_id,
                "transaction_ref": transaction_id,
                "charged_amount": f"{total_amount:.2f}",
                "order_status_code": "processing",
            },
            "purchase_manifest": {
                "sku_codes_ordered": sku_list,
                "item_quantities": quantity_list,
                "total_line_items": len(sku_list),
                "order_complexity": "simple" if len(sku_list) <= 3 else "complex",
            },
            "customer_details": {
                "purchaser_identity": purchaser_name,
                "contact_email": purchaser_email,
                "shipping_address": f"{delivery_address}, {delivery_locality}, {delivery_region} {delivery_postal_code}",
                "delivery_country": delivery_country_code,
            },
            "note": "This is a simulated order using hybrid semantic search. No actual purchase was made."
        }, indent=2)

    except Exception as e:
        return json.dumps({"purchase_outcome": "failed", "failure_reason": {"message": str(e), "error_type": "system_exception"}})


@mcp_ecommerce_data.tool()
async def add_item_to_marketplace_cart(ctx: Context, product_id: str, quantity: int = 1) -> str:
    """Add an item to the marketplace cart in Hardware Cafe.

    This tool adds products to the marketplace cart. The cart persists during the session.

    Args:
        ctx: The MCP server provided context
        product_id: The product ID to add to cart
        quantity: The quantity to add (default: 1)

    Returns:
        JSON formatted response with the current cart contents in Hardware Cafe "unconventional" format
    """
    try:
        shop_id = ctx.request_context.lifespan_context.shop_id
        search_engine = ctx.request_context.lifespan_context.search_engine
        shop_config = WEBMALL_SHOPS.get(shop_id, {})
        shop_url = shop_config.get("url", "")
        
        # Use global cart store
        cart = _global_cart_store.get_cart(shop_id)
        
        logger.debug(f"add_item_to_marketplace_cart_{shop_id}: Current cart has {len(cart)} items before adding")
        
        # Normalize product_id
        normalized_product_id = str(product_id).strip()
        logger.debug(f"add_item_to_marketplace_cart_{shop_id}: Normalized product_id from {repr(product_id)} to {repr(normalized_product_id)}")
        
        # Get product details
        try:
            product = await asyncio.wait_for(
                asyncio.to_thread(search_engine.get_product_by_id, normalized_product_id),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Get product timeout for ID: {normalized_product_id} in add_item_to_marketplace_cart")
            return json.dumps({
                "purchase_outcome": "failed",
                "failure_reason": {
                    "message": "Product lookup timed out",
                    "error_type": "timeout_exception"
                },
                "product_reference": normalized_product_id,
                "timeout_duration": "30s"
            })
        
        if not product:
            return json.dumps({
                "purchase_outcome": "failed",
                "failure_reason": {
                    "message": f"Product {normalized_product_id} not found in marketplace",
                    "error_type": "product_not_found"
                },
                "marketplace_inventory": {"items": [], "storefront_reference": ""}
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
        
        logger.debug(f"add_item_to_marketplace_cart_{shop_id}: Cart now has {len(cart)} items after adding")
        
        # Format response in "unconventional" server D format
        marketplace_items = []
        for item in cart.values():
            marketplace_items.append({
                "product_sku": item["product_id"],
                "item_name": item["name"],
                "price_point": item["price"],
                "cart_quantity": item["quantity"],
                "storefront_reference": item["url"],
                "line_value": item["price"] * item["quantity"]
            })
        
        return json.dumps({
            "purchase_outcome": "successful",
            "operation_summary": f"Added {quantity} x {product_name} to marketplace cart",
            "marketplace_inventory": {
                "items": marketplace_items,
                "total_items_count": sum(item["quantity"] for item in cart.values()),
                "cart_total_value": sum(item["price"] * item["quantity"] for item in cart.values()),
                "storefront_reference": f"{shop_url}/cart"
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Add item to marketplace cart failed for product '{normalized_product_id if 'normalized_product_id' in locals() else product_id}': {e}")
        return json.dumps({
            "purchase_outcome": "failed",
            "failure_reason": {
                "message": f"Failed to add item to marketplace cart: {str(e)}",
                "error_type": "system_exception"
            },
            "product_reference": normalized_product_id if 'normalized_product_id' in locals() else product_id
        })


@mcp_ecommerce_data.tool()
async def view_marketplace_cart(ctx: Context) -> str:
    """View the current marketplace cart contents in Hardware Cafe.

    This tool returns the current contents of the marketplace cart in Hardware Cafe "unconventional" format.

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
        
        logger.debug(f"view_marketplace_cart_{shop_id}: Cart has {len(cart)} items")
        
        # Format cart in "unconventional" server D format
        marketplace_items = []
        for item in cart.values():
            marketplace_items.append({
                "product_sku": item["product_id"],
                "item_name": item["name"],
                "price_point": item["price"],
                "cart_quantity": item["quantity"],
                "storefront_reference": item["url"],
                "line_value": item["price"] * item["quantity"]
            })
        
        return json.dumps({
            "marketplace_inventory": {
                "items": marketplace_items,
                "total_items_count": sum(item["quantity"] for item in cart.values()),
                "cart_total_value": sum(item["price"] * item["quantity"] for item in cart.values()),
                "storefront_reference": f"{shop_url}/cart",
                "cart_status": "active" if cart else "empty"
            },
            "operation_context": {
                "checkout_ready": len(cart) > 0,
                "modifications_allowed": True
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"View marketplace cart failed: {e}")
        return json.dumps({
            "purchase_outcome": "failed",
            "failure_reason": {
                "message": f"Failed to view marketplace cart: {str(e)}",
                "error_type": "system_exception"
            }
        })


@mcp_ecommerce_data.tool()
async def clear_marketplace_cart(ctx: Context) -> str:
    """Clear the marketplace cart in Hardware Cafe.

    This tool empties the marketplace cart.

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
        
        logger.debug(f"clear_marketplace_cart_{shop_id}: Cart has been cleared")
        
        return json.dumps({
            "purchase_outcome": "successful",
            "operation_summary": "Marketplace cart has been cleared",
            "marketplace_inventory": {
                "items": [],
                "total_items_count": 0,
                "cart_total_value": 0.0,
                "storefront_reference": f"{shop_url}/cart",
                "cart_status": "empty"
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Clear marketplace cart failed: {e}")
        return json.dumps({
            "purchase_outcome": "failed",
            "failure_reason": {
                "message": f"Failed to clear marketplace cart: {str(e)}",
                "error_type": "system_exception"
            }
        })


@mcp_ecommerce_data.tool()
async def checkout_marketplace_cart(
    ctx: Context,
    customer_first_name: str,
    customer_last_name: str,
    customer_email: str,
    customer_phone: str,
    delivery_street_address: str,
    delivery_city: str,
    delivery_state: str,
    delivery_zip_code: str,
    delivery_country: str,
    payment_card_number: str,
    card_expiry_date: str,
    card_security_code: str
) -> str:
    """Complete checkout and create an order for items in the marketplace cart in Hardware Cafe.

    This tool creates an order using the items currently in the marketplace cart.

    Args:
        ctx: The MCP server provided context
        customer_first_name: Customer's first name
        customer_last_name: Customer's last name
        customer_email: Customer's email address
        customer_phone: Customer's phone number
        delivery_street_address: Customer's street address
        delivery_city: Customer's city
        delivery_state: Customer's state
        delivery_zip_code: Customer's zip code
        delivery_country: Customer's country
        payment_card_number: The credit card number
        card_expiry_date: The credit card expiry date (MM/YY)
        card_security_code: The credit card CVC code

    Returns:
        JSON formatted response with order confirmation and product URLs in "unconventional" format
    """
    try:
        shop_id = ctx.request_context.lifespan_context.shop_id
        shop_config = WEBMALL_SHOPS.get(shop_id, {})
        shop_url = shop_config.get("url", "")
        
        # Use global cart store
        cart = _global_cart_store.get_cart(shop_id)
        
        logger.debug(f"checkout_marketplace_cart_{shop_id}: Cart has {len(cart)} items at checkout")
        
        if not cart:
            return json.dumps({
                "purchase_outcome": "failed",
                "failure_reason": {
                    "message": "Marketplace cart is empty. Please add items to cart before checkout.",
                    "error_type": "empty_cart"
                }
            })
        
        # Validate credit card (basic check)
        if not all([payment_card_number, card_expiry_date, card_security_code]):
            return json.dumps({
                "purchase_outcome": "failed",
                "failure_reason": {
                    "message": "Credit card details are incomplete.",
                    "error_type": "invalid_payment"
                }
            })
        
        # Create order ID
        order_id = f"{shop_id}_order_{str(uuid.uuid4())[:8]}"
        
        # Calculate total
        total_amount = sum(item["price"] * item["quantity"] for item in cart.values())
        
        # Get product URLs from cart
        storefront_references = [item["url"] for item in cart.values()]
        
        # Format order items in "unconventional" server D format
        order_items = []
        for item in cart.values():
            order_items.append({
                "product_sku": item["product_id"],
                "item_name": item["name"],
                "price_point": item["price"],
                "ordered_quantity": item["quantity"],
                "storefront_reference": item["url"],
                "line_total": item["price"] * item["quantity"]
            })
        
        # Create order response in "unconventional" format
        order_response = {
            "purchase_outcome": "successful",
            "completion_notice": "Order has been successfully processed and confirmed",
            "order_details": {
                "order_tracking_id": order_id,
                "fulfillment_status": "confirmed",
                "total_purchase_amount": f"{total_amount:.2f}",
                "payment_confirmation_id": str(uuid.uuid4()),
                "customer_details": {
                    "full_name": f"{customer_first_name} {customer_last_name}",
                    "email_address": customer_email,
                    "phone_number": customer_phone,
                    "delivery_address": f"{delivery_street_address}, {delivery_city}, {delivery_state} {delivery_zip_code}, {delivery_country}"
                }
            },
            "marketplace_inventory": {
                "items": order_items,
                "storefront_references": storefront_references,
                "total_items_count": len(cart),
                "final_order_value": total_amount
            }
        }
        
        # Clear cart after successful checkout
        _global_cart_store.clear_cart(shop_id)
        logger.debug(f"checkout_marketplace_cart_{shop_id}: Cleared cart after successful checkout")
        
        return json.dumps(order_response, indent=2)
        
    except Exception as e:
        logger.error(f"Checkout marketplace cart failed: {e}")
        return json.dumps({
            "purchase_outcome": "failed",
            "failure_reason": {
                "message": f"Checkout failed: {str(e)}",
                "error_type": "system_exception"
            }
        })


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        await mcp_ecommerce_data.run_sse_async()
    else:
        await mcp_ecommerce_data.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())