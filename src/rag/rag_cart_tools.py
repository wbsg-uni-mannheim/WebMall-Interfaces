"""
RAG Cart and Checkout Tools for WebMall shops.
These tools enable add to cart and checkout functionality for the RAG system.
"""

import uuid
from typing import Dict, List, Optional
from datetime import datetime
from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

# Global cart state for all shops
SHOP_CARTS = {
    "webmall_1": {},
    "webmall_2": {},
    "webmall_3": {},
    "webmall_4": {}
}

# Shop names for display
SHOP_NAMES = {
    "webmall_1": "E-Store Athletes",
    "webmall_2": "TechTalk", 
    "webmall_3": "CamelCases",
    "webmall_4": "Hardware Cafe"
}

# Shop URLs
SHOP_URLS = {
    "webmall_1": "https://webmall-1.informatik.uni-mannheim.de",
    "webmall_2": "https://webmall-2.informatik.uni-mannheim.de",
    "webmall_3": "https://webmall-3.informatik.uni-mannheim.de",
    "webmall_4": "https://webmall-4.informatik.uni-mannheim.de"
}


class AddToCartInput(BaseModel):
    """Input schema for add to cart tool."""
    urls: List[str] = Field(description="List of product URLs to add to cart")
    quantities: Optional[List[int]] = Field(
        default=None, 
        description="List of quantities for each URL (defaults to 1 for each)"
    )


class CheckoutInput(BaseModel):
    """Input schema for checkout tool."""
    first_name: str = Field(description="Customer's first name")
    last_name: str = Field(description="Customer's last name")
    email: str = Field(description="Customer's email address")
    phone: str = Field(description="Customer's phone number")
    address_1: str = Field(description="Customer's street address")
    city: str = Field(description="Customer's city")
    state: str = Field(description="Customer's state/province")
    postcode: str = Field(description="Customer's postal/zip code")
    country: str = Field(description="Customer's country")
    credit_card_number: str = Field(description="Credit card number")
    credit_card_expiry: str = Field(description="Credit card expiry date (MM/YY)")
    credit_card_cvc: str = Field(description="Credit card CVC code")


def reset_cart(shop_id: str):
    """Reset cart for a specific shop."""
    SHOP_CARTS[shop_id] = {}


def reset_all_carts():
    """Reset all shopping carts."""
    for shop_id in SHOP_CARTS:
        SHOP_CARTS[shop_id] = {}


def extract_product_info_from_url(url: str) -> Dict[str, str]:
    """Extract product information from URL."""
    # Extract product slug from URL
    parts = url.rstrip('/').split('/')
    if len(parts) >= 2 and parts[-2] == 'product':
        product_slug = parts[-1]
        # Convert slug to readable name
        product_name = product_slug.replace('-', ' ').title()
        return {
            "product_id": product_slug,
            "name": product_name,
            "url": url
        }
    else:
        # Fallback for non-standard URLs
        return {
            "product_id": url.split('/')[-1] or "unknown",
            "name": "Product from " + url,
            "url": url
        }


async def add_to_cart_webmall_1(urls: List[str], quantities: Optional[List[int]] = None) -> Dict:
    """Add products to WebMall-1 (E-Store Athletes) shopping cart."""
    return await add_to_cart_generic("webmall_1", urls, quantities)


async def add_to_cart_webmall_2(urls: List[str], quantities: Optional[List[int]] = None) -> Dict:
    """Add products to WebMall-2 (TechTalk) shopping cart."""
    return await add_to_cart_generic("webmall_2", urls, quantities)


async def add_to_cart_webmall_3(urls: List[str], quantities: Optional[List[int]] = None) -> Dict:
    """Add products to WebMall-3 (CamelCases) shopping cart."""
    return await add_to_cart_generic("webmall_3", urls, quantities)


async def add_to_cart_webmall_4(urls: List[str], quantities: Optional[List[int]] = None) -> Dict:
    """Add products to WebMall-4 (Hardware Cafe) shopping cart."""
    return await add_to_cart_generic("webmall_4", urls, quantities)


async def add_to_cart_generic(shop_id: str, urls: List[str], quantities: Optional[List[int]] = None) -> Dict:
    """Generic add to cart implementation."""
    try:
        cart = SHOP_CARTS[shop_id]
        
        # Default quantities to 1 if not provided
        if quantities is None:
            quantities = [1] * len(urls)
        elif len(quantities) != len(urls):
            return {
                "success": False,
                "error": f"Number of quantities ({len(quantities)}) must match number of URLs ({len(urls)})",
                "cart": list(cart.values())
            }
        
        added_items = []
        
        for url, quantity in zip(urls, quantities):
            # Validate the URL belongs to this shop
            expected_domain = SHOP_URLS[shop_id]
            if not url.startswith(expected_domain):
                continue  # Skip URLs from other shops
            
            product_info = extract_product_info_from_url(url)
            product_id = product_info["product_id"]
            
            # Add or update cart item
            if product_id in cart:
                cart[product_id]["quantity"] += quantity
            else:
                cart[product_id] = {
                    "product_id": product_id,
                    "name": product_info["name"],
                    "url": url,
                    "quantity": quantity
                }
            
            added_items.append(product_info["name"])
        
        # Get all product URLs from cart for evaluation
        cart_urls = [item["url"] for item in cart.values()]
        
        return {
            "success": True,
            "message": f"Added {len(added_items)} items to {SHOP_NAMES[shop_id]} cart",
            "added_items": added_items,
            "cart": list(cart.values()),
            "cart_urls": cart_urls,
            "total_items": sum(item["quantity"] for item in cart.values()),
            "shop_id": shop_id,
            "shop_name": SHOP_NAMES[shop_id]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "shop_id": shop_id,
            "cart": list(SHOP_CARTS[shop_id].values())
        }


async def checkout_webmall_1(
    first_name: str, last_name: str, email: str, phone: str,
    address_1: str, city: str, state: str, postcode: str, country: str,
    credit_card_number: str, credit_card_expiry: str, credit_card_cvc: str
) -> Dict:
    """Complete checkout for WebMall-1 (E-Store Athletes)."""
    return await checkout_generic(
        "webmall_1", first_name, last_name, email, phone,
        address_1, city, state, postcode, country,
        credit_card_number, credit_card_expiry, credit_card_cvc
    )


async def checkout_webmall_2(
    first_name: str, last_name: str, email: str, phone: str,
    address_1: str, city: str, state: str, postcode: str, country: str,
    credit_card_number: str, credit_card_expiry: str, credit_card_cvc: str
) -> Dict:
    """Complete checkout for WebMall-2 (TechTalk)."""
    return await checkout_generic(
        "webmall_2", first_name, last_name, email, phone,
        address_1, city, state, postcode, country,
        credit_card_number, credit_card_expiry, credit_card_cvc
    )


async def checkout_webmall_3(
    first_name: str, last_name: str, email: str, phone: str,
    address_1: str, city: str, state: str, postcode: str, country: str,
    credit_card_number: str, credit_card_expiry: str, credit_card_cvc: str
) -> Dict:
    """Complete checkout for WebMall-3 (CamelCases)."""
    return await checkout_generic(
        "webmall_3", first_name, last_name, email, phone,
        address_1, city, state, postcode, country,
        credit_card_number, credit_card_expiry, credit_card_cvc
    )


async def checkout_webmall_4(
    first_name: str, last_name: str, email: str, phone: str,
    address_1: str, city: str, state: str, postcode: str, country: str,
    credit_card_number: str, credit_card_expiry: str, credit_card_cvc: str
) -> Dict:
    """Complete checkout for WebMall-4 (Hardware Cafe)."""
    return await checkout_generic(
        "webmall_4", first_name, last_name, email, phone,
        address_1, city, state, postcode, country,
        credit_card_number, credit_card_expiry, credit_card_cvc
    )


async def checkout_generic(
    shop_id: str, first_name: str, last_name: str, email: str, phone: str,
    address_1: str, city: str, state: str, postcode: str, country: str,
    credit_card_number: str, credit_card_expiry: str, credit_card_cvc: str
) -> Dict:
    """Generic checkout implementation."""
    try:
        cart = SHOP_CARTS[shop_id]
        
        if not cart:
            return {
                "success": False,
                "error": "Cart is empty. Please add items to cart before checkout.",
                "shop_id": shop_id,
                "shop_name": SHOP_NAMES[shop_id]
            }
        
        # Validate required fields
        required_fields = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "address_1": address_1,
            "city": city,
            "postcode": postcode,
            "country": country,
            "credit_card_number": credit_card_number,
            "credit_card_expiry": credit_card_expiry,
            "credit_card_cvc": credit_card_cvc
        }
        
        missing_fields = [k for k, v in required_fields.items() if not v]
        if missing_fields:
            return {
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "shop_id": shop_id
            }
        
        # Generate order ID
        order_id = f"{shop_id}_order_{str(uuid.uuid4())[:8]}"
        
        # Get product URLs before clearing cart
        product_urls = [item["url"] for item in cart.values()]
        order_items = list(cart.values())
        
        # Clear cart after successful checkout
        cart.clear()
        
        # Create order response
        order_response = {
            "success": True,
            "message": "Order created successfully",
            "order_id": order_id,
            "shop_id": shop_id,
            "shop_name": SHOP_NAMES[shop_id],
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
            "items": order_items,
            "product_urls": product_urls,
            
            "payment_method": "credit_card",
            "transaction_id": str(uuid.uuid4()),
            "order_date": datetime.now().isoformat()
        }
        
        return order_response
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "shop_id": shop_id,
            "shop_name": SHOP_NAMES[shop_id]
        }


# Create LangChain tool instances
def get_cart_tools() -> List[StructuredTool]:
    """Get all cart and checkout tools for WebMall shops."""
    tools = []
    
    # Add to cart tools
    for shop_num in range(1, 5):
        shop_id = f"webmall_{shop_num}"
        shop_name = SHOP_NAMES[shop_id]
        
        # Add to cart tool using StructuredTool
        add_tool = StructuredTool(
            name=f"add_to_cart_{shop_id}",
            description=f"Add products to {shop_name} shopping cart. Accepts list of product URLs and optional quantities.",
            coroutine=globals()[f"add_to_cart_{shop_id}"],
            args_schema=AddToCartInput
        )
        tools.append(add_tool)
        
        # Checkout tool using StructuredTool
        checkout_tool = StructuredTool(
            name=f"checkout_{shop_id}",
            description=f"Complete checkout for {shop_name}. Requires customer details and payment information.",
            coroutine=globals()[f"checkout_{shop_id}"],
            args_schema=CheckoutInput
        )
        tools.append(checkout_tool)
    
    return tools