from api_mcp.hybrid_server_manager import HybridServerManager
from utils import calculation_results
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
import json
from typing import Dict, List, Set, Any, Tuple
from datetime import datetime
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import re
import time
import traceback
import sys
import csv
import logging

# Disable HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Import calculation function from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


load_dotenv()

URLS = {
    "URL_1": "https://webmall-1.informatik.uni-mannheim.de",
    "URL_2": "https://webmall-2.informatik.uni-mannheim.de",
    "URL_3": "https://webmall-3.informatik.uni-mannheim.de",
    "URL_4": "https://webmall-4.informatik.uni-mannheim.de",
    "URL_5": "https://webmall-solution.informatik.uni-mannheim.de"
}

# Hybrid MCP Server configurations for HTTP/SSE transport
# These servers use nlweb semantic search but return heterogeneous formats
HYBRID_MCP_SERVERS = {
    "E-Store Athletes": {
        "url": "http://localhost:8060/sse",
        "transport": "sse",
        "shop_id": "webmall_1",
        "format_type": "server_a_weird"
    },
    "TechTalk": {
        "url": "http://localhost:8061/sse",
        "transport": "sse",
        "shop_id": "webmall_2",
        "format_type": "server_b_bizarre"
    },
    "CamelCases": {
        "url": "http://localhost:8062/sse",
        "transport": "sse",
        "shop_id": "webmall_3",
        "format_type": "server_c_eccentric"
    },
    "Hardware Cafe": {
        "url": "http://localhost:8063/sse",
        "transport": "sse",
        "shop_id": "webmall_4",
        "format_type": "server_d_unconventional"
    }
}


def fill_urls(text: str, urls: Dict[str, str]) -> str:
    """Replace URL placeholders with actual URLs."""
    for key, val in urls.items():
        text = text.replace("{{" + key + "}}", val)
    return text


def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing trailing slashes and converting to lowercase."""
    return url.rstrip('/').lower()


def get_model(model_name: str = "gpt-4", temperature: float = 0.0) -> Any:
    """
    Factory function to create the appropriate model based on the provider.

    Args:
        model_name: Model identifier (e.g., "gpt-4", "claude-3-opus")
        temperature: Temperature for the model

    Returns:
        Initialized LangChain chat model
    """
    if model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("openai:"):
        # OpenAI models - handle both "gpt-4" and "openai:gpt-4" formats
        clean_model_name = model_name.replace("openai:", "")
        return ChatOpenAI(model=clean_model_name, temperature=temperature)
    elif model_name.startswith("claude"):
        # Anthropic models
        return ChatAnthropic(model=model_name, temperature=temperature)
    else:
        # Default to OpenAI
        return ChatOpenAI(model=model_name, temperature=temperature)


def extract_urls_from_response(response_text: str) -> set[str]:
    """
    Extracts URLs from the agent's final response by first trying to parse it as JSON,
    then looking for JSON patterns within mixed content, then falling back to a regex search.
    """
    if not isinstance(response_text, str):
        return set()

    # Try to parse the entire response as JSON first
    try:
        data = json.loads(response_text.strip())
        if isinstance(data, dict) and "urls" in data:
            # New JSON object format: {"urls": ["url1", "url2"]}
            urls = data["urls"]
            if isinstance(urls, list):
                return set(u for u in urls if isinstance(u, str) and u.strip().lower() != "done")
        elif isinstance(data, list):
            # Legacy JSON array format: ["url1", "url2"]
            return set(u for u in data if isinstance(u, str) and u.strip().lower() != "done")
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON patterns within mixed content
    # Look for {"urls": [...]} pattern
    json_pattern = r'\{"urls":\s*\[.*?\]\}'
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    for match in json_matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "urls" in data and isinstance(data["urls"], list):
                return set(u for u in data["urls"] if isinstance(u, str) and u.strip().lower() != "done")
        except (json.JSONDecodeError, TypeError):
            continue

    # Look for legacy JSON array pattern
    array_pattern = r'\[(?:["\'][^"\']*["\'](?:\s*,\s*)?)+\]'
    array_matches = re.findall(array_pattern, response_text)
    for match in array_matches:
        try:
            urls = json.loads(match)
            if isinstance(urls, list):
                url_set = set(u for u in urls if isinstance(
                    u, str) and u.strip().lower() != "done")
                if url_set:  # Only return if we found valid URLs
                    return url_set
        except (json.JSONDecodeError, TypeError):
            continue

    # Final fallback to regex if no JSON patterns worked
    urls_found = re.findall(r'https?://\S+', response_text)
    return set([url.strip(')>."\',') for url in urls_found])


def extract_urls_from_hybrid_response(tool_output: str, format_type: str) -> set[str]:
    """Extract URLs from hybrid MCP tool response JSON based on format type."""
    try:
        response_data = json.loads(tool_output)
        urls = set()
        
        # Debug logging for format parsing
        print(f"📋 FORMAT DEBUG - Format: {format_type}")
        print(f"  Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")

        # Handle different heterogeneous formats
        if format_type == "server_a_weird":
            # Server A search format: "results" -> "addresses" -> "selfLink"/"shareLink"
            if "results" in response_data:
                for result in response_data["results"]:
                    addresses = result.get("addresses", {})
                    if "selfLink" in addresses:
                        urls.add(addresses["selfLink"])
                    if "shareLink" in addresses:
                        urls.add(addresses["shareLink"])
            # Server A cart format: "basketContents" -> "items" -> "addresses"
            if "basketContents" in response_data:
                basket = response_data["basketContents"]
                if "items" in basket:
                    for item in basket["items"]:
                        addresses = item.get("addresses", {})
                        if "selfLink" in addresses:
                            urls.add(addresses["selfLink"])
                        if "shareLink" in addresses:
                            urls.add(addresses["shareLink"])
            # Server A checkout format: "productAddresses" or "orderDetails" -> "productAddresses"
            if "productAddresses" in response_data:
                print(f"  🏪 Server A: Found productAddresses ({len(response_data['productAddresses'])} items)")
                for addr in response_data["productAddresses"]:
                    if "selfLink" in addr:
                        urls.add(addr["selfLink"])
                        print(f"    ✅ Added URL: {addr['selfLink']}")
                    if "shareLink" in addr:
                        urls.add(addr["shareLink"])
                        print(f"    ✅ Added URL: {addr['shareLink']}")
            if "orderDetails" in response_data:
                print(f"  🏪 Server A: Found orderDetails")
                order = response_data["orderDetails"]
                if "productAddresses" in order:
                    print(f"    Found productAddresses in orderDetails ({len(order['productAddresses'])} items)")
                    for addr in order["productAddresses"]:
                        if "selfLink" in addr:
                            urls.add(addr["selfLink"])
                            print(f"      ✅ Added URL: {addr['selfLink']}")
                        if "shareLink" in addr:
                            urls.add(addr["shareLink"])
                            print(f"      ✅ Added URL: {addr['shareLink']}")

        elif format_type == "server_b_bizarre":
            # Server B search format: "catalog_entries" -> "direct_link"
            if "catalog_entries" in response_data:
                for entry in response_data["catalog_entries"]:
                    if "direct_link" in entry:
                        urls.add(entry["direct_link"])
            # Server B cart/checkout format: "shopping_container" -> "catalog_entries" -> "direct_link"
            if "shopping_container" in response_data:
                container = response_data["shopping_container"]
                if "catalog_entries" in container:
                    for entry in container["catalog_entries"]:
                        if "direct_link" in entry:
                            urls.add(entry["direct_link"])
            # Server B checkout format: "purchase_summary" -> "direct_links"
            if "purchase_summary" in response_data:
                summary = response_data["purchase_summary"]
                if "direct_links" in summary:
                    urls.update(summary["direct_links"])
                if "catalog_entries" in summary:
                    for entry in summary["catalog_entries"]:
                        if "direct_link" in entry:
                            urls.add(entry["direct_link"])

        elif format_type == "server_c_eccentric":
            # Server C search format: "warehouse_catalog" -> items -> "catalog_reference" -> "storefront_link"
            if "warehouse_catalog" in response_data:
                catalog = response_data["warehouse_catalog"]
                # Handle both list format (search) and dict format (cart)
                if isinstance(catalog, list):
                    for item in catalog:
                        catalog_ref = item.get("catalog_reference", {})
                        if "storefront_link" in catalog_ref:
                            urls.add(catalog_ref["storefront_link"])
                elif isinstance(catalog, dict):
                    # Cart format with items list
                    if "items" in catalog:
                        for item in catalog["items"]:
                            catalog_ref = item.get("catalog_reference", {})
                            if "storefront_link" in catalog_ref:
                                urls.add(catalog_ref["storefront_link"])
                    # Checkout format with catalog_references
                    if "catalog_references" in catalog:
                        for ref in catalog["catalog_references"]:
                            if "storefront_link" in ref:
                                urls.add(ref["storefront_link"])
            
            # Server C checkout format: "order_documentation" 
            if "order_documentation" in response_data:
                print(f"  🏪 Server C: Found order_documentation")
                order_doc = response_data["order_documentation"]
                
                # Look for URLs in various fields within order_documentation
                if isinstance(order_doc, dict):
                    # Check for direct URLs in order doc
                    for key, value in order_doc.items():
                        if isinstance(value, str) and value.startswith("http"):
                            urls.add(value)
                            print(f"    ✅ Added URL from {key}: {value}")
                        elif isinstance(value, list):
                            # Check if it's a list of URLs or objects with URLs
                            for item in value:
                                if isinstance(item, str) and item.startswith("http"):
                                    urls.add(item)
                                    print(f"    ✅ Added URL from {key} list: {item}")
                                elif isinstance(item, dict):
                                    # Look for storefront_link in nested objects
                                    if "storefront_link" in item:
                                        urls.add(item["storefront_link"])
                                        print(f"    ✅ Added URL from {key} object: {item['storefront_link']}")
                        elif isinstance(value, dict):
                            # Check nested objects for storefront_link
                            if "storefront_link" in value:
                                urls.add(value["storefront_link"])
                                print(f"    ✅ Added URL from {key} object: {value['storefront_link']}")
                
                    # Also check if there are product references that might contain URLs
                    if "product_references" in order_doc:
                        print(f"    Found product_references in order_documentation")
                        for ref in order_doc["product_references"]:
                            if isinstance(ref, dict) and "storefront_link" in ref:
                                urls.add(ref["storefront_link"])
                                print(f"      ✅ Added URL: {ref['storefront_link']}")
                    
                    if "items" in order_doc:
                        print(f"    Found items in order_documentation")
                        for item in order_doc["items"]:
                            if isinstance(item, dict):
                                if "url" in item:
                                    urls.add(item["url"])
                                    print(f"      ✅ Added URL: {item['url']}")
                                elif "storefront_link" in item:
                                    urls.add(item["storefront_link"])
                                    print(f"      ✅ Added URL: {item['storefront_link']}")
                                elif "catalog_reference" in item:
                                    catalog_ref = item["catalog_reference"]
                                    if isinstance(catalog_ref, dict) and "storefront_link" in catalog_ref:
                                        urls.add(catalog_ref["storefront_link"])
                                        print(f"      ✅ Added URL: {catalog_ref['storefront_link']}")

        elif format_type == "server_d_unconventional":
            # Server D search format: "marketplace_inventory" -> "storefront_reference"
            if "marketplace_inventory" in response_data:
                inventory = response_data["marketplace_inventory"]
                # Handle both list format (search) and dict format (cart)
                if isinstance(inventory, list):
                    for item in inventory:
                        if "storefront_reference" in item:
                            urls.add(item["storefront_reference"])
                elif isinstance(inventory, dict):
                    # Cart format with items list
                    if "items" in inventory:
                        for item in inventory["items"]:
                            if "storefront_reference" in item:
                                urls.add(item["storefront_reference"])
                    # Checkout format with storefront_references
                    if "storefront_references" in inventory:
                        urls.update(inventory["storefront_references"])

        print(f"  📊 Final URLs extracted: {len(urls)} URLs")
        for url in urls:
            print(f"    🔗 {url}")
        return urls
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"  ❌ Format parsing error: {e}")
        print(f"  Raw output: {tool_output[:200]}...")
        return set()


async def reset_all_hybrid_carts(tools):
    """Reset all shopping carts before each task for hybrid servers."""
    # Find all cart reset/clear tools for hybrid servers
    reset_tools = []
    for tool in tools:
        tool_name = tool.name.lower()
        # Different hybrid servers use different cart reset tool names
        if any(keyword in tool_name for keyword in [
            'reset_cart',           # Server A uses this
            'clear_shopping_cart',  # Server B uses this  
            'empty_warehouse_cart', # Server C uses this
            'clear_marketplace_cart' # Server D uses this
        ]):
            reset_tools.append(tool)
    
    print(f"Found {len(reset_tools)} cart reset tools for hybrid servers")
    for tool in reset_tools:
        try:
            await tool.ainvoke({"messages": []})
            print(f"✓ Reset cart using {tool.name}")
        except Exception as e:
            print(f"Warning: Failed to reset cart using {tool.name}: {e}")


async def ask_agent_with_retry(user_input, max_retries=3, delay=5, client=None):
    """Create and query the Hybrid MCP-powered agent with retry logic."""
    for attempt in range(max_retries):
        try:
            return await ask_agent(user_input, client=client)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed. Last error: {e}")
                # Return error response format
                return {
                    "messages": [{"content": f"Error: {str(e)}", "__class__": {"__name__": "AIMessage"}}]
                }, [], 0, 0


async def ask_agent(user_input, client=None):
    """Create and query the Hybrid MCP-powered agent using HTTP transport."""
    # Create MultiServerMCPClient with Hybrid servers if not provided
    if client is None:
        client = MultiServerMCPClient(HYBRID_MCP_SERVERS)

    # Get tools from all MCP servers
    tools = await client.get_tools()

    # Define the model being used
    model_name = os.getenv('BENCHMARK_MODEL', 'claude-sonnet-4-20250514')

    # Create proper LangChain model object
    model = get_model(model_name, temperature=0.0)

    # Create the React agent with MCP tools
    agent = create_react_agent(
        model=model,
        tools=tools
    )

    # Create system prompt aligned with nlweb_mcp benchmark style
    system_prompt = """You are an advanced e-commerce agent that can browse four hybrid stores, find products, manage shopping carts, and complete purchases.
These stores use powerful search but return data in heterogeneous formats.

AVAILABLE STORES:
- E-Store Athletes (Hybrid Store A)
- TechTalk (Hybrid Store B) 
- CamelCases (Hybrid Store C)
- Hardware Cafe (Hybrid Store D)

TASK-SPECIFIC INSTRUCTIONS:

FOR SEARCH TASKS:
- Search ALL FOUR stores using the available search tools
- If you need specific product details, fetch more details using the deticated endpoints
- Return JSON object with EXACT URLs of all relevant products found
- Be sure to check that the answers align with the user's query
- Required JSON format: {"urls": ["url1", "url2", ...]}

FOR ADD TO CART TASKS:
- First search for products across all stores
- Extract product_id from search results (NOT URLs)
- Use the appropriate add-to-cart tools with the product_id for each store
- Return URLs of products successfully added to carts
- Required JSON format: {"urls": ["url1", "url2", ...]}

FOR CHECKOUT TASKS:
- Add products to cart first using product_id from search results
- Use checkout tools with provided customer and payment details
- Cart persists across tool calls - items added will remain until checkout
- Return URLs of products from completed orders
- Required JSON format: {"urls": ["url1", "url2", ...]}

WORKFLOW AND RESPONSE FORMAT REQUIREMENTS:
- Complete ALL necessary tool operations first (search, add-to-cart, checkout as needed)
- During tool calling phases, you may provide brief explanations of your actions
- Your FINAL response (after all tools are used) MUST be valid JSON only
- Do NOT include explanatory text in your final JSON response
- Do NOT wrap the final JSON in code blocks or add any formatting

EXAMPLES OF CORRECT WORKFLOWS:

FOR SEARCH TASKS:
✓ [Brief explanation] → [Use search tools] → {"urls": ["url1", "url2"]}

FOR ADD TO CART TASKS:
✓ [Brief explanation] → [Use search tools] → [Use add-to-cart tools] → {"urls": ["url1", "url2"]}

FOR CHECKOUT TASKS:
✓ [Brief explanation] → [Use search tools] → [Use add-to-cart tools] → [Use checkout tools] → {"urls": ["url1", "url2"]}

EXAMPLES OF INCORRECT FINAL RESPONSES:
✗ Here are the products I found: {"urls": ["url1"]}
✗ ```json\n{"urls": ["url1"]}\n```
✗ I searched all stores and found these URLs: ["url1", "url2"]
✗ The search was successful. {"urls": ["url1"]}

IMPORTANT NOTES:
- Always search all four stores to find the best options
- Make sure the answer aligns with the user's query for example if the user asks for xg438qr, xg43uq would be incorrect
- When users ask for "cheapest" products, ONLY return the cheapest option(s) - do NOT include more expensive alternatives
- When users specify price constraints (e.g., "under $100", "cheapest laptop"), strictly adhere to those constraints
- If no products are found, return {"urls": []}
- Your response will be parsed by a JSON parser - any non-JSON content will cause errors"""

    with get_openai_callback() as cb:
        response = await agent.ainvoke(
            {"messages": [SystemMessage(
                content=system_prompt), HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )
        return response, tools, cb.prompt_tokens, cb.completion_tokens


def get_tool_call_results(messages, tool_call_id):
    """Extract tool call results from messages."""
    for msg in messages:
        if hasattr(msg, 'tool_call_id') and msg.tool_call_id == tool_call_id:
            return msg.content
    return None


def get_hybrid_mcp_tools_dict(tools):
    """Create mapping from tool names to MCP server names for Hybrid servers."""
    tools_dict = {}
    for tool in tools:
        tool_name = tool.name

        # Map tools to their specific server based on tool name patterns
        # E-Store Athletes tools
        if any(keyword in tool_name.lower() for keyword in ['search_products', 'get_product', 'get_categories', 'add_to_cart', 'view_cart', 'reset_cart', 'checkout']):
            tools_dict[tool_name] = "E-Store Athletes (Hybrid - Weird Format)"
        # TechTalk tools
        elif any(keyword in tool_name.lower() for keyword in ['find_items_techtalk', 'retrieve_item_details_techtalk', 'list_item_categories_techtalk', 'add_to_shopping_cart_techtalk', 'view_shopping_cart_techtalk', 'clear_shopping_cart_techtalk', 'checkout_cart_techtalk']):
            tools_dict[tool_name] = "TechTalk (Hybrid - Bizarre Format)"
        # CamelCases tools
        elif any(keyword in tool_name.lower() for keyword in ['query_stock', 'get_product_info_by_id', 'fetch_product_types', 'add_item_to_warehouse_cart', 'view_warehouse_cart', 'empty_warehouse_cart', 'process_warehouse_cart_checkout']):
            tools_dict[tool_name] = "CamelCases (Hybrid - Eccentric Format)"
        # Hardware Cafe tools
        elif any(keyword in tool_name.lower() for keyword in ['get_items_by_keyword', 'find_cheap_items', 'get_item_details_by_id', 'get_all_categories', 'add_item_to_marketplace_cart', 'view_marketplace_cart', 'clear_marketplace_cart', 'checkout_marketplace_cart']):
            tools_dict[tool_name] = "Hardware Cafe (Hybrid - Unconventional Format)"
        else:
            # Fallback for any unrecognized hybrid tools
            tools_dict[tool_name] = "Hybrid MCP Server"

    return tools_dict


def get_format_type_from_server_name(server_name: str) -> str:
    """Get format type from server name."""
    if "E-Store Athletes" in server_name:
        return "server_a_weird"
    elif "TechTalk" in server_name:
        return "server_b_bizarre"
    elif "CamelCases" in server_name:
        return "server_c_eccentric"
    elif "Hardware Cafe" in server_name:
        return "server_d_unconventional"
    else:
        return "unknown"


def create_enhanced_hybrid_tool_call_log(messages: List[Any], tools_dict: Dict[str, str]) -> Tuple[List[Dict], Set[str], Set[str], Set[str], Set[str], bool]:
    """Create comprehensive tool call log adapted for hybrid MCP servers."""
    tool_calls_log = []
    all_hybrid_urls = set()
    cart_checkout_urls = set()  # Legacy - for backward compatibility
    cart_only_urls = set()      # URLs from cart operations only (if available)
    checkout_only_urls = set()  # URLs from checkout operations only (if available)
    checkout_successful = False  # Track if any checkout operation succeeded

    # Process tool calls from hybrid MCP servers
    for msg in messages:
        if msg.__class__.__name__ == "AIMessage":
            tool_calls = msg.additional_kwargs.get("tool_calls")

            if tool_calls is None:
                tool_calls = msg.tool_calls

            if tool_calls is not None:
                for tool_call in tool_calls:
                    try:
                        tool_call_dict = dict(tool_call)
                        tool_output = get_tool_call_results(
                            messages, tool_call_dict.get("id"))

                        # Extract tool name and arguments - handle both OpenAI and Claude formats
                        tool_name = None
                        tool_args = None

                        # First try Claude/Anthropic format (direct attributes)
                        if "name" in tool_call_dict:
                            tool_name = tool_call_dict.get("name")
                            tool_args = tool_call_dict.get("args")
                        # Then try OpenAI format (nested in function)
                        elif "function" in tool_call_dict:
                            function_info = tool_call_dict.get("function", {})
                            tool_name = function_info.get("name")
                            tool_args = function_info.get("arguments")

                        # Skip if we couldn't extract a tool name
                        if not tool_name:
                            continue

                        # Parse arguments if they're a string
                        try:
                            if isinstance(tool_args, str):
                                parsed_args = json.loads(
                                    tool_args) if tool_args.strip() else {}
                            else:
                                parsed_args = tool_args if tool_args else {}
                        except (json.JSONDecodeError, TypeError):
                            parsed_args = {"raw_arguments": str(tool_args)}

                        # Determine tool type for hybrid servers
                        tool_type = "unknown"
                        # Search tools
                        if any(keyword in tool_name.lower() for keyword in ['search_products', 'get_product', 'find_items', 'query_stock', 'get_items_by_keyword', 'get_product_info_by_id', 'retrieve_item_details', 'find_cheap_items', 'get_item_details_by_id']):
                            tool_type = "search"
                        # Cart tools - be specific about each server's cart tool names
                        elif any(keyword in tool_name.lower() for keyword in ['add_to_cart', 'add_to_shopping_cart', 'add_item_to_warehouse_cart', 'add_item_to_marketplace_cart', 'view_cart', 'view_shopping_cart', 'view_warehouse_cart', 'view_marketplace_cart', 'reset_cart', 'clear_shopping_cart', 'empty_warehouse_cart', 'clear_marketplace_cart']):
                            tool_type = "cart"
                        # Checkout tools - be specific about each server's checkout tool names
                        elif any(keyword in tool_name.lower() for keyword in ['checkout', 'checkout_cart', 'process_warehouse_cart_checkout', 'checkout_marketplace_cart', 'process_customer_order', 'execute_order']):
                            tool_type = "checkout"
                        # Category tools
                        elif any(keyword in tool_name.lower() for keyword in ['categories', 'types', 'get_all_categories']):
                            tool_type = "categories"
                        
                        # Debug logging for tool detection
                        if tool_type == "checkout":
                            print(f"🔧 TOOL DEBUG - Detected CHECKOUT tool: {tool_name}")
                        elif tool_type == "cart":
                            print(f"🛒 TOOL DEBUG - Detected CART tool: {tool_name}")
                        elif tool_type == "search":
                            print(f"🔍 TOOL DEBUG - Detected SEARCH tool: {tool_name}")
                        else:
                            print(f"❓ TOOL DEBUG - Unknown tool type for: {tool_name}")

                        # Get server info and format type
                        server_name = tools_dict.get(tool_name, "Unknown")
                        format_type = get_format_type_from_server_name(
                            server_name)

                        # Extract URLs from hybrid response
                        extracted_urls = extract_urls_from_hybrid_response(
                            tool_output or "", format_type)

                        # Add to comprehensive URL tracking
                        all_hybrid_urls.update(extracted_urls)

                        # Track cart/checkout URLs separately (for future hybrid cart/checkout support)
                        if tool_type in ["cart", "checkout"]:
                            cart_checkout_urls.update(extracted_urls)

                        # Separate cart and checkout URLs
                        if tool_type == "cart":
                            cart_only_urls.update(extracted_urls)
                        elif tool_type == "checkout":
                            checkout_only_urls.update(extracted_urls)
                            # Check if checkout was successful for hybrid formats (aligned with nlweb_mcp)
                            tool_output_lower = tool_output.lower() if tool_output else ""
                            has_error_field = "error" in tool_output_lower
                            has_cart_empty_error = "cart is empty" in tool_output_lower
                            has_order_success = "order" in tool_output_lower and (
                                "created" in tool_output_lower or "successful" in tool_output_lower)
                            
                            # Debug logging for checkout detection
                            print(f"CHECKOUT DEBUG - Tool: {tool_name}")
                            print(f"  Raw output: {tool_output[:200]}...")
                            print(f"  Has error: {has_error_field}")
                            print(f"  Cart empty: {has_cart_empty_error}")
                            print(f"  Order success: {has_order_success}")
                            print(f"  URLs extracted: {list(extracted_urls)}")

                            # Use same logic as nlweb_mcp: check response_type and all error conditions
                            if format_type != "error" and not has_error_field and not has_cart_empty_error and has_order_success:
                                checkout_successful = True
                                print(f"  ✅ CHECKOUT SUCCESSFUL!")
                            else:
                                print(f"  ❌ CHECKOUT FAILED - Error: {has_error_field}, Empty cart: {has_cart_empty_error}, Success: {has_order_success}")

                        # Create comprehensive tool call entry
                        tool_call_entry = {
                            "tool_call_id": tool_call_dict.get("id"),
                            "tool_name": tool_name,
                            "tool_type": tool_type,
                            "mcp_server": server_name,
                            "format_type": format_type,
                            "tool_arguments": parsed_args,
                            "tool_output_raw": tool_output,
                            "urls_extracted": list(extracted_urls),
                            "urls_count": len(extracted_urls),
                            "timestamp": datetime.now().isoformat()
                        }

                        tool_calls_log.append(tool_call_entry)

                    except Exception as tool_parse_error:
                        print(
                            f"Error parsing hybrid tool call: {tool_parse_error}")
                        continue

    return tool_calls_log, all_hybrid_urls, cart_checkout_urls, cart_only_urls, checkout_only_urls, checkout_successful


def run_benchmark(benchmark_path):
    """Run the benchmark using Hybrid MCP servers with server lifecycle management."""
    execution_history = []

    # Initialize performance tracking
    total_execution_time = 0.0
    total_tool_calls = 0
    total_search_tools = 0
    total_cart_tools = 0
    total_checkout_tools = 0
    
    # Token tracking
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Task success tracking
    successful_tasks = 0
    failed_tasks = 0

    # Initialize HybridServerManager
    server_manager = HybridServerManager(show_logs=False)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Start servers and wrap execution in try/finally
    try:
        # Start all servers initially
        print("Starting Hybrid MCP servers...")
        if not server_manager.start_all_servers(debug=False):
            raise Exception("Failed to start all servers")
        time.sleep(10)  # Give servers time to initialize

        # Create initial MCP client to reset carts before starting
        async def reset_carts_initial():
            initial_client = None
            try:
                print("Creating initial MCP client to reset carts...")
                initial_client = MultiServerMCPClient(HYBRID_MCP_SERVERS)
                tools_for_reset = await initial_client.get_tools()
                await reset_all_hybrid_carts(tools_for_reset)
                print("Reset all shopping carts before starting benchmark")
                return tools_for_reset
            except Exception as e:
                print(f"Warning: Failed to reset carts initially: {e}")
                return None
            finally:
                # Close initial client
                if initial_client:
                    try:
                        if hasattr(initial_client, 'close'):
                            if asyncio.iscoroutinefunction(initial_client.close):
                                await initial_client.close()
                            else:
                                initial_client.close()
                    except:
                        pass

        tools_for_reset = asyncio.run(reset_carts_initial())

        for task_set in benchmark:
            #if task_set["id"] != "WEBMALL_CHECKOUT_V1" and task_set["id"] != "WEBMALL_END_TO_END_V1" and task_set["id"] != "WEBMALL_SINGLE_PRODUCT_SEARCH_V1" and task_set["id"] != "WEBMALL_ADD_TO_CART_V1":
            #    continue
            # Restart servers for each task set
            print("\n=== Restarting servers for new task set ===")
            server_manager.stop_all_servers()
            time.sleep(2)  # Give servers time to fully stop

            if not server_manager.start_all_servers(debug=False):
                print("Warning: Failed to restart servers, continuing anyway...")
            time.sleep(10)  # Give servers time to initialize

            for task in task_set["tasks"]:
                print(f"\n=== HYBRID TASK {task['id']} ===")
                print("--------------------------------")

                # Create MCP client for this task to maintain cart state
                async def setup_task_client():
                    task_client = None
                    try:
                        task_client = MultiServerMCPClient(HYBRID_MCP_SERVERS)

                        # Reset carts before each task using the task client
                        if tools_for_reset:
                            await reset_all_hybrid_carts(tools_for_reset)
                            print("Reset carts for new task")
                        return task_client
                    except Exception as e:
                        print(f"Warning: Failed to create task client or reset carts: {e}")
                        # Create a fresh client if the previous attempt failed
                        if task_client is None:
                            try:
                                task_client = MultiServerMCPClient(HYBRID_MCP_SERVERS)
                                return task_client
                            except Exception as client_error:
                                print(f"Failed to create task client: {client_error}")
                                return None
                        return task_client

                task_client = asyncio.run(setup_task_client())
                if task_client is None:
                    continue

                user_task = task["task"] if "task" in task else None
                if not user_task:
                    # your JSON nests <task> inside the "instruction" string
                    # rough extraction:
                    start = task["instruction"].find("<task>")
                    end = task["instruction"].find("</task>") + len("</task>")
                    user_task = task["instruction"][start:end]

                if "<task>" in user_task:
                    user_task = user_task.replace("<task>\n", "")
                    user_task = user_task.replace("\n</task>", "")

                if "{{email}}" in user_task:
                    # Replace product URL placeholder
                    product_urls = [fill_urls(x, URLS)
                                    for x in task["correct_answer"]["answers"]]
                    user_task = user_task.replace(
                        "{{product_url}}", str(product_urls))

                    # Replace user details placeholders
                    user_details = task["user_details"]
                    user_task = user_task.replace("{{name}}", user_details["name"])
                    user_task = user_task.replace(
                        "{{email}}", user_details["email"])
                    user_task = user_task.replace(
                        "{{street}}", user_details["street"])
                    user_task = user_task.replace(
                        "{{house_number}}", user_details["house_number"])
                    user_task = user_task.replace("{{zip}}", user_details["zip"])
                    user_task = user_task.replace("{{city}}", user_details["city"])
                    user_task = user_task.replace(
                        "{{state}}", user_details["state"])
                    user_task = user_task.replace(
                        "{{country}}", user_details["country"])

                    # Replace payment info placeholders
                    payment_info = task["payment_info"]
                    user_task = user_task.replace("{{card}}", payment_info["card"])
                    user_task = user_task.replace("{{cvv}}", payment_info["cvv"])
                    user_task = user_task.replace(
                        "{{expiry_date}}", payment_info["expiry_date"])

                # Handle different task categories
                task_category = task.get("category", "")

                if task_category == "Checkout" or task_category == "FindAndOrder":
                    # Replace product URL placeholder
                    product_urls = [fill_urls(x, URLS)
                                    for x in task["correct_answer"]["answers"]]
                    user_task = user_task.replace(
                        "{{product_url}}", str(product_urls))

                    # Replace user details placeholders  
                    user_details = task["user_details"]
                    user_task = user_task.replace(
                        "{{name}}", user_details["name"])
                    user_task = user_task.replace(
                        "{{email}}", user_details["email"])
                    user_task = user_task.replace(
                        "{{street}}", user_details["street"])
                    user_task = user_task.replace(
                        "{{house_number}}", user_details["house_number"])
                    user_task = user_task.replace(
                        "{{zip}}", user_details["zip"])
                    user_task = user_task.replace(
                        "{{state}}", user_details["state"])
                    user_task = user_task.replace(
                        "{{country}}", user_details["country"])

                    # Replace payment info placeholders
                    payment_info = task["payment_info"]
                    user_task = user_task.replace(
                        "{{card}}", payment_info["card"])
                    user_task = user_task.replace(
                        "{{cvv}}", payment_info["cvv"])
                    user_task = user_task.replace(
                        "{{expiry_date}}", payment_info["expiry_date"])

                elif task_category == "Add_To_Cart":
                    # For cart tasks, we just need to fill URLs in the task description
                    pass

                # Extract URLs from the answers dictionary and print them as a list
                correct_answer = task.get(
                    "correct_answer", {}).get("answers", [])
                expected_flat = [fill_urls(x, URLS) for x in correct_answer]

                user_task = fill_urls(user_task, URLS)
                if "{{product_url}}" in user_task:
                    user_task = user_task.replace(
                        "{{product_url}}", str(expected_flat))

                print("The user task is: " + user_task)

                # Try to run the task with error handling
                task_start_time = time.time()
                try:
                    response, tools, input_tokens, output_tokens = asyncio.run(
                        ask_agent_with_retry(user_task, client=task_client))

                    messages = response.get("messages", [])

                    # Get tool mapping for MCP servers
                    tools_dict = get_hybrid_mcp_tools_dict(tools)

                    # Track tool usage statistics
                    tool_calls_log, hybrid_urls, _, cart_only_urls, checkout_only_urls, checkout_successful = create_enhanced_hybrid_tool_call_log(
                        messages, tools_dict)

                    # Track tool usage statistics
                    for tool_log in tool_calls_log:
                        tool_type = tool_log.get("tool_type", "unknown")
                        if tool_type == "search":
                            total_search_tools += 1
                        elif tool_type == "cart":
                            total_cart_tools += 1
                        elif tool_type == "checkout":
                            total_checkout_tools += 1

                    total_tool_calls += len(tool_calls_log)
                    successful_tasks += 1
                    task_error = None

                except Exception as e:
                    print(f"Task failed with error: {e}")
                    print(f"Full traceback: {traceback.format_exc()}")

                    # Create minimal error response to continue processing
                    response = {"messages": []}
                    tools = []
                    input_tokens = 0
                    output_tokens = 0
                    messages = []
                    tools_dict = {}
                    tool_calls_log = []
                    hybrid_urls = set()
                    cart_only_urls = set()
                    checkout_only_urls = set()
                    checkout_successful = False
                    task_error = str(e)
                    failed_tasks += 1

                # Calculate task execution time
                task_execution_time = time.time() - task_start_time
                total_execution_time += task_execution_time

                # Track token usage
                total_prompt_tokens += input_tokens
                total_completion_tokens += output_tokens

                # Print final AI messages for debugging
                for msg in messages:
                    if msg.__class__.__name__ == "AIMessage":
                        # Only print the final answer, not the intermediate tool-calling messages
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            print("\n--- AI Message ---")
                            print(msg.content)
                            print("--------------------------------")

                # Handle final answer processing with error handling
                if messages and len(messages) > 0:
                    # Look for the final non-tool-call message as the answer
                    final_answer = None
                    for msg in reversed(messages):
                        if (hasattr(msg, '__class__') and msg.__class__.__name__ == "AIMessage" and
                            hasattr(msg, 'content') and msg.content and
                                not (hasattr(msg, 'tool_calls') and msg.tool_calls)):
                            final_answer = msg.content.strip()
                            break

                    if final_answer is None:
                        # Fallback to last message with content
                        final_answer = messages[-1].content.strip() if hasattr(
                            messages[-1], 'content') else "Error: No response content"

                    print("Final Answer: " + final_answer)
                else:
                    final_answer = f"Error: {task_error}" if 'task_error' in locals(
                    ) else "Error: No response received"
                    print("Final Answer: " + final_answer)

                got = extract_urls_from_response(final_answer)

                # Normalize both got and expected URLs
                got_normalized = [normalize_url(url) for url in got]
                expected_normalized = [normalize_url(url) for url in expected_flat]

                # Determine task category and evaluation strategy
                task_category = task.get("category", "search")

                # Apply differentiated evaluation based on task category
                evaluation_urls = []
                evaluation_strategy = "search_only"  # Default strategy

                print(f"\n🎯 EVALUATION DEBUG - Task Category: {task_category}")
                print(f"  Checkout successful: {checkout_successful}")
                print(f"  Cart only URLs: {list(cart_only_urls)}")
                print(f"  Checkout only URLs: {list(checkout_only_urls)}")
                print(f"  Final answer URLs: {list(got)}")

                if task_category == "Add_To_Cart":
                    # For cart tasks, only evaluate cart operations
                    evaluation_urls = [normalize_url(url)
                                       for url in cart_only_urls]
                    evaluation_strategy = "cart_only"
                    print(f"  ✅ Using CART strategy: {len(evaluation_urls)} URLs")
                elif task_category in ["Checkout", "FindAndOrder"]:
                    # For checkout tasks, only give credit if checkout was successful
                    if checkout_successful:
                        evaluation_urls = [normalize_url(
                            url) for url in checkout_only_urls]
                        evaluation_strategy = "checkout_successful"
                        print(f"  ✅ Using CHECKOUT SUCCESSFUL strategy: {len(evaluation_urls)} URLs")
                        
                        # FALLBACK: If checkout was successful but no URLs extracted from checkout response,
                        # use final answer URLs as fallback (this handles parsing issues)
                        if not evaluation_urls and got_normalized:
                            evaluation_urls = got_normalized
                            evaluation_strategy = "checkout_successful_fallback"
                            print(f"  🔄 FALLBACK: Checkout successful but no URLs extracted, using final answer: {len(evaluation_urls)} URLs")
                    else:
                        evaluation_urls = []  # No credit for failed checkout
                        evaluation_strategy = "checkout_failed"
                        print(f"  ❌ Using CHECKOUT FAILED strategy: No credit given")
                else:
                    # For search tasks, use final response URLs
                    evaluation_urls = got_normalized
                    evaluation_strategy = "search_only"
                    print(f"  🔍 Using SEARCH strategy: {len(evaluation_urls)} URLs")

                # Calculate metrics using the utils function
                task_metrics = calculation_results(
                    expected_normalized, evaluation_urls)
                print(
                    f"Metrics for Task {task['id']} (Strategy: {evaluation_strategy}): {task_metrics}")

                # correct answer urls (based on evaluation strategy)
                correct_model_answers = [
                    url for url in expected_flat if normalize_url(url) in evaluation_urls]

                # additional urls that are not in the correct answers (based on evaluation strategy)
                additional_urls = [
                    url for url in evaluation_urls if url not in expected_normalized]

                # urls that are in the correct answers but not in the model response (based on evaluation strategy)
                missing_urls = [
                    url for url in expected_normalized if url not in evaluation_urls]

                # Compare Hybrid URLs with ground truth
                hybrid_urls_normalized = [
                    normalize_url(url) for url in hybrid_urls]
                hybrid_correct_retrieved = [
                    url for url in hybrid_urls_normalized if url in expected_normalized]
                hybrid_additional_retrieved = [
                    url for url in hybrid_urls_normalized if url not in expected_normalized]

                # Calculate Hybrid metrics using utils function
                hybrid_metrics = calculation_results(
                    expected_normalized, hybrid_urls_normalized)

                # Enhanced history entry with detailed evaluation metadata
                history_entry = {
                    "task_id": task["id"],
                    "task_category": task_category,
                    "task_completion_rate": task_metrics["task_completion_rate"],
                    "precision": task_metrics["avg_precision"],
                    "recall": task_metrics["avg_recall"],
                    "f1_score": task_metrics["f1_score"],
                    "task": user_task,
                    "raw_response": final_answer,
                    "correct_model_answers": correct_model_answers,
                    "additional_urls": additional_urls,
                    "missing_urls": missing_urls,
                    "metrics": task_metrics,
                    "evaluation_strategy": evaluation_strategy,
                    "evaluation_urls": evaluation_urls,
                    "cart_only_urls": list(cart_only_urls),
                    "checkout_only_urls": list(checkout_only_urls),
                    "checkout_successful": checkout_successful,
                    "hybrid_correct_retrieved": hybrid_correct_retrieved,
                    "hybrid_additional_retrieved": hybrid_additional_retrieved,
                    "hybrid_metrics": hybrid_metrics,
                    "tool_calls": tool_calls_log,
                    "response": str(response),
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "execution_time_seconds": task_execution_time,
                    "total_tool_calls": len(tool_calls_log),
                    "search_tool_calls": len([t for t in tool_calls_log if t.get("tool_type") == "search"]),
                    "cart_tool_calls": len([t for t in tool_calls_log if t.get("tool_type") == "cart"]),
                    "checkout_tool_calls": len([t for t in tool_calls_log if t.get("tool_type") == "checkout"]),
                    "approach": "hybrid_semantic_search_heterogeneous_formats"
                }

                # Add error information if task failed
                if 'task_error' in locals() and task_error:
                    history_entry["error"] = task_error
                    history_entry["error_occurred"] = True
                else:
                    history_entry["error_occurred"] = False

                execution_history.append(history_entry)

                # Save the history entry to a file
                with open("history_entry.json", "w") as f:
                    json.dump(history_entry, f, indent=4)

                print("Correct: " + str(correct_model_answers))
                print("Model Response: " + str(evaluation_urls))
                print("Additional: " + str(additional_urls))
                print("Missing: " + str(missing_urls))
                print("--------------------------------" + "\n"*10)

                # Clean up task client
                if 'task_client' in locals() and task_client:
                    try:
                        # Check if client has a close method
                        if hasattr(task_client, 'close'):
                            if asyncio.iscoroutinefunction(task_client.close):
                                asyncio.run(task_client.close())
                            else:
                                task_client.close()
                        elif hasattr(task_client, 'cleanup'):
                            if asyncio.iscoroutinefunction(task_client.cleanup):
                                asyncio.run(task_client.cleanup())
                            else:
                                task_client.cleanup()
                        else:
                            # No cleanup method available, just delete reference
                            print(f"No cleanup method available for task client, skipping cleanup")
                    except Exception as cleanup_error:
                        print(f"Warning: Failed to close task client: {cleanup_error}")
                

    finally:
        # Always stop servers on exit
        print("\nStopping all servers...")
        server_manager.stop_all_servers()

    # Generate enhanced benchmark summary
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    avg_execution_time = total_execution_time / \
        len(execution_history) if execution_history else 0

    enhanced_summary = {
        "benchmark_metadata": {
            "timestamp": current_timestamp,
            "version": "hybrid_mcp_enhanced",
            "total_tasks": len(execution_history),
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / len(execution_history) if execution_history else 0,
            "total_execution_time_seconds": total_execution_time,
            "average_execution_time_seconds": avg_execution_time,
            "total_tool_calls": total_tool_calls,
            "avg_tool_calls_per_task": total_tool_calls / len(execution_history) if execution_history else 0
        },
        "tool_usage_summary": {
            "search_tools": total_search_tools,
            "cart_tools": total_cart_tools,
            "checkout_tools": total_checkout_tools
        },
        "token_usage_summary": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "avg_tokens_per_task": (total_prompt_tokens + total_completion_tokens) / len(execution_history) if execution_history else 0
        },
        "results": execution_history
    }

    # Print comprehensive statistics
    print("\n" + "=" * 60)
    print("ENHANCED HYBRID MCP BENCHMARK RESULTS")
    print("=" * 60)
    print(f"🎯 Total Tasks: {len(execution_history)}")
    print(f"✅ Successful: {successful_tasks}")
    print(f"❌ Failed: {failed_tasks}")
    print(
        f"📊 Success Rate: {(successful_tasks / len(execution_history) * 100):.1f}%" if execution_history else "0%")
    print(f"⏱️  Total Execution Time: {total_execution_time:.2f} seconds")
    print(f"⚡ Average per Task: {avg_execution_time:.2f} seconds")
    print(f"\n🛠️  TOOL USAGE BREAKDOWN:")
    print(f"  - Search Tools: {total_search_tools}")
    print(f"  - Cart Tools: {total_cart_tools}")
    print(f"  - Checkout Tools: {total_checkout_tools}")
    print(f"  - Total Tool Calls: {total_tool_calls}")
    print(
        f"  - Avg per Task: {(total_tool_calls / len(execution_history)):.1f}" if execution_history else "0")
    print(f"\n💭 TOKEN USAGE:")
    print(f"  - Prompt Tokens: {total_prompt_tokens:,}")
    print(f"  - Completion Tokens: {total_completion_tokens:,}")
    print(
        f"  - Total Tokens: {(total_prompt_tokens + total_completion_tokens):,}")
    print("=" * 60)

    return enhanced_summary


def generate_csv_metrics(enhanced_summary):
    """Generate CSV file with task metrics for easy analysis."""
    if not enhanced_summary.get("results"):
        print("No results to export to CSV")
        return

    current_timestamp = enhanced_summary["benchmark_metadata"]["timestamp"]
    csv_filename = f"hybrid_mcp_enhanced_metrics_{current_timestamp}.csv"

    # Prepare CSV data
    csv_data = []
    for result in enhanced_summary["results"]:

        csv_row = {
            "task_category": result.get("task_id", "").split("_Task")[0],
            "task_id": result.get("task_id", ""),
            "task_completion_rate": result.get("task_completion_rate", 0),
            "precision": result.get("precision", 0),
            "recall": result.get("recall", 0),
            "f1_score": result.get("f1_score", 0),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "execution_time_seconds": result.get("execution_time_seconds", 0),
            "total_tool_calls": result.get("total_tool_calls", 0),
            "search_tool_calls": result.get("search_tool_calls", 0),
            "cart_tool_calls": result.get("cart_tool_calls", 0),
            "checkout_tool_calls": result.get("checkout_tool_calls", 0),
            "evaluation_strategy": result.get("evaluation_strategy", ""),
            "checkout_successful": result.get("checkout_successful", False),
            "error_occurred": result.get("error_occurred", False)
        }
        csv_data.append(csv_row)

    # Write CSV file
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["task_category", "task_id", "task_completion_rate", "precision",
                          "recall", "f1_score", "prompt_tokens", "completion_tokens",
                          "execution_time_seconds", "total_tool_calls", "search_tool_calls",
                          "cart_tool_calls", "checkout_tool_calls", "evaluation_strategy",
                          "checkout_successful", "error_occurred"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        print(f"📊 CSV metrics exported to: {csv_filename}")
        return csv_filename
    except Exception as e:
        print(f"Failed to generate CSV: {e}")
        return None


if __name__ == "__main__":
    BENCHMARK_JSON_PATH = "../task_sets.json"  # adjust as needed
    enhanced_summary = {}

    try:
        enhanced_summary = run_benchmark(BENCHMARK_JSON_PATH)
    except Exception as e:
        print(f"Benchmark run failed with error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")

        # Even if the benchmark failed, try to create minimal summary
        enhanced_summary = {
            "benchmark_metadata": {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "version": "hybrid_mcp_enhanced",
                "error": str(e),
                "traceback": traceback.format_exc()
            },
            "results": []
        }
    finally:
        # Always save results, even if there were errors
        current_timestamp = enhanced_summary.get("benchmark_metadata", {}).get(
            "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_file = f"hybrid_execution_history_{current_timestamp}.json"

        try:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(enhanced_summary, f, indent=4)
            print(f"📁 Enhanced results saved to: {output_file}")

            # Generate CSV metrics if we have results
            if enhanced_summary.get("results"):
                generate_csv_metrics(enhanced_summary)

        except Exception as save_error:
            print(f"Failed to save enhanced results: {save_error}")
            # As a last resort, print the results to console
            print("Enhanced results (failed to save to file):")
            print(json.dumps(enhanced_summary, indent=2))
