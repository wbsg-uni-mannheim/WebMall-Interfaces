from nlweb_mcp.start_all_servers import ServerManager
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
import re
import time
import traceback
import logging
import csv
import sys

# Import calculation function from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


load_dotenv()

# Disable HTTP request logging
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("openai").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)

URLS = {
    "URL_1": "https://webmall-1.informatik.uni-mannheim.de",
    "URL_2": "https://webmall-2.informatik.uni-mannheim.de",
    "URL_3": "https://webmall-3.informatik.uni-mannheim.de",
    "URL_4": "https://webmall-4.informatik.uni-mannheim.de",
    "URL_5": "https://webmall-solution.informatik.uni-mannheim.de"
}

# NLWeb MCP Server configurations for HTTP/SSE transport
NLWEB_MCP_SERVERS = {
    "WebMall-1": {
        "url": "http://localhost:8001/sse",
        "transport": "sse",
        "shop_id": "webmall_1"
    },
    "WebMall-2": {
        "url": "http://localhost:8002/sse",
        "transport": "sse",
        "shop_id": "webmall_2"
    },
    "WebMall-3": {
        "url": "http://localhost:8003/sse",
        "transport": "sse",
        "shop_id": "webmall_3"
    },
    "WebMall-4": {
        "url": "http://localhost:8004/sse",
        "transport": "sse",
        "shop_id": "webmall_4"
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
                url_set = set(u for u in urls if isinstance(u, str) and u.strip().lower() != "done")
                if url_set:  # Only return if we found valid URLs
                    return url_set
        except (json.JSONDecodeError, TypeError):
            continue
    
    # Final fallback to regex if no JSON patterns worked
    urls_found = re.findall(r'https?://\S+', response_text)
    return set([url.strip(')>."\',') for url in urls_found])


def extract_urls_from_mcp_response(tool_output: str, tool_name: str = "") -> Dict[str, Any]:
    """Enhanced URL extraction from MCP tool response JSON with detailed metadata."""
    result = {
        "urls": set(),
        "response_type": "unknown",
        "metadata": {},
        "raw_response": tool_output
    }

    if not tool_output or not isinstance(tool_output, str):
        return result

    try:
        response_data = json.loads(tool_output)
        result["parsed_response"] = response_data

        # Handle search results (ask_webmall_X tools)
        if "results" in response_data:
            result["response_type"] = "search"
            result["metadata"]["results_count"] = len(response_data["results"])
            for result_item in response_data["results"]:
                if "url" in result_item:
                    result["urls"].add(result_item["url"])

            # Additional search metadata
            if "query" in response_data:
                result["metadata"]["query"] = response_data["query"]
            if "total_results" in response_data:
                result["metadata"]["total_results"] = response_data["total_results"]

        # Handle cart responses (add_to_cart_webmall_X, view_cart_webmall_X)
        elif "cart" in response_data:
            result["response_type"] = "cart"
            if isinstance(response_data["cart"], list):
                result["metadata"]["cart_items_count"] = len(
                    response_data["cart"])
                for item in response_data["cart"]:
                    if "url" in item:
                        result["urls"].add(item["url"])

            # Track cart-specific URLs if available
            if "cart_urls" in response_data:
                result["urls"].update(response_data["cart_urls"])
                result["metadata"]["cart_urls_count"] = len(
                    response_data["cart_urls"])

            # Additional cart metadata
            if "total_items" in response_data:
                result["metadata"]["total_items"] = response_data["total_items"]
            if "total_price" in response_data:
                result["metadata"]["total_price"] = response_data["total_price"]

        # Handle checkout responses (checkout_webmall_X)
        elif "product_urls" in response_data or "items" in response_data:
            result["response_type"] = "checkout"

            # Priority to product_urls if available
            if "product_urls" in response_data and isinstance(response_data["product_urls"], list):
                result["urls"].update(response_data["product_urls"])
                result["metadata"]["product_urls_count"] = len(
                    response_data["product_urls"])

            # Fallback to extracting from items
            elif "items" in response_data and isinstance(response_data["items"], list):
                result["metadata"]["checkout_items_count"] = len(
                    response_data["items"])
                for item in response_data["items"]:
                    if "url" in item:
                        result["urls"].add(item["url"])

            # Additional checkout metadata
            if "order_id" in response_data:
                result["metadata"]["order_id"] = response_data["order_id"]
            if "total" in response_data:
                result["metadata"]["total_amount"] = response_data["total"]
            if "status" in response_data:
                result["metadata"]["checkout_status"] = response_data["status"]

        # Handle reset cart responses or other generic responses
        else:
            result["response_type"] = "generic"
            if "status" in response_data:
                result["metadata"]["status"] = response_data["status"]
            if "message" in response_data:
                result["metadata"]["message"] = response_data["message"]

        # Try to extract any URLs that might be in other fields
        urls_in_response = re.findall(r'https?://\S+', tool_output)
        if urls_in_response:
            result["urls"].update([url.strip('"\\,}')
                                  for url in urls_in_response])

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        result["response_type"] = "error"
        result["metadata"]["parse_error"] = str(e)

        # Fallback regex extraction for malformed JSON
        urls_in_response = re.findall(r'https?://\S+', tool_output)
        if urls_in_response:
            result["urls"].update([url.strip('"\\,}')
                                  for url in urls_in_response])

    return result


async def reset_all_carts(tools):
    """Reset all shopping carts before each task."""
    reset_tools = [
        tool for tool in tools if tool.name.startswith("reset_cart_")]
    for tool in reset_tools:
        try:
            await tool.ainvoke({"messages": []})
        except Exception as e:
            print(f"Warning: Failed to reset cart using {tool.name}: {e}")


async def ask_agent_with_retry(user_input, max_retries=3, delay=5, client=None):
    """Create and query the NLWeb MCP-powered agent with retry logic."""
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
    """Create and query the NLWeb MCP-powered agent using HTTP transport."""
    # Create MultiServerMCPClient with NLWeb servers if not provided
    if client is None:
        client = MultiServerMCPClient(NLWEB_MCP_SERVERS)

    # Get tools from all MCP servers
    tools = await client.get_tools()

    # Define the model being used
    model_name = f"openai:{os.getenv('BENCHMARK_MODEL', 'gpt-4.1')}" if os.getenv('BENCHMARK_MODEL', 'gpt-4.1').startswith('gpt') else f"anthropic:{os.getenv('BENCHMARK_MODEL', 'gpt-4.1')}"
    # Create the React agent with MCP tools
    agent = create_react_agent(
        model_name,
        tools
    )

    with get_openai_callback() as cb:
        response = await agent.ainvoke(
            {"messages": [
                {"role": "system",
                 "content": """You are an advanced e-commerce agent that can browse four webshops, find products, manage shopping carts, and complete purchases.

AVAILABLE TOOLS:
- ask_webmall_1 through ask_webmall_4: Search for products in specific shops
- add_to_cart_webmall_1 through add_to_cart_webmall_4: Add products to specific shop carts using product_id
- view_cart_webmall_1 through view_cart_webmall_4: View current cart contents
- checkout_webmall_1 through checkout_webmall_4: Complete purchases with customer details

AVAILABLE SHOPS:
- WebMall-1 (E-Store Athletes)
- WebMall-2 (TechTalk)
- WebMall-3 (CamelCases)
- WebMall-4 (Hardware Cafe)

TASK-SPECIFIC INSTRUCTIONS:

FOR SEARCH TASKS:
- Search ALL FOUR stores using ask_webmall_X tools with the same query
- Return JSON object with EXACT URLs of all relevant products found
- Be sure to check that the answers align with the user's query
- Required JSON format: {"urls": ["url1", "url2", ...]}

FOR ADD TO CART TASKS:
- First search for products across all shops
- Extract product_id from search results (NOT URLs)
- Use add_to_cart_webmall_X with the product_id for each shop
- Return URLs of products successfully added to carts
- Required JSON format: {"urls": ["url1", "url2", ...]}

FOR CHECKOUT TASKS:
- Add products to cart first using product_id from search
- Use checkout_webmall_X with provided customer and payment details
- Cart persists across tool calls - items added will remain until checkout
- Return URLs of products from completed orders
- Required JSON format: {"urls": ["url1", "url2", ...], "status": "completed"}

WORKFLOW AND RESPONSE FORMAT REQUIREMENTS:
- Complete ALL necessary tool operations first (search, add-to-cart, checkout as needed)
- During tool calling phases, you may provide brief explanations of your actions
- Your FINAL response (after all tools are used) MUST be valid JSON only
- Do NOT include explanatory text in your final JSON response
- Do NOT wrap the final JSON in code blocks or add any formatting

EXAMPLES OF CORRECT WORKFLOWS:

FOR SEARCH TASKS:
✓ [Brief explanation] → [Use ask_webmall_X tools] → {"urls": ["url1", "url2"]}

FOR ADD TO CART TASKS:
✓ [Brief explanation] → [Use ask_webmall_X tools] → [Use add_to_cart_webmall_X tools] → {"urls": ["url1", "url2"]}

FOR CHECKOUT TASKS:
✓ [Brief explanation] → [Use ask_webmall_X tools] → [Use add_to_cart_webmall_X tools] → [Use checkout_webmall_X tools] → {"urls": ["url1", "url2"], "status": "completed"}

EXAMPLES OF INCORRECT FINAL RESPONSES:
✗ Here are the products I found: {"urls": ["url1"]}
✗ ```json\n{"urls": ["url1"]}\n```
✗ I searched all stores and found these URLs: ["url1", "url2"]
✗ The search was successful. {"urls": ["url1"]}

IMPORTANT NOTES:
- Always search all four shops to find the best options
- Make sure the answer aligns with the user's query for example if the user asks for xg438qr, xg43uq would be incorrect
- When users ask for "cheapest" products, ONLY return the cheapest option(s) - do NOT include more expensive alternatives
- When users specify price constraints (e.g., "under $100", "cheapest laptop"), strictly adhere to those constraints
- If no products are found, return {"urls": []}
- Your response will be parsed by a JSON parser - any non-JSON content will cause errors
                    
                    """},
                {"role": "user", "content": user_input}
            ]}
        )
        return response, tools, cb.prompt_tokens, cb.completion_tokens


def get_tool_call_results(messages, tool_call_id):
    """Extract tool call results from messages."""
    for msg in messages:
        if hasattr(msg, 'tool_call_id') and msg.tool_call_id == tool_call_id:
            return msg.content
    return None


def create_enhanced_tool_call_log(messages: List[Any], tools_dict: Dict[str, str]) -> Tuple[List[Dict], Set[str], Set[str], Set[str], Set[str], bool]:
    """Create comprehensive tool call log adapted to work with original MCP message processing."""
    tool_calls_log = []
    all_mcp_urls = set()
    cart_checkout_urls = set()  # Legacy - for backward compatibility
    cart_only_urls = set()      # URLs from add_to_cart operations only
    checkout_only_urls = set()  # URLs from checkout operations only
    checkout_successful = False  # Track if any checkout operation succeeded

    # Use the original MCP tool call processing logic that was working
    for msg in messages:
        if msg.__class__.__name__ == "AIMessage":
            tool_calls = msg.additional_kwargs.get("tool_calls")

            if tool_calls is None:
                tool_calls = msg.tool_calls

            if tool_calls is not None:
                for tool_call in tool_calls:
                    tool_call_dict = dict(tool_call)
                    tool_output = get_tool_call_results(
                        messages, tool_call_dict.get("id"))

                    # Extract tool name and arguments - handle both OpenAI and Claude formats
                    # Claude format: tool_call_dict has "name" and "args" at top level
                    # OpenAI format: tool_call_dict has "function" containing "name" and "arguments"
                    
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

                    # Determine tool type and category
                    tool_type = "unknown"
                    if tool_name.startswith("ask_webmall_"):
                        tool_type = "search"
                    elif tool_name.startswith("add_to_cart_webmall_"):
                        tool_type = "cart"
                    elif tool_name.startswith("view_cart_webmall_"):
                        tool_type = "view_cart"
                    elif tool_name.startswith("checkout_webmall_"):
                        tool_type = "checkout"
                    elif tool_name.startswith("reset_cart_webmall_"):
                        tool_type = "reset_cart"
                    elif tool_name.startswith("get_products_by_urls_webmall_"):
                        tool_type = "get_products_by_urls"

                    # Enhanced URL extraction with metadata
                    url_extraction_result = extract_urls_from_mcp_response(
                        tool_output or "", tool_name)
                    extracted_urls = url_extraction_result["urls"]
                    extraction_metadata = url_extraction_result["metadata"]
                    response_type = url_extraction_result["response_type"]

                    # Add to comprehensive URL tracking
                    all_mcp_urls.update(extracted_urls)

                    # Track cart/checkout URLs separately
                    if tool_type in ["cart", "checkout"]:
                        cart_checkout_urls.update(
                            extracted_urls)  # Legacy compatibility

                    # New: Track cart and checkout URLs separately
                    if tool_type == "cart":
                        cart_only_urls.update(extracted_urls)
                    elif tool_type == "checkout":
                        checkout_only_urls.update(extracted_urls)
                        # Check if checkout was successful - be more specific about error detection
                        tool_output_lower = tool_output.lower() if tool_output else ""
                        has_error_field = "error" in tool_output_lower
                        has_cart_empty_error = "cart is empty" in tool_output_lower
                        has_order_success = "order" in tool_output_lower and (
                            "created" in tool_output_lower or "successful" in tool_output_lower)

                        if response_type != "error" and not has_error_field and not has_cart_empty_error and has_order_success:
                            checkout_successful = True

                    # Create comprehensive tool call entry
                    tool_call_entry = {
                        "tool_call_id": tool_call_dict.get("id"),
                        "tool_name": tool_name,
                        "tool_type": tool_type,
                        "mcp_server": tools_dict.get(tool_name, "Unknown"),
                        "tool_arguments": parsed_args,
                        "tool_output_raw": tool_output,
                        "response_type": response_type,
                        "urls_extracted": list(extracted_urls),
                        "urls_count": len(extracted_urls),
                        "extraction_metadata": extraction_metadata,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Add tool-specific metrics
                    if tool_type == "search" and "results_count" in extraction_metadata:
                        tool_call_entry["search_results_count"] = extraction_metadata["results_count"]
                    elif tool_type == "cart" and "cart_items_count" in extraction_metadata:
                        tool_call_entry["cart_items_count"] = extraction_metadata["cart_items_count"]
                    elif tool_type == "checkout" and "order_id" in extraction_metadata:
                        tool_call_entry["order_id"] = extraction_metadata["order_id"]
                        tool_call_entry["checkout_status"] = extraction_metadata.get(
                            "checkout_status", "unknown")

                    tool_calls_log.append(tool_call_entry)

    return tool_calls_log, all_mcp_urls, cart_checkout_urls, cart_only_urls, checkout_only_urls, checkout_successful


def get_nlweb_mcp_tools_dict(tools):
    """Create mapping from tool names to MCP server names for NLWeb servers."""
    tools_dict = {}
    for tool in tools:
        tool_name = tool.name

        # Map tools to their specific WebMall servers based on the tool name suffix
        if tool_name.endswith('_webmall_1'):
            tools_dict[tool_name] = "WebMall-1 (E-Store Athletes)"
        elif tool_name.endswith('_webmall_2'):
            tools_dict[tool_name] = "WebMall-2 (TechTalk)"
        elif tool_name.endswith('_webmall_3'):
            tools_dict[tool_name] = "WebMall-3 (CamelCases)"
        elif tool_name.endswith('_webmall_4'):
            tools_dict[tool_name] = "WebMall-4 (Hardware Cafe)"
        else:
            # Fallback for any unrecognized NLWeb tools
            tools_dict[tool_name] = "NLWeb MCP Server"

    return tools_dict


def run_benchmark_enhanced(benchmark_path):
    """Enhanced benchmark runner with comprehensive metrics tracking."""
    execution_history = []

    # Initialize performance tracking
    total_execution_time = 0.0
    total_tool_calls = 0
    total_search_tools = 0
    total_cart_tools = 0
    total_checkout_tools = 0
    total_view_cart_tools = 0
    total_reset_cart_tools = 0

    # Token tracking
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Task success tracking
    successful_tasks = 0
    failed_tasks = 0

    # Initialize ServerManager
    server_manager = ServerManager(show_logs=False)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Start servers and wrap execution in try/finally
    try:
        # Start all servers initially
        print("Starting NLWeb MCP servers...")
        if not server_manager.start_all_servers(debug=False):
            raise Exception("Failed to start all servers")
        time.sleep(10)  # Give servers time to initialize

        # Create initial MCP client to reset carts before starting
        async def reset_carts_initial():
            initial_client = None
            try:
                print("Creating initial MCP client to reset carts...")
                initial_client = MultiServerMCPClient(NLWEB_MCP_SERVERS)
                tools_for_reset = await initial_client.get_tools()
                await reset_all_carts(tools_for_reset)
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

                print(f"\n=== MCP TASK {task['id']} ===")
                print("--------------------------------")

                # Create MCP client for this task to maintain cart state
                async def setup_task_client():
                    task_client = None
                    try:
                        task_client = MultiServerMCPClient(NLWEB_MCP_SERVERS)

                        # Reset carts before each task using the task client
                        if tools_for_reset:
                            await reset_all_carts(tools_for_reset)
                            print("Reset carts for new task")
                        return task_client
                    except Exception as e:
                        print(
                            f"Warning: Failed to create task client or reset carts: {e}")
                        # Create a fresh client if the previous attempt failed
                        if task_client is None:
                            try:
                                task_client = MultiServerMCPClient(
                                    NLWEB_MCP_SERVERS)
                                return task_client
                            except Exception as client_error:
                                print(
                                    f"Failed to create task client: {client_error}")
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
                    tools_dict = get_nlweb_mcp_tools_dict(tools)

                    # Use enhanced tool call tracking system
                    tool_calls_log, mcp_urls, cart_checkout_urls, cart_only_urls, checkout_only_urls, checkout_successful = create_enhanced_tool_call_log(
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
                        elif tool_type == "view_cart":
                            total_view_cart_tools += 1
                        elif tool_type == "reset_cart":
                            total_reset_cart_tools += 1

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
                    mcp_urls = set()
                    cart_checkout_urls = set()
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

                # Determine which URLs to use for evaluation based on task type
                if task_category == "Add_To_Cart":
                    # For cart tasks, only evaluate cart operation URLs
                    evaluation_urls = [normalize_url(
                        url) for url in cart_only_urls]
                    evaluation_strategy = "cart_only"
                    print(
                        f"Using cart-only URLs for evaluation: {cart_only_urls}")
                elif task_category in ["Checkout", "FindAndOrder"]:
                    # For checkout tasks, only evaluate checkout operation URLs and require success
                    if checkout_successful:
                        evaluation_urls = [normalize_url(
                            url) for url in checkout_only_urls]
                        evaluation_strategy = "checkout_successful"
                        print(
                            f"Using checkout URLs for evaluation (checkout successful): {checkout_only_urls}")
                    else:
                        evaluation_urls = []  # No credit for failed checkout
                        evaluation_strategy = "checkout_failed"
                        print(
                            f"No credit - checkout failed. Cart URLs were: {cart_only_urls}, Checkout attempted but failed")
                else:
                    # For search tasks, use the final answer
                    evaluation_urls = [normalize_url(url) for url in got]
                    evaluation_strategy = "final_answer"
                    print(f"Using final answer URLs for evaluation: {got}")

                # Normalize expected URLs
                expected_normalized = [normalize_url(
                    url) for url in expected_flat]

                # Calculate metrics using the appropriate URL set
                task_metrics = calculation_results(
                    benchmark_solutions=expected_normalized, model_solution=evaluation_urls)
                print(f"Metrics for Task {task['id']}: {task_metrics}")

                # correct answer urls
                correct_model_answers = [
                    url for url in expected_flat if normalize_url(url) in evaluation_urls]

                # additional urls that are not in the correct answers
                additional_urls = [
                    url for url in evaluation_urls if url not in expected_normalized]

                # urls that are in the correct answers but not in the model response
                missing_urls = [
                    url for url in expected_normalized if url not in evaluation_urls]

                # Compare MCP URLs with ground truth
                mcp_urls_normalized = [normalize_url(url) for url in mcp_urls]
                mcp_correct_retrieved = [
                    url for url in mcp_urls_normalized if url in expected_normalized]
                mcp_additional_retrieved = [
                    url for url in mcp_urls_normalized if url not in expected_normalized]

                # Calculate MCP metrics
                mcp_metrics = calculation_results(
                    benchmark_solutions=expected_normalized, model_solution=mcp_urls_normalized)

                # Enhanced history entry with comprehensive metrics
                history_entry = {
                    "task_id": task["id"],
                    "task_category": task_category,
                    "task": user_task,
                    "task_completion_rate": task_metrics["task_completion_rate"],
                    "precision": task_metrics["avg_precision"],
                    "recall": task_metrics["avg_recall"],
                    "f1_score": task_metrics["f1_score"],
                    "raw_response": final_answer,
                    "parsed_response": evaluation_urls,
                    "correct_model_answers": correct_model_answers,
                    "additional_urls": additional_urls,
                    "missing_urls": missing_urls,
                    "metrics": task_metrics,
                    "mcp_correct_retrieved": mcp_correct_retrieved,
                    "mcp_additional_retrieved": mcp_additional_retrieved,
                    "mcp_metrics": mcp_metrics,
                    "tool_calls": tool_calls_log,
                    "response": str(response),
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "execution_time_seconds": task_execution_time,
                    "total_tool_calls": len(tool_calls_log),
                    "search_tool_calls": len([t for t in tool_calls_log if t.get("tool_type") == "search"]),
                    "cart_tool_calls": len([t for t in tool_calls_log if t.get("tool_type") == "cart"]),
                    "checkout_tool_calls": len([t for t in tool_calls_log if t.get("tool_type") == "checkout"]),
                    "view_cart_tool_calls": len([t for t in tool_calls_log if t.get("tool_type") == "view_cart"]),
                    "urls_from_cart_checkout": list(cart_checkout_urls),
                    "urls_from_cart_only": list(cart_only_urls),
                    "urls_from_checkout_only": list(checkout_only_urls),
                    "checkout_successful": checkout_successful,
                    "urls_from_all_mcp": list(mcp_urls),
                    "expected_urls": expected_flat,
                    "evaluation_strategy": evaluation_strategy
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
                            print(
                                f"No cleanup method available for task client, skipping cleanup")
                    except Exception as cleanup_error:
                        print(
                            f"Warning: Failed to close task client: {cleanup_error}")

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
            "version": "nlweb_mcp_enhanced",
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
            "checkout_tools": total_checkout_tools,
            "view_cart_tools": total_view_cart_tools,
            "reset_cart_tools": total_reset_cart_tools
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
    print("ENHANCED NLWeb MCP BENCHMARK RESULTS")
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
    print(f"  - View Cart Tools: {total_view_cart_tools}")
    print(f"  - Reset Cart Tools: {total_reset_cart_tools}")
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


def run_benchmark(benchmark_path):
    """Legacy function for backward compatibility - calls enhanced version."""
    return run_benchmark_enhanced(benchmark_path)


def generate_csv_metrics(enhanced_summary):
    """Generate CSV file with task metrics for easy analysis."""
    if not enhanced_summary.get("results"):
        print("No results to export to CSV")
        return

    current_timestamp = enhanced_summary["benchmark_metadata"]["timestamp"]
    csv_filename = f"nlweb_mcp_enhanced_metrics_{current_timestamp}.csv"

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
        }
        csv_data.append(csv_row)

    # Write CSV file
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["task_category", "task_id", "task_completion_rate", "precision",
                          "recall", "f1_score", "prompt_tokens", "completion_tokens"]
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
        enhanced_summary = run_benchmark_enhanced(BENCHMARK_JSON_PATH)
    except Exception as e:
        print(f"Benchmark run failed with error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")

        # Even if the benchmark failed, try to create minimal summary
        enhanced_summary = {
            "benchmark_metadata": {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "version": "nlweb_mcp_enhanced",
                "error": str(e),
                "traceback": traceback.format_exc()
            },
            "results": []
        }
    finally:
        # Always save results, even if there were errors
        current_timestamp = enhanced_summary.get("benchmark_metadata", {}).get(
            "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_file = f"nlweb_mcp_enhanced_results_{current_timestamp}.json"

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
