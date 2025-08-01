import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv


async def get_mcp_functions(mcp_name: str, mcp_url: str, mcp_transport: str = "sse") -> list[str]:
    """
    Connects to a single MCP server and returns a list of its available function names.

    Args:
        mcp_name: A descriptive name for the MCP server (e.g., "E-Store Athletes").
        mcp_url: The full URL of the MCP server's endpoint (e.g., "http://localhost:8050/sse").
        mcp_transport: The transport protocol, defaults to "sse".

    Returns:
        A list of function names (strings) offered by the MCP server.
    """
    client = MultiServerMCPClient({
        mcp_name: {
            "url": mcp_url,
            "transport": mcp_transport,
        }
    })
    try:
        tools = await client.get_tools()
        function_names = [tool.name for tool in tools]
        return function_names
    except Exception as e:
        print(f"Error connecting to {mcp_name} at {mcp_url}: {e}")
        return []


# Example of how to use this function
async def get_all_mcp_functions_dict():
    """
    Example usage of get_mcp_functions to query each MCP server
    and print its available functions.
    """
    load_dotenv()

    servers = {
        "E-Store Athletes": f"http://localhost:{os.getenv('PORT', '8050')}/sse",
        "TechTalk": f"http://localhost:{os.getenv('PORT_PRODUCT_CATALOG', '8051')}/sse",
        "CamelCases": f"http://localhost:{os.getenv('PORT_STORE_INVENTORY', '8052')}/sse",
        "Hardware Cafe": f"http://localhost:{os.getenv('PORT_ECOMMERCE_DATA', '8053')}/sse",
    }

    mcp_functions_dict = {}

    for name, url in servers.items():
        print(f"Querying functions for: {name}")
        functions = await get_mcp_functions(mcp_name=name, mcp_url=url)
        if functions:
            for func_name in functions:
                mcp_functions_dict[func_name] = name
        else:
            print("  - No functions found or server unavailable.")
        print("-" * 20)

    return mcp_functions_dict
