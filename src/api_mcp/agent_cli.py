import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
import re

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "E-Store Athletes": {
                "url": f"http://localhost:{os.getenv("PORT", "8050")}/sse",
                "transport": os.getenv("TRANSPORT", "sse"),
            },
            "TechTalk": {
                "url": f"http://localhost:{os.getenv("PORT_PRODUCT_CATALOG", "8051")}/sse",
                "transport": os.getenv("TRANSPORT", "sse"),
            },
            "CamelCases": {
                "url": f"http://localhost:{os.getenv("PORT_STORE_INVENTORY", "8052")}/sse",
                "transport": os.getenv("TRANSPORT", "sse"),
            },
            "Hardware Cafe": {
                "url": f"http://localhost:{os.getenv("PORT_ECOMMERCE_DATA", "8053")}/sse",
                "transport": os.getenv("TRANSPORT", "sse"),
            },
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(
        f"openai:{os.getenv('BENCHMARK_MODEL', 'gpt-4.1')}" if os.getenv('BENCHMARK_MODEL', 'gpt-4.1').startswith('gpt') else f"anthropic:{os.getenv('BENCHMARK_MODEL', 'gpt-4.1')}",
        tools
    )
    while True:
        user_input = input("> ")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        response = await agent.ainvoke(
            {"messages": [
                {"role": "system",
                    "content": """You are an AI assistant that can access information from multiple e-commerce stores. 
                    Solve the task below using these four webshops:\\n\\nE-Store Athletes\\nTechTalk\\nCamelCases\\nHardware Cafe\\n\\n
                    Before you give the answer, make sure that you have searched all the webshops.
                    When asked about products, inventory, or categories, consider searching across all available tools (woocommerce, product-catalog, store-inventory, ecommerce-data) to provide a comprehensive answer. 
                    
                    Reply with a JSON array containing the EXACT URLs. Format: [\"url1\", \"url2\", ...] or [\"Done\"] if no products found.
                    
                    """},
                {"role": "user", "content": user_input}
            ]}
        )
        pretty_print_agent_response(response)


def pretty_print_agent_response(agent_response):
    import json
    import re
    console = Console()
    messages = agent_response.get("messages", [])

    # Build a mapping from tool_call_id to (tool name, args)
    tool_call_map = {}
    for msg in messages:
        # AIMessage: has tool_calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                tool_call_map[getattr(call, "id", None)] = {
                    "name": getattr(call, "name", None),
                    "args": getattr(call, "args", None),
                }

    # Print tool results as tables, with function name and args
    for msg in messages:
        if msg.__class__.__name__ == "ToolMessage":
            tool_call_id = getattr(msg, "tool_call_id", None)
            tool_info = tool_call_map.get(tool_call_id)
            if tool_info:
                console.print(
                    f"[bold yellow]Tool Call:[/bold yellow] [green]{tool_info['name']}[/green] [cyan]{tool_info['args']}[/cyan]")
            else:
                console.print(
                    f"[bold yellow]Tool Call ID:[/bold yellow] {tool_call_id}")
            # Show the raw returned content
            try:
                data = json.loads(msg.content)
                console.print("[bold]MCP Response:[/bold]")
                console.print_json(json.dumps(data, indent=2))
                # Optionally, still print as table if products present
                if "products" in data:
                    table = Table(title="Products")
                    table.add_column("ID")
                    table.add_column("Name")
                    table.add_column("Price")
                    for p in data["products"]:
                        table.add_row(str(p["id"]), p["name"], str(p["price"]))
                    console.print(table)
            except Exception:
                # If not JSON, just print the content
                console.print(msg.content)
    # Print the final AI answer
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            content = msg.content
            print("Raw content: ", content)
            # Extract and print URLs as plain text
            urls = re.findall(r"https?://\S+", content)
            for url in urls:
                console.print(url)
            # Remove URLs from content before printing as Markdown
            content_no_urls = re.sub(r"https?://\S+", "", content)
            if content_no_urls.strip():
                console.print(Markdown(content_no_urls))
            break


if __name__ == "__main__":
    asyncio.run(main())
