#!/usr/bin/env python3
"""
MCP Server for WebMall-1 (E-Store Athletes)
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/aaronsteiner/Documents/GitHub/webmall-alternative-interfaces/.env")

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nlweb_mcp.mcp_servers.base_server import run_server_cli
from nlweb_mcp.config import WEBMALL_SHOPS

if __name__ == "__main__":
    shop_config = WEBMALL_SHOPS["webmall_1"]
    run_server_cli(
        shop_id="webmall_1",
        index_name=shop_config["index_name"],
        port=shop_config["mcp_port"]
    )