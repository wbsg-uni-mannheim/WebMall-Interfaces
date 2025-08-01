#!/usr/bin/env python3
"""
MCP Server for WebMall-3 (CamelCases)
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nlweb_mcp.mcp_servers.base_server import run_server_cli
from nlweb_mcp.config import WEBMALL_SHOPS

if __name__ == "__main__":
    shop_config = WEBMALL_SHOPS["webmall_3"]
    run_server_cli(
        shop_id="webmall_3",
        index_name=shop_config["index_name"],
        port=shop_config["mcp_port"]
    )