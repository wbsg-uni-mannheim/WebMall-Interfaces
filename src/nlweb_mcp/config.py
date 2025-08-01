import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv("../../.env")

# Elasticsearch configuration
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
INDEX_PREFIX = "webmall"
INDEX_SUFFIX = "_nlweb"

# WebMall shop configurations
WEBMALL_SHOPS = {
    "webmall_1": {
        "url": os.getenv("WOO_STORE_URL_1", "https://webmall-1.informatik.uni-mannheim.de"),
        "index_name": f"{INDEX_PREFIX}_1{INDEX_SUFFIX}",
        "mcp_port": 8001,
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_1", ""),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_1", "")
    },
    "webmall_2": {
        "url": os.getenv("WOO_STORE_URL_2", "https://webmall-2.informatik.uni-mannheim.de"),
        "index_name": f"{INDEX_PREFIX}_2{INDEX_SUFFIX}",
        "mcp_port": 8002,
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_2", ""),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_2", "")
    },
    "webmall_3": {
        "url": os.getenv("WOO_STORE_URL_3", "https://webmall-3.informatik.uni-mannheim.de"),
        "index_name": f"{INDEX_PREFIX}_3{INDEX_SUFFIX}",
        "mcp_port": 8003,
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_3", ""),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_3", "")
    },
    "webmall_4": {
        "url": os.getenv("WOO_STORE_URL_4", "https://webmall-4.informatik.uni-mannheim.de"),
        "index_name": f"{INDEX_PREFIX}_4{INDEX_SUFFIX}",
        "mcp_port": 8004,
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_4", ""),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_4", "")
    }
}

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_CHOICE", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536

# Search configuration
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# WooCommerce API configuration
WOOCOMMERCE_API_PATH = "/wp-json/wc/v3"
WOOCOMMERCE_TIMEOUT = 30