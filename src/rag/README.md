# RAG (Retrieval-Augmented Generation) Interface

A comprehensive RAG system for e-commerce search using web-crawled product data and LangGraph agents.

## Overview

The RAG interface uses web-crawled product data to power LLM-based e-commerce search with intelligent cart management and checkout capabilities.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Content   │───▶│  Crawl & Index   │───▶│  Elasticsearch  │
│   (RSS/Sitemap) │    │  - Embeddings    │    │  Index          │
│                 │    │  - Summaries     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐              │
│   User Query    │───▶│  LangGraph Agent │◀─────────────┘
│                 │    │  - RAG Tools     │
│                 │    │  - Cart Tools    │
└─────────────────┘    └──────────────────┘
```

## Features

### Web Crawling & Indexing
- **Multiple data sources**: RSS feeds, sitemaps, direct URLs, link discovery
- **Content extraction**: Using unstructured library for clean text extraction
- **Smart filtering**: Only processes `/product/` URLs, skips cart/checkout pages
- **Parallel processing**: Configurable concurrency limits
- **Deduplication**: Prevents duplicate URL processing

### Search Capabilities  
- **Semantic search**: OpenAI embeddings with cosine similarity
- **LLM-powered summaries**: AI-generated product summaries for better matching

### LLM Integration
- **LangGraph agents**: Query processing and tool use
- **Cart functionality**: Add to cart, view cart, checkout operations
- **Multi-model support**: OpenAI GPT-4, Anthropic Claude

## Prerequisites

1. **Elasticsearch 8.x** running on `http://localhost:9200`
2. **OpenAI API Key** for embeddings and LLM calls  
3. **Python Dependencies**:
   ```bash
   pip install elasticsearch openai langchain langgraph crawl4ai unstructured beautifulsoup4 python-dotenv numpy
   ```

## Quick Start

1. **Crawl and index web content:**
   ```bash
   cd src/rag
   # Using RSS feeds (fastest for all WebMall shops)
   python crawl_websites.py --rss-list rss_feeds.json --reset-index
   ```

2. **Run RAG benchmark:**
   ```bash
   cd ..
   python benchmark_rag.py
   ```

## Crawling Options

### Data Sources

The RAG interface supports multiple data sources for product content:

#### RSS Feeds (Recommended)
```bash
# Crawl all WebMall shops using RSS feeds
python crawl_websites.py --rss-list rss_feeds.json --reset-index
```

#### Single URL
```bash
python crawl_websites.py https://webmall-1.informatik.uni-mannheim.de/product/canon-eos-r5-ii
```

#### Sitemap Crawling
```bash
python crawl_websites.py https://webmall-1.informatik.uni-mannheim.de/sitemap.xml --sitemap
```

#### Link Discovery
```bash
python crawl_websites.py https://webmall-1.informatik.uni-mannheim.de --discover --max-depth 3
```

### Crawling Parameters

- `--reset-index`: Delete and recreate Elasticsearch index before crawling
- `--max-depth N`: Maximum depth for link discovery (default: 2) 
- `--max-concurrent N`: Concurrent requests limit (default: 5)
- `--allow-domain-hopping`: Follow external domain links during discovery

## RSS Feed Configuration

The `rss_feeds.json` file contains URLs for all WebMall shops:

```json
[
    "https://webmall-1.informatik.uni-mannheim.de/?feed=products",
    "https://webmall-2.informatik.uni-mannheim.de/?feed=products", 
    "https://webmall-3.informatik.uni-mannheim.de/?feed=products",
    "https://webmall-4.informatik.uni-mannheim.de/?feed=products"
]
```

RSS feeds use Google Shopping namespace for product URLs:
```xml
<rss xmlns:g="http://base.google.com/ns/1.0" version="2.0">
  <channel>
    <item>
      <g:link>https://webmall-1.informatik.uni-mannheim.de/product/item1</g:link>
    </item>
  </channel>
</rss>
```


## Benchmark Execution

### Running Benchmarks

```bash
cd src
python benchmark_rag.py
```

### Benchmark Features

- **LangGraph agents**: Advanced reasoning and tool usage
- **Cart operations**: Add to cart, view cart, checkout workflow
- **Multi-model support**: OpenAI GPT-4, Anthropic Claude
- **Comprehensive metrics**: Precision, recall, F1-score, token usage
- **Task categories**: Product search, category browsing, checkout flow

### Results Analysis

Results are saved in timestamped files:
- **JSON**: Detailed execution logs with tool calls
- **CSV**: Structured metrics for analysis  

## Components

### Core Files

- **`crawl_websites.py`**: Web crawler that extracts and indexes product content
- **`elasticsearch_client.py`**: Elasticsearch integration with semantic search
- **`rag_tools.py`**: RAG search tools for LangGraph agents
- **`rag_cart_tools.py`**: Shopping cart and checkout tools
- **`rss_feeds.json`**: Configuration file with WebMall RSS feed URLs

### Benchmark Integration

- **`../benchmark_rag.py`**: Main benchmark script for RAG interface evaluation
- Uses LangGraph agents with RAG and cart tools
- Evaluates search quality, cart operations, and checkout flows
