# WebMall-Interfaces

A benchmark system for evaluating e-commerce search interfaces using the WebMall benchmark. This project implements three different approaches for querying e-commerce product data across four WebMall shops.

## Overview

WebMall-Interfaces provides three distinct interface approaches for benchmarking e-commerce search capabilities:

### 🔍 [NLWeb MCP](nlweb_mcp/) - Semantic Search Interface

-   **Purpose**: Semantic search using OpenAI embeddings with MCP (Model Context Protocol) servers
-   **Architecture**: 4 dedicated MCP servers, one per WebMall shop
-   **Search Method**: Embedding-based cosine similarity search
-   **Data Format**: Schema.org compliant JSON-LD product data

### 🤖 [RAG](rag/) - Retrieval-Augmented Generation

-   **Purpose**: LLM-powered search with retrieved context from web-crawled product data
-   **Architecture**: Elasticsearch with web-crawled content + LangGraph agents
-   **Search Method**: Semantic search with AI-generated summaries
-   **Data Format**: Chunked web content with embeddings and summaries

### 🔧 [API MCP](api_mcp/) - MCP Interface

-   **Purpose**: Combines semantic search with e-commerce API functionality
-   **Architecture**: Hybrid MCP servers with cart management and product search
-   **Search Method**: Uses NLWeb semantic search 
-   **Data Format**: Heterogeneous JSON + cart/checkout capabilities

## Architecture Comparison

| Feature             | NLWeb MCP            | RAG              | API MCP         |
| ------------------- | -------------------- | ---------------- | --------------- |
| **Search Type**     | Semantic only        | Semantic only    | Semantic + API  |
| **Data Source**     | WooCommerce API      | Web crawling     | WooCommerce API |
| **LLM Integration** | MCP + LangGraph      | LangGraph agents | MCP + LangGraph |
| **Cart Support**    | Built-in             | Via tools        | Built-in        |
| **Real-time**       | Yes                  | Yes              | Yes             |


## Prerequisites

### Required Services

1. **Elasticsearch 8.x** running on `http://localhost:9200`
2. **OpenAI API Key** for embeddings and LLM calls
3. **Python 3.8+** with required dependencies

### Environment Setup

1. **Copy environment template:**

    ```bash
    cp .env.example .env
    ```

2. **Configure environment variables:**

    ```env
    # Required
    OPENAI_API_KEY=your_openai_api_key_here
    ELASTICSEARCH_HOST=http://localhost:9200

    # Model Selection
    EMBEDDING_MODEL_CHOICE=text-embedding-3-small
    BENCHMARK_MODEL=gpt-4.1

    # Optional: WooCommerce API credentials (for authenticated access)
    WOO_CONSUMER_KEY_1=your_webmall_1_consumer_key
    WOO_CONSUMER_SECRET_1=your_webmall_1_consumer_secret
    # ... (repeat for webmall_2, webmall_3, webmall_4)
    ```

3. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt  
    # Or install manually:
    pip install elasticsearch openai langchain langgraph fastmcp numpy beautifulsoup4 python-dotenv crawl4ai unstructured
    ```

## Quick Start Guide

### Option 1: NLWeb MCP Interface 

1. **Index product data:**

    ```bash
    cd src/nlweb_mcp
    python ingest_data.py --shop all --force-recreate
    ```

2. **Run benchmark:**
    ```bash
    cd ..
    python benchmark_nlweb_mcp.py
    ```

### Option 2: RAG Interface 

1. **Crawl and index web content:**

    ```bash
    cd src/rag
    python crawl_websites.py --rss-list rss_feeds.json --reset-index
    ```

2. **Run benchmark:**
    ```bash
    cd ..
    python benchmark_rag.py
    ```

### Option 3: API MCP Interface 

1. **Ensure NLWeb data is indexed** (see Option 1, step 1)

2. **Run benchmark:**
    ```bash
    cd ..
    python benchmark_api_mcp.py
    ```

## Benchmarking Workflow

### Standard Benchmark Process

1. **Data Preparation**: Index/crawl product data from WebMall shops
2. **Benchmark Execution**: Run benchmark scripts with predefined tasks
3. **Results Analysis**: Review metrics and performance data

### Benchmark Tasks

The benchmark evaluates performance across multiple categories:

-   **Product Search**: Finding specific products by name/description
-   **Category Browsing**: Discovering products within categories
-   **Checkout Flow**: Adding products to cart and completing purchases (API MCP only)

### Metrics Collected

-   **Precision/Recall/F1**: URL retrieval accuracy
-   **Response Time**: Query execution speed
-   **Token Usage**: LLM API consumption
-   **Success Rate**: Task completion percentage


## Results and Analysis

Benchmark results are saved with timestamps in the following formats:

-   **JSON files**: Detailed execution logs with tool calls and responses
-   **CSV files**: Structured metrics for analysis
-   **Logs**: Server and ingestion process logs

## Troubleshooting

### Common Issues

1. **Elasticsearch Connection Failed**

    - Ensure Elasticsearch is running: `curl http://localhost:9200`
    - Check `ELASTICSEARCH_HOST` environment variable

2. **OpenAI API Errors**

    - Verify `OPENAI_API_KEY` is set correctly
    - Check API quota and billing status

3. **Empty Search Results**

    - Ensure data ingestion completed successfully
    - Check Elasticsearch indices: `curl http://localhost:9200/_cat/indices`

4. **MCP Server Connection Issues**
    - Verify servers are running on expected ports
    - Check for port conflicts with `netstat -an | grep LISTEN`
