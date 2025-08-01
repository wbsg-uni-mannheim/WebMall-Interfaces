# Agent querying WebAPIs via MCP

A MCP (Model Context Protocol) implementation that combines semantic search capabilities with full e-commerce API functionality including cart management, product search, and checkout operations.

## Overview

The API MCP interface extends the NLWeb semantic search with additional e-commerce features:

-   **🔍 Semantic Search**: Uses NLWeb's embedding-based product search
-   **🛒 Cart Management**: Thread-safe cart operations with session persistence
-   **💳 Checkout Process**: Complete e-commerce workflow support
-   **🏪 Multi-Shop Support**: 4 dedicated hybrid servers for each WebMall shop
-   **🔧 HTTP/SSE Transport**: RESTful API access via Server-Sent Events

## Architecture

### Hybrid Server Design

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LangGraph     │───▶│  Hybrid MCP      │───▶│   NLWeb Search  │
│   Agent         │    │  Server A/B/C/D  │    │   Engine        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Cart Management │    │  Elasticsearch  │
                       │  Session Store   │    │  Indices        │
                       └──────────────────┘    └─────────────────┘
```

### Server Configuration

| Server       | Shop Name        | Shop ID   | Port | Elasticsearch Index |
| ------------ | ---------------- | --------- | ---- | ------------------- |
| **Hybrid A** | E-Store Athletes | webmall_1 | 8060 | webmall_1_nlweb     |
| **Hybrid B** | TechTalk         | webmall_2 | 8061 | webmall_2_nlweb     |
| **Hybrid C** | CamelCases       | webmall_3 | 8062 | webmall_3_nlweb     |
| **Hybrid D** | Hardware Cafe    | webmall_4 | 8063 | webmall_4_nlweb     |

## Prerequisites

### Required Services

1. **Elasticsearch 8.x** with NLWeb indices populated
2. **OpenAI API Key** for embeddings
3. **NLWeb Data**: Product data must be indexed first

## Setup and Usage

### 1. Data Preparation

First, ensure NLWeb indices are populated with product data:

```bash
# Navigate to nlweb_mcp directory
cd ../nlweb_mcp

# Ingest product data from all WebMall shops
python ingest_data.py --shop all --force-recreate

# Verify data ingestion
python ingest_data.py --check-only
```

### 2. Run Benchmark

Execute the API MCP benchmark:

```bash
cd src/
python benchmark_api_mcp.py
```
