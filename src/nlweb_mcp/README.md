# NLWeb MCP Implementation

This directory contains a complete implementation of NLWeb using MCP (Model Context Protocol) servers for the WebMall benchmark. Each WebMall shop has its own dedicated MCP server with semantic search capabilities.

## Architecture

- **4 MCP Servers**: One for each WebMall shop (webmall_1, webmall_2, webmall_3, webmall_4)
- **4 Elasticsearch Indices**: `webmall_1_nlweb`, `webmall_2_nlweb`, `webmall_3_nlweb`, `webmall_4_nlweb`
- **Semantic Search**: Using OpenAI text-embedding-3-small embeddings with cosine similarity
- **Schema.org Compliance**: All products stored and returned in schema.org JSON-LD format

## Components

### Core Components

- `config.py` - Configuration settings for all components
- `elasticsearch_client.py` - Elasticsearch integration with semantic search
- `embedding_service.py` - OpenAI embedding generation and batch processing
- `woocommerce_client.py` - WooCommerce API integration for data extraction
- `schema_generator.py` - Schema.org Product JSON-LD generation
- `search_engine.py` - Semantic search engine with NLWeb-compatible responses
- `data_ingestion.py` - Complete data ingestion pipeline

### MCP Servers

- `mcp_servers/base_server.py` - Base MCP server implementation
- `mcp_servers/webmall_1_server.py` - MCP server for E-Store Athletes
- `mcp_servers/webmall_2_server.py` - MCP server for TechTalk  
- `mcp_servers/webmall_3_server.py` - MCP server for CamelCases
- `mcp_servers/webmall_4_server.py` - MCP server for Hardware Cafe

### Utilities

- `ingest_data.py` - Data ingestion script
- `start_all_servers.py` - Server management script

## Setup

### Prerequisites

Before running the benchmark, you **must** complete data indexing:

1. **Elasticsearch 8.x**: Must be running on `http://localhost:9200`
   ```bash
   # Verify Elasticsearch is running
   curl http://localhost:9200
   ```

2. **OpenAI API Key**: Required for embedding generation
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Python Dependencies**: Install required packages
   ```bash
   pip install elasticsearch openai fastmcp python-dotenv requests beautifulsoup4
   ```

4. **WooCommerce API Access**: Consumer keys and secrets for each WebMall shop (optional)
   - Public endpoints will be used if credentials are not provided
   - Authenticated access may provide more complete product data

### ⚠️ Important: Data Indexing Required

**Before starting servers or running benchmarks**, you must index product data from all WebMall shops. This creates the Elasticsearch indices that the MCP servers search against.


## Quick Start

### Step 1: Data Ingestion (Required First)

**This step must be completed before starting servers or running benchmarks.**

Ingest product data from all WebMall shops to create Elasticsearch indices:

```bash
cd src/nlweb_mcp
python ingest_data.py --shop all --force-recreate
```

**Verify ingestion completed successfully:**
```bash
python ingest_data.py --check-only
```

You should see output confirming all 4 indices were created with product counts.

### Step 2: Run Benchmark  

Execute the NLWeb MCP benchmark:

```bash
cd ..
python benchmark_nlweb_mcp.py
```
