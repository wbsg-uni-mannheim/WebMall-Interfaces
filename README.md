# MCP vs RAG vs NLWeb vs HTML: A Comparison of the Effectiveness and Efficiency of Different Agent Interfaces to the Web

🌐 [WebMall-Interfaces Website](https://wbsg-uni-mannheim.github.io/WebMall-Interfaces/)

## Abstract

This repository contains the implementation and evaluation code for WebMall-Interfaces, a comprehensive comparison of different agent interfaces for web-based tasks. LLM agents use different architectures and interfaces to interact with the World Wide Web: some rely on traditional web browsers to navigate HTML pages, others retrieve web content by querying search engines, while others interact with site-specific Web APIs via the Model Context Protocol (MCP) or standardized interfaces like NLWeb.

This study presents the first experimental comparison of these four architectures using the same set of 91 e-commerce tasks across four simulated shops of the [WebMall benchmark](https://github.com/wbsg-uni-mannheim/WebMall). We compare the effectiveness (success rate, F1) and efficiency (runtime, token usage) of RAG agents, MCP agents, NLWeb agents, and browser-based HTML agents. The experiments demonstrate that specialized agents can deliver equal and in certain cases superior performance compared to browser-based agents, while incurring 5-10 times lower token costs.

For detailed information about the benchmark design, interface specifications, and evaluation results, please refer to our [website](https://wbsg-uni-mannheim.github.io/WebMall-Interfaces/).

## Interface Approaches

This repository implements three of the four interface approaches evaluated in our research:

### Browser-based HTML Interface (Reference)

The first interface approach - browser-based agents that navigate HTML pages using accessibility trees and screenshots - is implemented in the original [WebMall benchmark](https://wbsg-uni-mannheim.github.io/WebMall/). We use their best-performing configuration (AX+MEM) as a baseline for comparison with our specialized interfaces.

### RAG - Retrieval-Augmented Generation Interface

An LLM-powered search system that combines web-crawled product data with LangGraph agents. This approach enables intelligent query processing with cart and checkout functionality, using semantic search.

### API MCP - Hybrid MCP Interface

A comprehensive hybrid system that combines semantic search capabilities with full e-commerce API functionality. This approach features heterogeneous response formats for testing format adaptability, built-in cart management with session persistence, and complete checkout workflows.

### NLWeb MCP - Natural Language Web Interface

An implementation of Microsoft's [NLWeb (Natural Language for Web)](https://github.com/nlweb-ai/NLWeb) proposal, which defines a standardized natural language interface for websites. This approach requires website vendors to implement and host an "ask" endpoint that accepts natural language queries and returns structured responses in schema.org format. Our implementation combines NLWeb's ask functionality with MCP servers to enable comprehensive e-commerce operations including cart management and checkout processes.

## Setting up WebMall-Interfaces

### Environment Requirements

-   WebMall-Interfaces requires Python 3.8 or higher
-   Elasticsearch 8.x running on `http://localhost:9200`
-   OpenAI API key for embedding generation and LLM calls
-   Optional: Anthropic API key for Claude model support

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Setup Environment Variables

WebMall-Interfaces expects a `.env` file in the root directory containing API keys and configuration. Make a copy of `.env.example` and rename it to `.env`, then configure the following variables:

## Running the WebMall-Interfaces Benchmark

### NLWeb MCP Interface

```bash
# Data ingestion (required first step)
cd src/nlweb_mcp
python ingest_data.py --shop all --force-recreate

# Run benchmark
cd ..
python benchmark_nlweb_mcp.py
```

### RAG Interface

```bash
# Web crawling and indexing
cd src/rag
python crawl_websites.py --rss-list rss_feeds.json --reset-index

# Run benchmark
cd ..
python benchmark_rag.py
```

### API MCP Interface

```bash
# Ensure NLWeb data is indexed (required dependency)
cd src/nlweb_mcp
python ingest_data.py --shop all --force-recreate

# Run benchmark (servers start automatically)
cd ..
python benchmark_api_mcp.py
```

## Task Categories

The benchmark includes 91 e-commerce tasks organized into multiple categories:

**Basic Tasks (48 tasks)**:

-   Find Specific Product (12 tasks): Locate particular products by name or model number
-   Find Cheapest Offer (10 tasks): Identify lowest-priced options across shops
-   Products Fulfilling Specific Requirements (11 tasks): Find products matching precise technical specifications
-   Add to Cart (7 tasks): Add selected products to shopping cart
-   Checkout (8 tasks): Complete purchase process with payment and shipping information

**Advanced Tasks (43 tasks)**:

-   Cheapest Offer with Specific Requirements (10 tasks): Find affordable products meeting detailed criteria
-   Products Satisfying Vague Requirements (8 tasks): Interpret and fulfill imprecise requirements
-   Cheapest Offer with Vague Requirements (6 tasks): Combine price optimization with fuzzy requirement matching
-   Find Substitutes (6 tasks): Identify alternative products when requested items are unavailable
-   Find Compatible Products (5 tasks): Locate accessories or components compatible with given products
-   End To End (8 tasks): Complete full shopping workflows from search to checkout

For more details, see the [WebMall benchmark website](https://wbsg-uni-mannheim.github.io/WebMall/).

### Evaluation Metrics

-   **Task Completion Rate**: Percentage of tasks completed successfully
-   **Precision/Recall/F1-Score**: URL retrieval accuracy for product search tasks
-   **Response Time**: Query execution latency
-   **Token Efficiency**: LLM API usage
-   **Tool Usage Statistics**: Detailed analysis of agent tool calling patterns

## Repository Structure

```
WebMall-Interfaces/
├── README.md                    # This file
├── .env.example                # Environment template
├── src/                        # Main implementation directory
│   ├── README.md              # Detailed setup and usage guide
│   ├── nlweb_mcp/            # NLWeb MCP interface implementation
│   ├── rag/                  # RAG interface implementation
│   ├── api_mcp/              # API MCP interface implementation
│   ├── benchmark_nlweb_mcp.py # NLWeb MCP benchmark script
│   ├── benchmark_rag.py       # RAG benchmark script
│   ├── benchmark_api_mcp.py   # API MCP benchmark script
│   └── utils.py               # Shared utilities and metrics
├── results/                   # Benchmark results (generated)
└── website/                   # Results visualization and analysis
```

## Results Analysis

Benchmark results are automatically saved in timestamped files with comprehensive metrics:

-   **JSON Format**: Detailed execution logs with tool calls, responses, and intermediate states
-   **CSV Format**: Structured metrics for statistical analysis and visualization

For analysis and visualization, use the provided Jupyter notebooks in the `src/` directory.

## License

This project is part of the WebMall benchmark suite developed at the University of Mannheim. Please refer to the license file for usage terms and conditions.
