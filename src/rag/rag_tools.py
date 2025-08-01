"""
RAG Tools for flexible product search using LangGraph.
These tools can be used by an agent to dynamically search for products.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI
from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field
from elasticsearch_client import ElasticsearchRAGClient
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
es_client = ElasticsearchRAGClient()

# Cache for embeddings to avoid regenerating them
embedding_cache: Dict[str, Tuple[List[float], datetime]] = {}
CACHE_EXPIRY_MINUTES = 30


async def get_embedding(text: str) -> Tuple[List[float], int]:
    """Get embedding vector from OpenAI with caching."""
    # Check cache
    if text in embedding_cache:
        embedding, timestamp = embedding_cache[text]
        if (datetime.now() - timestamp).total_seconds() < CACHE_EXPIRY_MINUTES * 60:
            return embedding, 0  # No new tokens used
    
    try:
        response = await async_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        tokens_used = response.usage.total_tokens
        embedding = response.data[0].embedding
        
        # Cache the embedding
        embedding_cache[text] = (embedding, datetime.now())
        
        # Clean old cache entries
        current_time = datetime.now()
        expired_keys = [
            k for k, (_, ts) in embedding_cache.items()
            if (current_time - ts).total_seconds() > CACHE_EXPIRY_MINUTES * 60
        ]
        for key in expired_keys:
            del embedding_cache[key]
        
        return embedding, tokens_used
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536, 0


class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(description="The search query to find products")
    num_results: int = Field(default=10, description="Number of results to return")


class SearchTool:
    """Tool for executing semantic search queries."""
    
    name = "search_products"
    description = """Search for products using semantic search. 
    Use this to find products based on descriptions, features, or requirements.
    Returns a list of products with their URLs, titles, and content."""
    
    async def __call__(self, query: str, num_results: int = 10) -> Dict:
        """Execute semantic search."""
        try:
            # Get embedding
            embedding, tokens_used = await get_embedding(query)
            
            # Search
            results = await es_client.semantic_search(embedding, num_results)
            
            # Format results
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "url": doc["url"],
                    "title": doc["title"],
                    "content": doc["content"][:500],  # Truncate for readability
                    "similarity": doc.get("similarity", 0),
                    "chunk_number": doc.get("chunk_number", 0)
                })
            
            return {
                "success": True,
                "query": query,
                "num_results": len(formatted_results),
                "results": formatted_results,
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }


class ExpandSearchInput(BaseModel):
    """Input schema for expand search tool."""
    query: str = Field(description="The original search query to expand")
    offset: int = Field(description="Number of results to skip")
    num_results: int = Field(default=10, description="Number of additional results to return")


class ExpandSearchTool:
    """Tool for getting more results from a previous search."""
    
    name = "expand_search"
    description = """Get additional results for a search query you've already performed.
    Use this when you need more results beyond what the initial search returned.
    Specify the offset to skip already retrieved results."""
    
    async def __call__(self, query: str, offset: int, num_results: int = 10) -> Dict:
        """Get more results for an existing query."""
        try:
            # Get embedding (hopefully from cache)
            embedding, tokens_used = await get_embedding(query)
            
            # Search with higher limit
            all_results = await es_client.semantic_search(
                embedding, 
                offset + num_results
            )
            
            # Return only the new results
            new_results = all_results[offset:offset + num_results] if offset < len(all_results) else []
            
            # Format results
            formatted_results = []
            for doc in new_results:
                formatted_results.append({
                    "url": doc["url"],
                    "title": doc["title"],
                    "content": doc["content"][:500],
                    "similarity": doc.get("similarity", 0),
                    "chunk_number": doc.get("chunk_number", 0)
                })
            
            return {
                "success": True,
                "query": query,
                "offset": offset,
                "num_results": len(formatted_results),
                "results": formatted_results,
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }


class AnalyzeDocumentInput(BaseModel):
    """Input schema for document analysis tool."""
    document: Dict = Field(description="The document to analyze (should have url, title, content)")
    criteria: str = Field(description="The criteria to check against (user's requirements)")


class AnalyzeDocumentTool:
    """Tool for analyzing document relevance using a small model."""
    
    name = "analyze_document"
    description = """Analyze a single document to determine if it matches the user's criteria.
    Uses a small, fast model to pre-filter results before sending to the main agent.
    Returns relevance score and reasoning."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
    
    async def __call__(self, document: Dict, criteria: str) -> Dict:
        """Analyze document relevance."""
        try:
            prompt = f"""Analyze if this product matches the user's criteria.

User criteria: {criteria}

Product information:
Title: {document.get('title', 'N/A')}
URL: {document.get('url', 'N/A')}
Content: {document.get('content', 'N/A')[:1000]}

Determine:
1. Does this product match the criteria? (yes/no/partial)
2. What specific features match or don't match?
3. Overall relevance score (0-10)

Respond in JSON format:
{{
    "matches": "yes/no/partial",
    "matching_features": ["feature1", "feature2"],
    "missing_features": ["feature1", "feature2"],
    "relevance_score": 0-10,
    "reasoning": "brief explanation"
}}"""

            response = await async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a product analysis assistant. Be precise and objective."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "success": True,
                "document_url": document.get("url"),
                "document_title": document.get("title"),
                "analysis": result,
                "tokens_used": response.usage.total_tokens,
                "model_name": self.model,
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "document_url": document.get("url", "unknown"),
                "analysis": {
                    "matches": "unknown",
                    "relevance_score": 0,
                    "reasoning": f"Analysis failed: {str(e)}"
                }
            }


# Create tool instances for LangChain
search_tool = Tool(
    name="search_products",
    description=SearchTool.description,
    func=None,  # Will be set to async function
    coroutine=SearchTool().__call__,
    args_schema=SearchInput
)

expand_search_tool = Tool(
    name="expand_search",
    description=ExpandSearchTool.description,
    func=None,
    coroutine=ExpandSearchTool().__call__,
    args_schema=ExpandSearchInput
)

analyze_document_tool = Tool(
    name="analyze_document",
    description=AnalyzeDocumentTool.description,
    func=None,
    coroutine=AnalyzeDocumentTool().__call__,
    args_schema=AnalyzeDocumentInput
)


# Utility function to create all tools
def get_rag_tools(analyze_model: str = "gpt-3.5-turbo") -> List[Tool]:
    """Get all RAG tools configured and ready to use."""
    analyzer = AnalyzeDocumentTool(model=analyze_model)
    
    # Update analyze tool with specific model
    analyze_tool = Tool(
        name="analyze_document",
        description=AnalyzeDocumentTool.description,
        func=None,
        coroutine=analyzer.__call__,
        args_schema=AnalyzeDocumentInput
    )
    
    return [search_tool, expand_search_tool, analyze_tool]