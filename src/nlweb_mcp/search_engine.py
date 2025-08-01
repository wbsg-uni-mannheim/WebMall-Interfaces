import logging
import json
from typing import Dict, List, Any, Optional

# Handle both relative and absolute imports
try:
    from .elasticsearch_client import ElasticsearchClient
    from .embedding_service import EmbeddingService
    from .config import DEFAULT_TOP_K, MAX_TOP_K
except ImportError:
    from elasticsearch_client import ElasticsearchClient
    from embedding_service import EmbeddingService
    from config import DEFAULT_TOP_K, MAX_TOP_K

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, elasticsearch_client: ElasticsearchClient, embedding_service: EmbeddingService, index_name: str):
        self.es_client = elasticsearch_client
        self.embedding_service = embedding_service
        self.index_name = index_name

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, include_descriptions: bool = True) -> Dict[str, Any]:
        """Perform semantic search and return results in NLWeb format"""

        # Validate parameters
        if not query or not query.strip():
            return self._empty_response("Empty query provided")

        top_k = min(max(1, top_k), MAX_TOP_K)

        try:
            # Create embedding for the query
            query_embedding = self.embedding_service.create_query_embedding(
                query)

            # Perform semantic search
            search_results = self.es_client.semantic_search(
                index_name=self.index_name,
                query_embedding=query_embedding,
                top_k=top_k
            )

            # Format results in NLWeb-compatible format
            return self._format_nlweb_response(query, search_results, include_descriptions)

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return self._error_response(f"Search failed: {str(e)}")

    def _format_nlweb_response(self, query: str, search_results: List[Dict[str, Any]], include_descriptions: bool = True) -> Dict[str, Any]:
        """Format search results to match NLWeb response structure"""

        results = []
        for result in search_results:
            # Always use _generate_basic_schema_org which now handles existing data intelligently
            schema_org = self._generate_basic_schema_org(
                result, include_descriptions)

            # Format result entry
            result_entry = {
                "url": result.get("url", ""),
                "name": result.get("title", ""),
                "site": self._get_site_name(),
                "siteUrl": self._get_site_name(),
                # Convert to 0-100 scale
                "score": int(result.get("_score", 0) * 100),
                "schema_object": schema_org,
                "ranking_type": "SEMANTIC_SEARCH"
            }

            # Only include description if requested
            # if include_descriptions:
            #    result_entry["description"] = result.get("description", "")

            results.append(result_entry)

        # Build complete NLWeb-style response
        response = {
            "asking_sites": {
                "message": f"Asking {self._get_site_name()}"
            },
            "tool_selection": {
                "selected_tool": "semantic_search",
                "parameters": {
                    "search_query": query,
                    "top_k": len(results)
                },
                "query": query,
                "time_elapsed": "N/A"
            },
            "offers": results,
            "query_id": "",
            # "chatbot_instructions": self._get_chatbot_instructions()
        }

        return response

    def _empty_response(self, message: str) -> Dict[str, Any]:
        """Return empty response with message"""
        return {
            "tool_selection": {
                "selected_tool": "semantic_search",
                "parameters": {
                    "search_query": "",
                    "top_k": 0
                },
                "query": "",
                "time_elapsed": "N/A"
            },
            "results": [],
            "query_id": "",
            "message": message
        }

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Return error response"""
        return {
            "error": error_message,
            "success": False,
            "results": []
        }

    def _get_site_name(self) -> str:
        """Get site name based on index name"""
        if "webmall_1" in self.index_name:
            return "WebMall-1"
        elif "webmall_2" in self.index_name:
            return "WebMall-2"
        elif "webmall_3" in self.index_name:
            return "WebMall-3"
        elif "webmall_4" in self.index_name:
            return "WebMall-4"
        else:
            return "WebMall"

    def _generate_basic_schema_org(self, result: Dict[str, Any], include_descriptions: bool = True) -> Dict[str, Any]:
        """Generate basic schema.org Product data from available fields, preferring existing schema_org data"""

        # First, check if we already have schema_org data
        existing_schema = result.get("schema_org", {})
        if existing_schema and isinstance(existing_schema, dict):
            # We have existing schema.org data, use it as the base
            schema_org = existing_schema.copy()

            # Ensure it has the basic required fields
            if not schema_org.get("@context"):
                schema_org["@context"] = "https://schema.org"
            if not schema_org.get("@type"):
                schema_org["@type"] = "Product"

            # Only modify description handling if requested
            if not include_descriptions and "description" in schema_org:
                del schema_org["description"]
            elif include_descriptions and not schema_org.get("description"):
                # Add description if we want it but don't have it
                description = result.get("description", "")
                if description:
                    schema_org["description"] = description

            return schema_org

        # Fallback: Generate basic schema.org structure from available fields
        title = result.get("title", "")
        price = result.get("price", 0)
        description = result.get(
            "description", "") if include_descriptions else ""
        category = result.get("category", "")
        url = result.get("url", "")

        # Create basic schema.org Product structure
        schema_org = {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": title,
            "category": category,
            "url": url
        }

        # Only add description if requested
        if include_descriptions and description:
            schema_org["description"] = description

        # Add offers if price is available
        if price and str(price) != "0":
            schema_org["offers"] = {
                "@type": "Offer",
                "price": str(price),
                "priceCurrency": "EUR",
                "availability": "https://schema.org/InStock"
            }

        # Add product ID if available
        product_id = result.get("product_id") or result.get("id")
        if product_id:
            schema_org["sku"] = str(product_id)
            schema_org["@id"] = str(product_id)

        return schema_org

    def _get_chatbot_instructions(self) -> str:
        """Get chatbot instructions for formatting results"""
        return """IMPORTANT FORMATTING INSTRUCTION: When presenting these results to the user: 
1. For each item in the 'results' array, format the 'name' field as a hyperlink using the
   'url' field as the link destination. For example, convert: 
     name: 'Open Studio', url: 'https://example.com/event' 
   to: 
     [Open Studio](https://example.com/event)
2. If the item has an 'image' field (which may be in the value of the schema_object field of the item), include the image in your response using markdown image syntax:
     ![Event Name](image_url)
3. Include relevant details like location, date, and description after the link. Every result should be presented with the name as a clickable link, an image if available, and key information about the event.
"""

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific product by ID"""
        try:
            logger.debug(
                f"SearchEngine.get_product_by_id: Searching for product_id='{product_id}' in index='{self.index_name}'")
            product = self.es_client.get_product_by_id(
                self.index_name, product_id)
            if product:
                logger.debug(
                    f"SearchEngine.get_product_by_id: Found product '{product.get('title', 'Unknown')}' for product_id='{product_id}'")
                return self._format_single_product_response(product)
            else:
                logger.warning(
                    f"SearchEngine.get_product_by_id: Product not found for product_id='{product_id}' in index='{self.index_name}'")
            return None
        except Exception as e:
            logger.error(
                f"SearchEngine.get_product_by_id: Failed to get product {product_id} from index '{self.index_name}': {e}")
            return None

    def get_products_by_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Get detailed product information for a list of URLs"""
        try:
            # Use the new elasticsearch client method
            raw_products = self.es_client.get_products_by_urls(
                self.index_name, urls)

            # Format the results
            products = []
            for product_data in raw_products:
                if "error" in product_data:
                    # Keep error entries as-is but add site name
                    product_data["site"] = self._get_site_name()
                    products.append(product_data)
                else:
                    # Format successful results with full details
                    formatted_product = self._format_single_product_response(
                        product_data)
                    products.append(formatted_product)

            return products

        except Exception as e:
            logger.error(f"Failed to get products by URLs: {e}")
            # Return error entries for all URLs
            return [{
                "url": url,
                "error": f"Processing failed: {str(e)}",
                "site": self._get_site_name()
            } for url in urls]

    def _format_single_product_response(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single product for response"""
        # Use the updated method that handles existing schema.org data
        schema_org = self._generate_basic_schema_org(
            product, include_descriptions=True)

        return {
            "url": product.get("url", ""),
            "name": product.get("title", ""),
            "site": self._get_site_name(),
            "description": product.get("description", ""),
            "price": product.get("price", 0),
            "category": product.get("category", ""),
            "schema_object": schema_org
        }

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the search engine"""
        try:
            # Test Elasticsearch connection
            stats = self.es_client.get_index_stats(self.index_name)

            # Test embedding service
            embedding_test = self.embedding_service.test_embedding_service()

            return {
                "status": "healthy" if embedding_test else "degraded",
                "elasticsearch": {
                    "connected": True,
                    "index_exists": "error" not in stats,
                    "document_count": stats.get("document_count", 0)
                },
                "embedding_service": {
                    "available": embedding_test
                },
                "index_name": self.index_name
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "index_name": self.index_name
            }

    def search_server_a_format(self, query: str, per_page: int = 10, page: int = 1, include_descriptions: bool = True) -> str:
        """Search and return results in server_a format (E-Store Athletes)"""
        try:
            # Validate per_page parameter
            per_page = min(max(1, per_page), 100)

            # Calculate how many results we need to fetch to handle pagination
            # Fetch more results and slice them for pagination
            total_needed = page * per_page

            # Create embedding for the query
            query_embedding = self.embedding_service.create_query_embedding(
                query)

            # Perform semantic search - fetch enough results for pagination
            search_results = self.es_client.hybrid_search(
                index_name=self.index_name,
                query=query,
                query_embedding=query_embedding,
                top_k=min(total_needed, 100)  # Limit to 100 max
            )

            # Handle pagination by slicing results
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_results = search_results[start_idx:end_idx]

            # Transform results to server_a format
            weird_products = []
            for result in paginated_results:
                schema_org = result.get("schema_org", {})

                # Extract price information with fallbacks
                price_info = schema_org.get("offers", {})
                if isinstance(price_info, list) and price_info:
                    price_info = price_info[0]

                # Fallback to direct price field if schema_org.offers is empty
                direct_price = result.get("price", 0)
                current_price = price_info.get("price") or price_info.get(
                    "lowPrice") or direct_price or ""

                # Extract product ID with better fallback logic
                product_id = result.get("_id") or result.get(
                    "id") or schema_org.get("@id") or ""

                # Extract SKU with fallbacks
                item_code = schema_org.get("sku") or result.get(
                    "sku") or result.get("product_id") or product_id

                # Extract description with fallbacks
                description = result.get(
                    "description", "") or schema_org.get("description", "")

                # Extract category with fallbacks
                category = schema_org.get(
                    "category") or result.get("category") or ""

                weird_products.append({
                    "ID": item_code,
                    "label": result.get("title") or schema_org.get("name", ""),
                    "shortCode": result.get("url", "").split("/")[-1] if result.get("url") else "",

                    "priceInfo": {
                        "current": str(current_price) if current_price else "",
                        "usual": price_info.get("highPrice", ""),
                        "dealPrice": str(current_price) if current_price else "",
                        "isOnDeal": bool(current_price),
                    },

                    "desc": {
                        "longVersion": (
                            description.strip()[:1024] + "..."
                        ) if len(description) > 1024 and include_descriptions else ("" if not include_descriptions else description),
                        "quickPitch": description.strip()[:200] if description and include_descriptions else "",
                    } if include_descriptions else {
                        "longVersion": "",
                        "quickPitch": "",
                    },

                    "stock": {
                        "itemCode": str(item_code) if item_code else "",
                        "status": "In stock",
                        "leftOverCount": schema_org.get("inventoryLevel", 0),
                    },

                    "labels": {
                        "categories": [category] if category else [],
                        "tags": schema_org.get("keywords", "").split(",") if schema_org.get("keywords") else [],
                    },

                    "snapshots": [schema_org.get("image", "")][:2] if schema_org.get("image") else [],

                    "addresses": {
                        "selfLink": result.get("url", ""),
                        "shareLink": result.get("url", ""),
                    },
                })

            return json.dumps({
                "searchLog": {
                    "askedFor": query,
                    "atPage": page,
                    "pageSize": per_page,
                    "foundCount": len(weird_products),
                },
                "results": weird_products
            }, indent=2)

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return f"Error searching products: {str(e)}"

    def search_server_b_format(self, query: str, limit: int = 5, page_num: int = 1, sort_by_price: str = "none", include_descriptions: bool = True) -> str:
        """Search and return results in server_b format (TechTalk)"""
        try:
            # Validate limit parameter
            limit = min(max(1, limit), 100)

            # Calculate how many results we need to fetch for pagination
            total_needed = page_num * limit

            # Create embedding for the query
            query_embedding = self.embedding_service.create_query_embedding(
                query)

            # Perform semantic search
            search_results = self.es_client.hybrid_search(
                index_name=self.index_name,
                query=query,
                query_embedding=query_embedding,
                top_k=min(total_needed, 100)
            )

            # Handle pagination by slicing results
            start_idx = (page_num - 1) * limit
            end_idx = start_idx + limit
            paginated_results = search_results[start_idx:end_idx]

            # Apply sorting if requested
            if sort_by_price in ["asc", "desc"]:
                paginated_results.sort(
                    key=lambda x: float(x.get("schema_org", {}).get(
                        "offers", {}).get("price", 0) or 0),
                    reverse=(sort_by_price == "desc")
                )

            # Transform results to server_b format
            bizarre_items = []
            for result in paginated_results:
                schema_org = result.get("schema_org", {})

                # Extract price information with fallbacks
                price_info = schema_org.get("offers", {})
                if isinstance(price_info, list) and price_info:
                    price_info = price_info[0]

                # Fallback to direct price field
                direct_price = result.get("price", 0)
                cost_amount = price_info.get("price") or price_info.get(
                    "lowPrice") or direct_price or ""

                # Extract product ID with better fallback logic
                catalog_id = result.get("_id") or result.get(
                    "id") or schema_org.get("@id") or ""

                # Extract SKU/identifier with fallbacks
                product_identifier = schema_org.get("sku") or result.get(
                    "sku") or result.get("product_id") or catalog_id

                # Extract description with fallbacks
                description = result.get(
                    "description", "") or schema_org.get("description", "")

                # Extract category with fallbacks
                category = schema_org.get(
                    "category") or result.get("category") or ""

                bizarre_items.append({
                    "catalog_entry_id": product_identifier,
                    "merchandise_title": result.get("title") or schema_org.get("name", ""),

                    "financial_details": {
                        "cost_amount": str(cost_amount) if cost_amount else "",
                        "standard_rate": price_info.get("highPrice", ""),
                        "discount_rate": str(cost_amount) if cost_amount else "",
                        "bargain_active": bool(cost_amount),
                    },

                    "content_sections": {
                        "detailed_info": description.strip()[:1024] + "..." if len(description) > 1024 and include_descriptions else ("" if not include_descriptions else description),
                        "quick_summary": description.strip()[:200] if description and include_descriptions else "",
                    } if include_descriptions else {
                        "detailed_info": "",
                        "quick_summary": "",
                    },

                    "inventory_tracking": {
                        "product_identifier": str(product_identifier) if product_identifier else "",
                        "availability_state": "On the shelf",
                        "units_remaining": schema_org.get("inventoryLevel", 0),
                    },

                    "classification_tags": [category] if category else [],
                    "visual_assets": [schema_org.get("image", "")][:2] if schema_org.get("image") else [],
                    "direct_link": result.get("url", ""),
                })

            return json.dumps({
                "query_summary": {
                    "search_terms": query,
                    "page_position": page_num,
                    "results_cap": limit,
                    "price_ordering": sort_by_price,
                    "matches_discovered": len(bizarre_items),
                },
                "catalog_entries": bizarre_items
            }, indent=2)

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return f"Error searching items: {str(e)}"

    def search_server_c_format(self, query: str, results_per_page: int = 10, page_number: int = 1, result_order: str = "relevance", include_descriptions: bool = True) -> str:
        """Search and return results in server_c format (CamelCases)"""
        try:
            # Validate results_per_page parameter
            results_per_page = min(max(1, results_per_page), 100)

            # Calculate how many results we need to fetch for pagination
            total_needed = page_number * results_per_page

            # Create embedding for the query
            query_embedding = self.embedding_service.create_query_embedding(
                query)

            # Perform semantic search
            search_results = self.es_client.hybrid_search(
                index_name=self.index_name,
                query=query,
                query_embedding=query_embedding,
                top_k=min(total_needed, 100)
            )

            # Handle pagination by slicing results
            start_idx = (page_number - 1) * results_per_page
            end_idx = start_idx + results_per_page
            paginated_results = search_results[start_idx:end_idx]

            # Apply sorting based on result_order
            if result_order == "price_low":
                paginated_results.sort(key=lambda x: float(
                    x.get("schema_org", {}).get("offers", {}).get("price", 0) or 0))
            elif result_order == "price_high":
                paginated_results.sort(key=lambda x: float(x.get("schema_org", {}).get(
                    "offers", {}).get("price", 0) or 0), reverse=True)
            elif result_order == "name":
                paginated_results.sort(
                    key=lambda x: x.get("title", "").lower())
            elif result_order == "stock":
                paginated_results.sort(key=lambda x: x.get(
                    "schema_org", {}).get("inventoryLevel", 0), reverse=True)

            # Transform results to server_c format
            eccentric_inventory = []
            for result in paginated_results:
                schema_org = result.get("schema_org", {})

                # Extract price information with fallbacks
                price_info = schema_org.get("offers", {})
                if isinstance(price_info, list) and price_info:
                    price_info = price_info[0]

                # Fallback to direct price field
                direct_price = result.get("price", 0)
                selling_price = price_info.get("price") or price_info.get(
                    "lowPrice") or direct_price or ""

                # Extract warehouse SKU with better fallback logic
                warehouse_sku = result.get("_id") or result.get(
                    "id") or schema_org.get("@id") or ""

                # Extract product identifier with fallbacks
                product_identifier = schema_org.get("sku") or result.get(
                    "sku") or result.get("product_id") or warehouse_sku

                # Extract description with fallbacks
                description = result.get(
                    "description", "") or schema_org.get("description", "")

                eccentric_inventory.append({
                    "warehouse_sku_code": product_identifier,
                    "merchandise_designation": result.get("title") or schema_org.get("name", ""),

                    "availability_metrics": {
                        "inventory_status": "instock",
                        "units_on_hand": schema_org.get("inventoryLevel", 0),
                        "stock_management": True,
                        "low_stock_threshold": 5,
                    },

                    "commercial_data": {
                        "selling_price": str(selling_price) if selling_price else "",
                        "market_value": price_info.get("highPrice", ""),
                        "promotional_rate": str(selling_price) if selling_price else "",
                        "discount_active": bool(selling_price),
                    },

                    "catalog_reference": {
                        "product_identifier": str(product_identifier) if product_identifier else "",
                        "storefront_link": result.get("url", ""),
                        "type_classification": schema_org.get("@type", "Product"),
                    },

                    "product_overview": {
                        "brief_description": description.strip()[:200] if description and include_descriptions else "",
                        "full_description": description.strip() if description and include_descriptions else "",
                    } if include_descriptions else {
                        "brief_description": "",
                        "full_description": "",
                    },
                })

            in_stock_count = len(
                [p for p in eccentric_inventory if p["availability_metrics"]["inventory_status"] == "instock"])
            out_of_stock_count = len(eccentric_inventory) - in_stock_count

            return json.dumps({
                "inventory_query": {
                    "search_criteria": query,
                    "page_index": page_number,
                    "batch_size": results_per_page,
                    "sorting_method": result_order,
                    "matches_found": len(eccentric_inventory),
                },
                "stock_intelligence": {
                    "total_items_analyzed": len(eccentric_inventory),
                    "in_stock_count": in_stock_count,
                    "out_of_stock_count": out_of_stock_count,
                },
                "warehouse_catalog": eccentric_inventory
            }, indent=2)

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return f"Error querying stock: {str(e)}"

    def search_server_d_format(self, query: str, results_limit: int = 10, page_number: int = 1, min_price: float = None, max_price: float = None, include_descriptions: bool = True) -> str:
        """Search and return results in server_d format (Hardware Cafe)"""
        try:
            # Validate results_limit parameter
            results_limit = min(max(1, results_limit), 100)

            # Calculate how many results we need to fetch for pagination
            total_needed = page_number * results_limit

            # Create embedding for the query
            query_embedding = self.embedding_service.create_query_embedding(
                query)

            # Perform semantic search
            search_results = self.es_client.hybrid_search(
                index_name=self.index_name,
                query=query,
                query_embedding=query_embedding,
                top_k=min(total_needed, 100)
            )

            # Handle pagination by slicing results
            start_idx = (page_number - 1) * results_limit
            end_idx = start_idx + results_limit
            paginated_results = search_results[start_idx:end_idx]

            # Apply price filtering if specified
            is_budget_search = min_price is not None or max_price is not None
            if is_budget_search:
                filtered_results = []
                for result in paginated_results:
                    schema_org = result.get("schema_org", {})
                    price_info = schema_org.get("offers", {})
                    if isinstance(price_info, list) and price_info:
                        price_info = price_info[0]

                    price = float(price_info.get("price", 0) or 0)
                    if min_price is not None and price < min_price:
                        continue
                    if max_price is not None and price > max_price:
                        continue
                    filtered_results.append(result)
                paginated_results = filtered_results

            # Transform results to server_d format
            unconventional_items = []
            for result in paginated_results:
                schema_org = result.get("schema_org", {})

                # Extract price information with fallbacks
                price_info = schema_org.get("offers", {})
                if isinstance(price_info, list) and price_info:
                    price_info = price_info[0]

                # Fallback to direct price field
                direct_price = result.get("price", 0)
                purchase_cost = price_info.get("price") or price_info.get(
                    "lowPrice") or direct_price or ""

                # Extract catalog SKU with better fallback logic
                catalog_sku = result.get("_id") or result.get(
                    "id") or schema_org.get("@id") or ""

                # Extract product identifier with fallbacks
                product_identifier = schema_org.get("sku") or result.get(
                    "sku") or result.get("product_id") or catalog_sku

                # Extract description with fallbacks
                description = result.get(
                    "description", "") or schema_org.get("description", "")

                # Extract category with fallbacks
                category = schema_org.get(
                    "category") or result.get("category") or ""

                unconventional_items.append({
                    "catalog_sku_code": product_identifier,
                    "item_designation": result.get("title") or schema_org.get("name", ""),

                    "economic_data": {
                        "purchase_cost": str(purchase_cost) if purchase_cost else "",
                        "baseline_price": price_info.get("highPrice", ""),
                        "markdown_price": str(purchase_cost) if purchase_cost else "",
                        "promotion_flag": bool(purchase_cost),
                    },

                    "descriptive_content": {
                        "brief_summary": description.strip()[:200] if description and include_descriptions else "",
                        "extended_details": description.strip()[:500] + "..." if len(description) > 500 and include_descriptions else ("" if not include_descriptions else description),
                    } if include_descriptions else {
                        "brief_summary": "",
                        "extended_details": "",
                    },

                    "organizational_tags": [category] if category else [],
                    "storefront_reference": result.get("url", ""),

                    "inventory_status": {
                        "product_identifier": str(product_identifier) if product_identifier else "",
                        "availability": "InStock",
                        "units_available": schema_org.get("inventoryLevel", 0),
                    },
                })

            # Calculate average price
            avg_price = 0
            if unconventional_items:
                valid_prices = []
                for item in unconventional_items:
                    cost = item["economic_data"]["purchase_cost"]
                    if cost and str(cost).replace('.', '').isdigit():
                        valid_prices.append(float(cost))
                if valid_prices:
                    avg_price = sum(valid_prices) / len(valid_prices)

            return json.dumps({
                "search_expedition": {
                    "search_keyword": query,
                    "current_page": page_number,
                    "limit_per_page": results_limit,
                    "budget_search_detected": is_budget_search,
                    "price_constraints": {
                        "minimum_threshold": min_price,
                        "maximum_threshold": max_price,
                    },
                },
                "discovery_metrics": {
                    "total_matched_items": len(unconventional_items),
                    "average_price": avg_price,
                    "price_range_applied": min_price is not None or max_price is not None,
                },
                "marketplace_inventory": unconventional_items
            }, indent=2)

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return f"Error retrieving items by keyword: {str(e)}"
