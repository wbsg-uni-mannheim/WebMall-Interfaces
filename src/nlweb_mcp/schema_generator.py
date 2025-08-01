import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class SchemaOrgGenerator:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def generate_product_schema(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate schema.org Product JSON-LD from product data"""
        
        # Extract basic product information
        product_id = product_data.get('product_id', '')
        title = product_data.get('title', '')
        description = product_data.get('description', '')
        price = product_data.get('price', 0.0)
        url = product_data.get('url', '')
        category = product_data.get('category', '')
        
        # Generate schema.org Product
        schema = {
            "@context": "https://schema.org/",
            "@type": "Product",
            "name": title,
            "description": description,
            "url": url,
            "identifier": product_id,
            "mpn": f"-{product_id}",  # Manufacturer Part Number
        }
        
        # Add brand information based on the WebMall shop
        brand_name = self._get_brand_name()
        schema["brand"] = {
            "@type": "Brand",
            "name": brand_name
        }
        
        # Add offer information
        if price > 0:
            schema["offers"] = {
                "@type": "Offer",
                "price": str(price),
                "priceCurrency": "USD",
                "availability": "https://schema.org/InStock",
                "itemCondition": "https://schema.org/NewCondition",
                "seller": {
                    "@type": "Organization",
                    "name": brand_name,
                    "description": "",
                    "url": self.base_url
                }
            }
        
        # Add category information
        if category:
            schema["category"] = category
        
        # Add image if available from raw product data
        raw_product = product_data.get('raw_product', {})
        if raw_product.get('images') and len(raw_product['images']) > 0:
            image_url = raw_product['images'][0].get('src', '')
            if image_url:
                schema["image"] = {
                    "@type": "ImageObject",
                    "url": image_url
                }
        
        # Add related products as recommendations
        related_products = product_data.get('related_products', [])
        if related_products:
            # Convert related product IDs to URLs
            related_urls = []
            for related_id in related_products:
                related_url = self._construct_product_url(related_id)
                if related_url:
                    related_urls.append(related_url)
            
            if related_urls:
                schema["isRelatedTo"] = [
                    {
                        "@type": "Product",
                        "url": url
                    } for url in related_urls
                ]
        
        # Add additional product properties if available
        if raw_product:
            # Add SKU if available
            if raw_product.get('sku'):
                schema["sku"] = raw_product['sku']
            
            # Add weight if available
            if raw_product.get('weight'):
                schema["weight"] = {
                    "@type": "QuantitativeValue",
                    "value": raw_product['weight'],
                    "unitCode": "LBR"  # Pounds
                }
            
            # Add dimensions if available
            dimensions = raw_product.get('dimensions', {})
            if dimensions and any(dimensions.values()):
                schema["depth"] = {
                    "@type": "QuantitativeValue",
                    "value": dimensions.get('length', ''),
                    "unitCode": "INH"  # Inches
                }
                schema["width"] = {
                    "@type": "QuantitativeValue",
                    "value": dimensions.get('width', ''),
                    "unitCode": "INH"
                }
                schema["height"] = {
                    "@type": "QuantitativeValue",
                    "value": dimensions.get('height', ''),
                    "unitCode": "INH"
                }
        
        return schema
    
    def _get_brand_name(self) -> str:
        """Get brand name based on the base URL"""
        if "webmall-1" in self.base_url:
            return "E-Store Athletes"
        elif "webmall-2" in self.base_url:
            return "TechTalk"
        elif "webmall-3" in self.base_url:
            return "CamelCases"
        elif "webmall-4" in self.base_url:
            return "Hardware Cafe"
        else:
            return "WebMall Store"
    
    def _construct_product_url(self, product_id: str) -> Optional[str]:
        """Construct a product URL from product ID"""
        if not product_id:
            return None
        
        # This is a simplified URL construction
        # In a real implementation, you might need to fetch the actual slug
        return f"{self.base_url}/product/product-{product_id}/"
    
    def generate_product_list_schema(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate schema.org ItemList for multiple products"""
        
        list_items = []
        for i, product_data in enumerate(products):
            product_schema = self.generate_product_schema(product_data)
            
            list_item = {
                "@type": "ListItem",
                "position": i + 1,
                "item": product_schema
            }
            list_items.append(list_item)
        
        schema = {
            "@context": "https://schema.org/",
            "@type": "ItemList",
            "numberOfItems": len(products),
            "itemListElement": list_items
        }
        
        return schema
    
    def validate_schema(self, schema: Dict[str, Any]) -> bool:
        """Basic validation of schema.org structure"""
        required_fields = ["@context", "@type"]
        
        for field in required_fields:
            if field not in schema:
                logger.error(f"Missing required field: {field}")
                return False
        
        if schema.get("@type") not in ["Product", "ItemList"]:
            logger.error(f"Invalid @type: {schema.get('@type')}")
            return False
        
        if schema.get("@type") == "Product":
            if not schema.get("name"):
                logger.warning("Product missing name field")
        
        return True