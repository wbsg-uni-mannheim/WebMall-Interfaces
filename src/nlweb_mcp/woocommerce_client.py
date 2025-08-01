import logging
from typing import Dict, List, Optional, Any
from woocommerce import API

# Handle both relative and absolute imports
try:
    from .config import WOOCOMMERCE_TIMEOUT
except ImportError:
    from config import WOOCOMMERCE_TIMEOUT

logger = logging.getLogger(__name__)

class WooCommerceClient:
    def __init__(self, base_url: str, consumer_key: Optional[str] = None, consumer_secret: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        
        # Initialize WooCommerce API client
        if consumer_key and consumer_secret:
            self.api = API(
                url=base_url,
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                version="wc/v3",
                timeout=WOOCOMMERCE_TIMEOUT
            )
        else:
            # For public endpoints or when authentication is not available
            self.api = API(
                url=base_url,
                consumer_key="",
                consumer_secret="",
                version="wc/v3",
                timeout=WOOCOMMERCE_TIMEOUT
            )
    
    def get_products(self, per_page: int = 100, page: int = 1, **kwargs) -> List[Dict[str, Any]]:
        """Get products from the WooCommerce API"""
        params = {
            'per_page': min(per_page, 100),  # WooCommerce max is 100
            'page': page,
            **kwargs
        }
        
        try:
            response = self.api.get("products", params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: HTTP {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Failed to get products: {e}")
            return []
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        """Get all products by paginating through all pages"""
        all_products = []
        page = 1
        
        while True:
            products = self.get_products(per_page=100, page=page)
            if not products:
                break
            
            all_products.extend(products)
            logger.info(f"Retrieved {len(products)} products from page {page}")
            
            # If we got less than 100 products, we've reached the end
            if len(products) < 100:
                break
            
            page += 1
        
        logger.info(f"Retrieved {len(all_products)} total products from {self.base_url}")
        return all_products
    
    def get_product_categories(self, per_page: int = 100, page: int = 1) -> List[Dict[str, Any]]:
        """Get product categories"""
        params = {
            'per_page': min(per_page, 100),
            'page': page
        }
        
        try:
            response = self.api.get("products/categories", params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: HTTP {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []
    
    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Get all categories by paginating through all pages"""
        all_categories = []
        page = 1
        
        while True:
            categories = self.get_product_categories(per_page=100, page=page)
            if not categories:
                break
            
            all_categories.extend(categories)
            
            if len(categories) < 100:
                break
            
            page += 1
        
        return all_categories
    
    def extract_product_data(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize product data from WooCommerce API response"""
        
        # Extract basic product info with consistent ID normalization
        raw_product_id = product.get('id', '')
        product_id = str(raw_product_id).strip()  # Normalize to string and strip whitespace
        logger.debug(f"WooCommerceClient.extract_product_data: Normalized product ID from {repr(raw_product_id)} to {repr(product_id)}")
        
        title = product.get('name', '')
        description = product.get('description', '') or product.get('short_description', '')
        
        # Extract price (handle different price types)
        price = 0.0
        if product.get('price'):
            try:
                price = float(product['price'])
            except (ValueError, TypeError):
                pass
        
        # Extract categories
        categories = []
        for cat in product.get('categories', []):
            categories.append(cat.get('name', ''))
        category = ', '.join(categories) if categories else ''
        
        # Extract related products
        related_products = []
        for related_id in product.get('related_ids', []):
            related_products.append(str(related_id))
        
        # Build product URL
        product_url = product.get('permalink', '')
        if not product_url and product_id:
            # Fallback: construct URL from base URL and product ID
            product_url = f"{self.base_url}/product/{product.get('slug', product_id)}/"
        
        # Generate schema.org structured data
        schema_org = self.generate_schema_org_data(product)
        
        return {
            'product_id': product_id,
            'url': product_url,
            'title': title,
            'price': price,
            'description': description,
            'related_products': related_products,
            'category': category,
            'schema_org': schema_org,
            'raw_product': product  # Keep original data for reference
        }
    
    def generate_schema_org_data(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Generate schema.org Product structured data from WooCommerce product"""
        
        # Extract basic product info
        product_name = product.get('name', '')
        description = product.get('description', '') or product.get('short_description', '')
        
        # Extract images
        images = []
        for img in product.get('images', []):
            if img.get('src'):
                images.append(img['src'])
        
        # Extract price information
        offers = {}
        if product.get('price'):
            try:
                price_value = float(product['price'])
                offers = {
                    "@type": "Offer",
                    "url": product.get('permalink', ''),
                    "priceCurrency": "EUR",  # Assuming EUR based on the price_html format seen in debug
                    "price": str(price_value),
                    "priceValidUntil": "",  # Could add sale end date if available
                    "availability": "https://schema.org/InStock" if product.get('stock_status') == 'instock' else "https://schema.org/OutOfStock",
                    "itemCondition": "https://schema.org/NewCondition"  # Default, could be determined from description
                }
                
                # Add sale price information if available
                if product.get('sale_price') and product.get('regular_price'):
                    try:
                        regular_price = float(product['regular_price'])
                        sale_price = float(product['sale_price'])
                        offers.update({
                            "lowPrice": str(sale_price),
                            "highPrice": str(regular_price)
                        })
                    except (ValueError, TypeError):
                        pass
            except (ValueError, TypeError):
                pass
        
        # Extract categories
        categories = []
        for cat in product.get('categories', []):
            categories.append(cat.get('name', ''))
        
        # Extract brand from description or use empty string
        brand = ""
        desc_text = description.lower() if description else ""
        # Simple brand extraction - could be enhanced
        if 'nintendo' in desc_text:
            brand = "Nintendo"
        elif 'sony' in desc_text:
            brand = "Sony"
        elif 'microsoft' in desc_text:
            brand = "Microsoft"
        # Add more brand patterns as needed
        
        # Build schema.org Product structure
        schema_org = {
            "@context": "https://schema.org/",
            "@type": "Product",
            "name": product_name,
            "image": images[:1],  # Take first image for main image
            "description": description.strip() if description else "",
            "sku": product.get('sku', ''),
            "mpn": product.get('sku', ''),  # Using SKU as MPN if available
            "brand": {
                "@type": "Brand",
                "name": brand
            } if brand else {},
            "category": categories[0] if categories else "",
            "offers": offers if offers else {},
            "aggregateRating": {
                "@type": "AggregateRating",
                "ratingValue": product.get('average_rating', '0'),
                "reviewCount": product.get('rating_count', 0)
            } if product.get('rating_count', 0) > 0 else {},
            "url": product.get('permalink', ''),
            "identifier": {
                "@type": "PropertyValue",
                "name": "WooCommerce ID",
                "value": str(product.get('id', ''))
            }
        }
        
        # Clean up empty fields
        schema_org = {k: v for k, v in schema_org.items() if v not in [None, '', {}, []]}
        
        return schema_org
    
    def test_connection(self) -> bool:
        """Test connection to WooCommerce API"""
        try:
            # Try to get a single product to test the connection
            response = self.api.get("products", params={"per_page": 1})
            if response.status_code == 200:
                logger.info(f"Successfully connected to WooCommerce API at {self.base_url}")
                return True
            else:
                logger.error(f"Failed to connect to WooCommerce API at {self.base_url}: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to WooCommerce API at {self.base_url}: {e}")
            return False