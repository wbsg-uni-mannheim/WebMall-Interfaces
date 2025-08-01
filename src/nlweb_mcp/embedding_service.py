import openai
import logging
import time
import asyncio
from typing import List, Union, Dict, Any

# Handle both relative and absolute imports
try:
    from .config import OPENAI_API_KEY, EMBEDDING_MODEL
except ImportError:
    from config import OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

print(OPENAI_API_KEY)

class EmbeddingService:
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = EMBEDDING_MODEL):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            timeout=30.0,  # Reduced to 30 second timeout
            max_retries=2  # Reduced retries to prevent hanging
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=30.0,  # 30 second timeout for async client
            max_retries=2
        )
        self.model = model
        self.rate_limit_delay = 0.05  # Reduced to 50ms delay between requests
        
    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for a single text"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * 1536  # Return zero vector for empty text
        
        try:
            # Add rate limiting
            time.sleep(self.rate_limit_delay)
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text.strip(),
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Created embedding for text: {text[:100]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create embedding for text: {text[:100]}... Error: {e}")
            raise
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Create embeddings for multiple texts in batches"""
        if not texts:
            return []
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._create_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)
            
            logger.info(f"Created embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return embeddings
    
    def _create_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts"""
        # Filter out empty texts and keep track of indices
        non_empty_texts = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text.strip())
                text_indices.append(i)
        
        if not non_empty_texts:
            # Return zero vectors for all texts
            return [[0.0] * 1536 for _ in texts]
        
        try:
            # Add rate limiting
            time.sleep(self.rate_limit_delay * len(non_empty_texts))
            
            response = self.client.embeddings.create(
                model=self.model,
                input=non_empty_texts,
                encoding_format="float"
            )
            
            # Map embeddings back to original positions
            embeddings = [[0.0] * 1536 for _ in texts]
            for i, embedding_data in enumerate(response.data):
                original_index = text_indices[i]
                embeddings[original_index] = embedding_data.embedding
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create batch embeddings. Error: {e}")
            # Fallback to individual embedding creation
            return [self.create_embedding(text) for text in texts]
    
    def create_query_embedding(self, query: str) -> List[float]:
        """Create an embedding for a search query"""
        return self.create_embedding(query)
    
    async def async_create_embedding(self, text: str) -> List[float]:
        """Create an embedding for a single text asynchronously"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * 1536  # Return zero vector for empty text
        
        try:
            # Add rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=text.strip(),
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Created async embedding for text: {text[:100]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create async embedding for text: {text[:100]}... Error: {e}")
            raise
    
    async def async_create_query_embedding(self, query: str) -> List[float]:
        """Create an embedding for a search query asynchronously"""
        return await self.async_create_embedding(query)
    
    def create_product_text_representation(self, product_data: Dict[str, Any]) -> str:
        """Create a text representation of a product for embedding (legacy method)"""
        # Format: "product_id - url - title - price - description - related_products - category"
        parts = [
            str(product_data.get('product_id', '')),
            str(product_data.get('url', '')),
            str(product_data.get('title', '')),
            f"{product_data.get('price', 0):.2f} EUR",
            str(product_data.get('description', '')),
            ', '.join(product_data.get('related_products', [])),
            str(product_data.get('category', ''))
        ]
        
        return ' - '.join(parts)
    
    def create_title_text_representation(self, product_data: Dict[str, Any]) -> str:
        """Create a text representation of product title with price and category for embedding"""
        parts = []
        
        # Title is the main component
        title = str(product_data.get('title', ''))
        if title:
            parts.append(title)
        
        # Add price information
        price = product_data.get('price', 0)
        if price and price > 0:
            parts.append(f"Price: {price:.2f} EUR")
        
        # Add category for better context
        category = str(product_data.get('category', ''))
        if category:
            parts.append(f"Category: {category}")
        
        return ' - '.join(parts) if parts else "Unknown Product"
    
    def create_content_text_representation(self, product_data: Dict[str, Any]) -> str:
        """Create a text representation of product content (description) for embedding"""
        parts = []
        
        # Description is the main component
        description = str(product_data.get('description', ''))
        if description:
            parts.append(description)
        
        # Add related products information
        related_products = product_data.get('related_products', [])
        if related_products:
            parts.append(f"Related products: {', '.join(related_products)}")
        
        return ' '.join(parts) if parts else "No description available"
    
    def create_separate_embeddings(self, product_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Create separate embeddings for title and content"""
        # Create text representations
        title_text = self.create_title_text_representation(product_data)
        content_text = self.create_content_text_representation(product_data)
        
        # Create embeddings
        title_embedding = self.create_embedding(title_text)
        content_embedding = self.create_embedding(content_text)
        
        return {
            'title_embedding': title_embedding,
            'content_embedding': content_embedding,
            'title_text': title_text,
            'content_text': content_text
        }
    
    def create_separate_embeddings_batch(self, products_data: List[Dict[str, Any]]) -> List[Dict[str, List[float]]]:
        """Create separate embeddings for multiple products in batches"""
        if not products_data:
            return []
        
        # Create text representations for all products
        title_texts = []
        content_texts = []
        
        for product_data in products_data:
            title_text = self.create_title_text_representation(product_data)
            content_text = self.create_content_text_representation(product_data)
            title_texts.append(title_text)
            content_texts.append(content_text)
        
        # Create embeddings in batches
        title_embeddings = self.create_embeddings_batch(title_texts)
        content_embeddings = self.create_embeddings_batch(content_texts)
        
        # Combine results
        results = []
        for i, product_data in enumerate(products_data):
            results.append({
                'title_embedding': title_embeddings[i],
                'content_embedding': content_embeddings[i],
                'title_text': title_texts[i],
                'content_text': content_texts[i]
            })
        
        return results
    
    def test_embedding_service(self) -> bool:
        """Test the embedding service with a simple request"""
        try:
            test_text = "This is a test product for embedding."
            embedding = self.create_embedding(test_text)
            
            if len(embedding) == 1536:  # text-embedding-3-small dimensions
                logger.info("Embedding service test successful")
                return True
            else:
                logger.error(f"Unexpected embedding dimensions: {len(embedding)}")
                return False
                
        except Exception as e:
            logger.error(f"Embedding service test failed: {e}")
            return False