import asyncio
import time
from typing import Dict, List, Any

from app.logger import logger
from .multi_agent_pipeline import MultiAgentPipeline

class ResponseCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value

class SwarmCoordinator:
    def __init__(self):
        # Initialize MultiAgentPipeline instead of TaskPipeline
        self.task_pipeline = MultiAgentPipeline()
        
        # Keep the response cache for potential future use
        self.response_cache = ResponseCache()
    
    async def process_query(self, query):
        start_time = time.time()
        
        logger.info("🚀 Starting Multi-Agent Processing...")
        
        try:
            # Use the new multi-agent pipeline to process the query
            result = await self.task_pipeline.execute_workflow(query)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Multi-Agent Processing completed in {elapsed_time:.2f} seconds")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Multi-Agent Pipeline error: {e}")
            
            # Fallback mechanism
            try:
                from app.llm import LLM
                fallback_llm = LLM()
                fallback_result = await fallback_llm.ask(
                    messages=[{
                        "role": "user", 
                        "content": f"Provide a concise answer to: {query}\n\nError occurred in multi-agent pipeline: {str(e)}"
                    }],
                    stream=False
                )
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"Fallback error: {fallback_error}")
                return f"Unable to process query. Error: {str(e)}"