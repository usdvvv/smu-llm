import asyncio
import json
import time
import traceback
from typing import Dict, List, Any, Optional

from app.logger import logger
from app.llm import LLM
from app.tool import ToolCollection

class TaskPipeline:
    def __init__(self):
        # Simplified agent configuration
        self.agents = {
            "browser_agent": LLM(),  # Dedicated browser/research agent
            "synthesizer": LLM()     # Synthesis agent
        }
        
        # Optimize model configurations
        self.agents["browser_agent"].model = "mistral:latest"
        self.agents["browser_agent"].temperature = 0.3
        
        self.agents["synthesizer"].model = "deepseek-r1:latest"
        self.agents["synthesizer"].temperature = 0.2
        
        self.task_results = {}
    
    async def execute_workflow(self, query: str):
        start_time = time.time()
        logger.info(f"🚀 Starting workflow for query: {query}")
        
        try:
            # Combine browser navigation and analysis in a single task
            browser_task_result = await self._execute_browser_task(query)
            
            # Directly synthesize results
            final_answer = await self._synthesize_results(query, browser_task_result)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Task Pipeline completed in {elapsed_time:.2f} seconds")
            
            return final_answer
        
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _execute_browser_task(self, query: str):
        """Execute browser task with tool support"""
        try:
            # Use ask_tool for browser-specific tasks
            result = await self.agents["browser_agent"].ask_tool(
                messages=[{
                    "role": "user", 
                    "content": f"Use browser tool to navigate and describe: {query}"
                }],
                tools=ToolCollection().to_params(),
                stream=False,
                timeout=20  # Strict timeout
            )
            
            # Extract content, handling different result types
            task_result = result.content if hasattr(result, 'content') else str(result)
            
            logger.info(f"🌐 Browser task completed: {task_result[:100]}...")
            return task_result
        
        except Exception as e:
            logger.error(f"Browser task error: {e}")
            return f"Error in browser task: {str(e)}"
    
    async def _synthesize_results(self, original_query: str, browser_result: str):
        """Synthesize final results with focused prompt"""
        synthesis_prompt = f"""
        Synthesize a concise description based on the browser navigation:
        
        Query: {original_query}
        Browser Result: {browser_result}
        
        Requirements:
        - Provide a clear, direct description
        - Focus on key visual and informational elements
        - Keep the answer under 250 words
        """
        
        final_answer = await self.agents["synthesizer"].ask(
            messages=[{"role": "user", "content": synthesis_prompt}],
            stream=False,
            temperature=0.2  # More focused synthesis
        )
        
        return final_answer