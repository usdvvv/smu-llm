import asyncio
import json
import time
import traceback
from typing import Dict, List, Any, Optional

from app.logger import logger
from app.llm import LLM
from app.tool import ToolCollection

class Agent:
    def __init__(self, name: str, model: str, role: str):
        self.name = name
        self.llm = LLM()
        self.llm.model = model
        self.llm.temperature = 0.3  # More focused responses
        self.role = role

    async def process_task(self, task: Dict, shared_context: Dict):
        """
        Process a task with access to shared context
        """
        try:
            context_str = "\n".join([
                f"{k}: {v}" for k, v in shared_context.items()
            ])
            
            full_instructions = f"""
            Role: {self.role}
            Task: {task['instructions']}
            
            Shared Context:
            {context_str}
            
            Be concise and direct in your response.
            """
            
            # Execute task based on type
            if task.get('tool_type'):
                result = await self.llm.ask_tool(
                    messages=[{"role": "user", "content": full_instructions}],
                    tools=ToolCollection().to_params(),
                    stream=False,
                    timeout=30  # Strict timeout
                )
            else:
                result = await self.llm.ask(
                    messages=[{"role": "user", "content": full_instructions}],
                    stream=False,
                    timeout=30  # Strict timeout
                )
            
            return {
                "agent": self.name,
                "role": self.role,
                "result": result.content if hasattr(result, 'content') else str(result)
            }
        
        except Exception as e:
            logger.error(f"Agent {self.name} task error: {e}")
            return {
                "agent": self.name,
                "role": self.role,
                "result": f"Error: {str(e)}"
            }

class MultiAgentPipeline:
    def __init__(self):
        # Create specialized agents with Llama 3 and other models
        self.agents = [
            Agent("browser_navigator", "mistral:latest", "Web Navigation"),
            Agent("researcher", "deepseek-r1:latest", "Research"),
            Agent("synthesizer", "llama3:8b", "Result Synthesis")
        ]
        
        self.shared_context = {}
    
    async def execute_workflow(self, query: str):
        start_time = time.time()
        logger.info(f"🚀 Multi-Agent Workflow Starting: {query}")
        
        try:
            # Determine the type of query
            is_web_query = any(keyword in query.lower() for keyword in 
                               ["navigate", "website", "web page", "homepage", "browser"])
            
            if is_web_query:
                # Browser-based query
                browser_result = await self._execute_browser_task(query)
                final_answer = await self._synthesize_results(browser_result, query)
            else:
                # Standard research query
                research_result = await self._execute_research_task(query)
                final_answer = await self._synthesize_results(research_result, query)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Multi-Agent Workflow Completed in {elapsed_time:.2f} seconds")
            
            return final_answer
        
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _execute_browser_task(self, query: str):
        """Execute browser task with tool support"""
        browser_agent = next(agent for agent in self.agents if agent.name == "browser_navigator")
        
        browser_task = {
            "id": "browser_navigation",
            "instructions": f"Navigate and extract detailed information: {query}",
            "tool_type": "browser"
        }
        
        try:
            result = await browser_agent.process_task(browser_task, {})
            return result['result']
        
        except Exception as e:
            logger.error(f"Browser task error: {e}")
            return f"Error in browser navigation: {str(e)}"
    
    async def _execute_research_task(self, query: str):
        """Execute research task for non-web queries"""
        researcher = next(agent for agent in self.agents if agent.name == "researcher")
        
        research_task = {
            "id": "general_research",
            "instructions": f"Provide a comprehensive explanation for: {query}",
        }
        
        try:
            result = await researcher.process_task(research_task, {})
            return result['result']
        
        except Exception as e:
            logger.error(f"Research task error: {e}")
            return f"Error in research: {str(e)}"
    
    async def _synthesize_results(self, result: str, query: str):
        """Synthesize final results with focused prompt"""
        synthesizer = next(agent for agent in self.agents if agent.name == "synthesizer")
        
        synthesis_prompt = f"""
        Synthesize a clear and concise answer for the query:
        
        Query: {query}
        Source Information: {result}
        
        Requirements:
        - Provide a direct, informative response
        - Focus on key points
        - Use clear, accessible language
        - Keep the answer under 250 words
        """
        
        final_answer = await synthesizer.llm.ask(
            messages=[{"role": "user", "content": synthesis_prompt}],
            stream=False,
            temperature=0.2  # Very focused synthesis
        )
        
        return final_answer.content if hasattr(final_answer, 'content') else str(final_answer)