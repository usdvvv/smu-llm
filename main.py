import asyncio
import os
import time
from pathlib import Path

from app.agent.manus import Manus
from app.logger import logger
from app.config import config
from app.utils.ollama_check import check_ollama_models, test_ollama_connection
from app.swarm import SwarmCoordinator  # Import SwarmCoordinator from the new module


async def main():
    # Check Ollama connection if using Ollama
    if config.llm["default"].api_type == "ollama":
        logger.info("Checking Ollama connection...")
        models = await check_ollama_models(config.llm["default"].base_url)
        
        if not models:
            logger.warning("No Ollama models found. Make sure Ollama is running and models are pulled.")
            logger.info("You can pull models with: ollama pull mistral:latest")
            return
        
        model_name = config.llm["default"].model
        connection_ok = await test_ollama_connection(config.llm["default"].base_url, model_name)
        
        if not connection_ok:
            logger.warning(f"Could not connect to Ollama with model {model_name}. Check if Ollama is running.")
            return
        
        logger.info(f"Successfully connected to Ollama using {model_name}")
    
    # Initialize both traditional agent and swarm coordinator
    agent = Manus()
    swarm = SwarmCoordinator()
    
    try:
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        # Ask if they want to use the swarm approach
        use_swarm = input("Use faster swarm processing? (y/n, default=n): ").lower().startswith('y')
        
        logger.info("Processing your request...")
        start_time = time.time()
        
        if use_swarm:
            # Use swarm approach
            logger.info("Using swarm processing...")
            result = await swarm.process_query(prompt)
            
            # Format the result
            final_answer = result
            elapsed_time = time.time() - start_time
            logger.info(f"Swarm processing completed in {elapsed_time:.2f} seconds")
        else:
            # Use traditional approach
            logger.info("Using traditional processing...")
            result = await agent.run(prompt)
            
            # Extract and display the final answer if it's in the expected format
            if result and "===== FINAL ANSWER =====" in result:
                try:
                    final_answer = result.split("===== FINAL ANSWER =====")[1].split("=======================")[0].strip()
                except Exception:
                    final_answer = result
            else:
                final_answer = result
                
            elapsed_time = time.time() - start_time
            logger.info(f"Traditional processing completed in {elapsed_time:.2f} seconds")
        
        # Display result
        print("\n" + "="*60)
        print(" "*20 + "FINAL ANSWER")
        print("="*60)
        print(final_answer)
        print("="*60 + "\n")
        
        logger.info(f"Request processing completed in {elapsed_time:.2f} seconds.")
        
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        print("\nAn error occurred while processing your request.")
        
        # Try to get a direct answer to the specific question
        try:
            from app.llm import LLM
            
            # Simplify the prompt to improve chances of success
            simple_prompt = f"{prompt} Please keep your answer brief and simple."
            
            llm = LLM()
            direct_answer = await llm.ask(
                messages=[{"role": "user", "content": simple_prompt}],
                stream=False
            )
            
            print("\n" + "="*60)
            print(" "*20 + "DIRECT ANSWER")
            print("="*60)
            print(direct_answer)
            print("="*60 + "\n")
        except Exception as fallback_error:
            logger.error(f"Error generating direct answer: {fallback_error}")
            print("\n" + "="*60)
            print("COULD NOT GENERATE ANSWER")
            print("="*60)
            print(f"Sorry, I couldn't generate an answer to your question: '{prompt}'\nPlease try asking again or rephrasing your question.")
            print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())