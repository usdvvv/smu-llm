from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import sys
import os
import asyncio

# Add project directory to path
sys.path.append(os.path.dirname(__file__))

# Import modules from main.py
from app.agent.manus import Manus
from app.logger import logger
from app.config import config
from app.utils.ollama_check import check_ollama_models, test_ollama_connection
from app.llm import LLM

app = FastAPI(title="SMU LLM API Server")

# Initialize the agent
agent = None

class RequestData(BaseModel):
    query: str


@app.on_event("startup")
async def startup_event():
    """Runs when the server starts up."""
    global agent
    
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
    
    # Initialize the agent
    agent = Manus()
    logger.info("Agent initialized and server is ready to process requests")


@app.get("/")
async def root():
    """API root endpoint."""
    return {"message": "SMU LLM Server is running. Use /docs to see the API documentation."}


async def process_with_agent(prompt):
    """Process a prompt using the Manus agent."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized. Check server logs.")
    
    try:
        result = await agent.run(prompt)
        
        # Format the response
        response = {"full_result": result}
        
        # Extract final answer if available
        if result and "===== FINAL ANSWER =====" in result:
            try:
                final_answer = result.split("===== FINAL ANSWER =====")[1].split("=======================")[0].strip()
                response["final_answer"] = final_answer
            except Exception as e:
                logger.error(f"Error extracting final answer: {e}")
                response["final_answer"] = None
        else:
            response["final_answer"] = None
            
        return response
    except Exception as e:
        logger.error(f"Error processing request with agent: {e}")
        
        # Fallback to direct LLM response
        try:
            llm = LLM()
            simple_prompt = f"{prompt} Please keep your answer brief and simple."
            direct_answer = await llm.ask(
                messages=[{"role": "user", "content": simple_prompt}],
                stream=False
            )
            return {
                "full_result": None, 
                "final_answer": None,
                "direct_answer": direct_answer,
                "error": str(e)
            }
        except Exception as fallback_error:
            logger.error(f"Error generating direct answer: {fallback_error}")
            raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")


@app.post("/process")
async def process_request(data: RequestData):
    """Process a request through the LLM."""
    if not data.query.strip():
        raise HTTPException(status_code=400, detail="Empty query provided")
    
    logger.info(f"Processing request: {data.query[:50]}...")
    result = await process_with_agent(data.query)
    logger.info("Request processing completed")
    
    return JSONResponse(content=result)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global agent
    return {
        "status": "healthy",
        "agent_initialized": agent is not None
    }


if __name__ == "__main__":
    # Run the server on 0.0.0.0 to make it accessible on the network
    # on port 8000 (or any port you prefer)
    uvicorn.run(app, host="0.0.0.0", port=8000)
