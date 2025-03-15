from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import sys
import os
from typing import Optional

# Add project directory to path
sys.path.append(os.path.dirname(__file__))

# Import modules from main.py
from app.agent.manus import Manus
from app.logger import logger
from app.config import config
from app.utils.ollama_check import check_ollama_models, test_ollama_connection
from app.llm import LLM

# Import for ngrok
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    logger.warning("pyngrok not installed. Ngrok functionality will be disabled.")

app = FastAPI(title="SMU LLM API Server")

# Initialize the agent
agent = None
public_url = None

class RequestData(BaseModel):
    query: str


@app.on_event("startup")
async def startup_event():
    """Runs when the server starts up."""
    global agent, public_url
    
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
    global public_url
    if public_url:
        return {
            "message": "SMU LLM Server is running.",
            "docs": f"{public_url}/docs",
            "public_url": public_url
        }
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
    global agent, public_url
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "public_url": public_url
    }


def start_ngrok(port: int, auth_token: Optional[str] = None):
    """Start ngrok tunnel to expose the local server."""
    global public_url
    
    if not NGROK_AVAILABLE:
        logger.error("Cannot start ngrok: pyngrok is not installed. Run 'pip install pyngrok'")
        return
    
    # Set auth token if provided
    if auth_token:
        ngrok.set_auth_token(auth_token)
    
    try:
        # Open an HTTP tunnel to the specified port
        public_url = ngrok.connect(port).public_url
        logger.info(f"🌐 Ngrok tunnel started at: {public_url}")
        logger.info(f"🌐 API documentation available at: {public_url}/docs")
        return public_url
    except Exception as e:
        logger.error(f"Error starting ngrok tunnel: {e}")
        return None


def run_server(host: str = "0.0.0.0", port: int = 8000, 
               use_ngrok: bool = False, ngrok_token: Optional[str] = None):
    """Start the server with optional ngrok tunnel."""
    if use_ngrok and NGROK_AVAILABLE:
        start_ngrok(port, ngrok_token)
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the SMU LLM Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--ngrok", action="store_true", help="Use ngrok to expose the server to the internet")
    parser.add_argument("--token", type=str, help="Ngrok auth token (if required)")
    
    args = parser.parse_args()
    
    run_server(args.host, args.port, args.ngrok, args.token)
