import asyncio
import httpx
from app.logger import logger

async def check_ollama_models(base_url="http://localhost:11434"):
    """Check available Ollama models and their status"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                if "models" in data:
                    models = data["models"]
                    
                    logger.info(f"Found {len(models)} available Ollama models:")
                    for model in models:
                        logger.info(f"  - {model['name']} ({model.get('size', 'unknown size')})")
                    
                    return models
                    
                else:
                    logger.warning("Unexpected response format from Ollama API")
                    return []
            else:
                logger.error(f"Failed to retrieve models: HTTP {response.status_code}")
                return []
    except Exception as e:
        logger.error(f"Error checking Ollama models: {e}")
        return []

async def test_ollama_connection(base_url="http://localhost:11434", model="llama3:8b"):
    """Test if Ollama is responding with a simple query"""
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello, are you working?"}],
                "stream": False
            }
            
            response = await client.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully connected to Ollama with model {model}")
                return True
            else:
                logger.error(f"Failed to connect to Ollama: HTTP {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"Error testing Ollama connection: {e}")
        return False

if __name__ == "__main__":
    # Run a simple test if this file is executed directly
    asyncio.run(check_ollama_models())
    asyncio.run(test_ollama_connection())