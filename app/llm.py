from typing import Dict, List, Optional, Union
import json
import re
import ast
import httpx 
import uuid
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config import LLMSettings, config
from app.logger import logger
from app.schema import Message, TOOL_CHOICE_TYPE, ROLE_VALUES, TOOL_CHOICE_VALUES, ToolChoice


# Define exception classes to maintain compatibility with existing code
class OpenAIError(Exception): pass
class APIError(OpenAIError): pass
class AuthenticationError(OpenAIError): pass
class RateLimitError(OpenAIError): pass


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url
            
            # For Ollama, we don't need a client instance
            self.client = None

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """Format messages for LLM by converting them to OpenAI message format."""
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # If message is already a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # If message is a Message object, convert it to dict
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a prompt to the LLM and get the response."""
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            return await self._ollama_ask(messages, stream, temperature)

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise
            
    async def _ollama_ask(
        self,
        messages: List[dict],
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a request to Ollama and get the response."""
        temp = temperature or self.temperature
        
        # Ollama API expects specific structure
        ollama_payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temp,
                "num_predict": self.max_tokens,
            }
        }
        
        logger.info(f"Sending request to Ollama with model: {self.model}")
        
        async with httpx.AsyncClient() as client:
            if not stream:
                # Non-streaming request
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=ollama_payload,
                    timeout=120.0
                )
                response.raise_for_status()
                result = response.json()
                
                if "message" in result and "content" in result["message"]:
                    return result["message"]["content"]
                raise ValueError(f"Unexpected response format from Ollama: {result}")
            
            # Streaming response
            collected_messages = []
            
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=ollama_payload,
                timeout=180.0
            ) as response:
                response.raise_for_status()
                
                async for chunk in response.aiter_text():
                    try:
                        if chunk.strip():
                            chunk_data = json.loads(chunk)
                            if "message" in chunk_data and "content" in chunk_data["message"]:
                                chunk_text = chunk_data["message"]["content"]
                                collected_messages.append(chunk_text)
                                print(chunk_text, end="", flush=True)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Ollama chunk: {chunk}")
                        continue
                
                print()  # Newline after streaming
                full_response = "".join(collected_messages).strip()
                if not full_response:
                    raise ValueError("Empty response from streaming Ollama")
                return full_response

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
    self,
    messages: List[Union[dict, Message]],
    system_msgs: Optional[List[Union[dict, Message]]] = None,
    timeout: int = 300,
    tools: Optional[List[dict]] = None,
    tool_choice: str = ToolChoice.AUTO,  # Changed to use the value directly
    temperature: Optional[float] = None,
    **kwargs,
):
        """Ask LLM using functions/tools and return the response."""
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Handle tool calls
            if tools and tool_choice != ToolChoice.NONE:
                # Add tool definitions to the system message
                tool_descriptions = "\n\n".join([
                    f"Tool: {tool['function']['name']}\n"
                    f"Description: {tool['function']['description']}\n"
                    f"Parameters: {json.dumps(tool['function']['parameters'], indent=2)}"
                    for tool in tools
                ])
                
                # Prepare instructions for tool usage
                tool_instructions = f"""
                You have access to the following tools:
                
                {tool_descriptions}
                
                To use a tool, respond ONLY with a JSON object in the following format:
                
                ```json
                {{
                  "tool_calls": [
                    {{
                      "id": "call_{uuid.uuid4().hex[:8]}",
                      "type": "function",
                      "function": {{
                        "name": "tool_name",
                        "arguments": "{{\\\"param1\\\": \\\"value1\\\", \\\"param2\\\": \\\"value2\\\"}}"
                      }}
                    }}
                  ]
                }}
                ```
                
                Replace "tool_name" with the actual name of the tool you want to use, and provide the appropriate arguments.
                Always use double quotes for JSON keys and values. Escape quotes in the arguments value.
                DO NOT PROVIDE ANY TEXT OUTSIDE THE JSON OBJECT.
                """
                
                # Add tool instructions to the messages
                # Find or add system message
                has_system = False
                for i, msg in enumerate(messages):
                    if msg["role"] == "system":
                        messages[i]["content"] = f"{msg['content']}\n\n{tool_instructions}"
                        has_system = True
                        break
                
                if not has_system:
                    messages.insert(0, {"role": "system", "content": tool_instructions})
            
            # Get response from Ollama
            response_text = await self._ollama_ask(messages, False, temperature)
            
            # If tools were provided, check if the response is in JSON format
            if tools and tool_choice != ToolChoice.NONE:
                # Check if the response is a tool call (JSON format)
                try:
                    # Extract JSON if wrapped in code blocks
                    if "```json" in response_text:
                        json_text = response_text.split("```json")[1].split("```")[0].strip()
                        response_json = json.loads(json_text)
                    else:
                        # Try to parse directly as JSON
                        response_json = json.loads(response_text.strip())
                    
                    if "tool_calls" in response_json:
                        # Format tool calls for consistent processing
                        tool_calls = []
                        for tc in response_json["tool_calls"]:
                            # Ensure id and type exist
                            if "id" not in tc:
                                tc["id"] = f"call_{uuid.uuid4().hex[:8]}"
                            if "type" not in tc:
                                tc["type"] = "function"
                            
                            # Ensure function arguments are properly formatted
                            if isinstance(tc["function"]["arguments"], str):
                                try:
                                    # Try to parse arguments as JSON string
                                    json.loads(tc["function"]["arguments"])
                                except json.JSONDecodeError:
                                    # If not valid JSON, try to clean it up
                                    tc["function"]["arguments"] = json.dumps(
                                        {k.strip(): v for k, v in 
                                         [line.split(":", 1) for line in 
                                          tc["function"]["arguments"].strip().split("\n") 
                                          if ":" in line]}
                                    )
                            
                            tool_calls.append(tc)
                        
                        # Create a Message with tool calls
                        return Message(
                            role="assistant",
                            tool_calls=tool_calls
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse tool call JSON: {e}")
                    # Not a valid tool call JSON, treat as regular text response
            
            # Return as normal message
            return Message.assistant_message(content=response_text)

        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise