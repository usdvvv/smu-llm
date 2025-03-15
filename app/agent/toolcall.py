import json

from typing import Any, List, Literal, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message, ToolCall, TOOL_CHOICE_TYPE, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection
from app.llm import LLM


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "fusion"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None
    
    # Store the original question
    original_question: str = ""
    # Store tool results for summary
    tool_results: List[str] = Field(default_factory=list)
    # Final summary
    final_summary: str = ""

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        # Get response with tool options
        try:
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=[Message.system_message(self.system_prompt)]
                if self.system_prompt
                else None,
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
            self.tool_calls = response.tool_calls

            # Log response info
            logger.info(f"✨ {self.name}'s thoughts: {response.content}")
            logger.info(
                f"🛠️ {self.name} selected {len(response.tool_calls) if response.tool_calls else 0} tools to use"
            )
            if response.tool_calls:
                logger.info(
                    f"🧰 Tools being prepared: {[call.function.name for call in response.tool_calls]}"
                )

            try:
                # Handle different tool_choices modes
                if self.tool_choices == ToolChoice.NONE:
                    if response.tool_calls:
                        logger.warning(
                            f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                        )
                    if response.content:
                        self.memory.add_message(Message.assistant_message(response.content))
                        return True
                    return False

                # Create and add assistant message
                assistant_msg = (
                    Message.from_tool_calls(
                        content=response.content, tool_calls=self.tool_calls
                    )
                    if self.tool_calls
                    else Message.assistant_message(response.content)
                )
                self.memory.add_message(assistant_msg)

                if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                    return True  # Will be handled in act()

                # For 'auto' mode, continue with content if no commands but content exists
                if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                    return bool(response.content)

                return bool(self.tool_calls)
            except Exception as e:
                logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
                self.memory.add_message(
                    Message.assistant_message(
                        f"Error encountered while processing: {str(e)}"
                    )
                )
                return False
        except Exception as e:
            # If we hit an error with the LLM, try to generate a summary and terminate
            logger.error(f"LLM communication error: {e}")
            
            # Generate a simple answer to the original question
            await self._generate_fallback_summary()
            
            # Mark the agent as finished
            self.state = AgentState.FINISHED
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            try:
                result = await self.execute_tool(command)

                if self.max_observe:
                    result = result[: self.max_observe]

                logger.info(
                    f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
                )
                
                # Store tool results for summary
                self.tool_results.append(f"{command.function.name}: {result}")

                # Add tool response to memory
                tool_msg = Message.tool_message(
                    content=result, tool_call_id=command.id, name=command.function.name
                )
                self.memory.add_message(tool_msg)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing tool {command.function.name}: {e}")
                results.append(f"Error executing tool {command.function.name}: {e}")

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            # Ensure 'action' is included for browser_use tool
            if name == "browser_use" and "action" not in args:
                # Default to navigate if no action specified
                args['action'] = 'navigate'
                if 'url' not in args:
                    args['url'] = 'https://example.com'  # Provide a default URL
                logger.warning(f"No action specified for browser_use tool. Defaulting to navigate to {args['url']}")

            # Execute the tool - pass all parameters as a single dictionary
            logger.info(f"🔧 Activating tool: '{name}'...")
            
            # Use tool_input parameter to pass all arguments as a single dictionary
            result = await self.available_tools.execute(name=name, tool_input=args)

            # Format result for display
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            # Handle special tools like `finish`
            await self._handle_special_tool(name=name, result=result, args=args)

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, args=None, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if name.lower() == "terminate" and self._should_finish_execution(name=name, result=result, **kwargs):
            # Generate summary before finishing
            logger.info(f"🏁 Special tool '{name}' called - generating final summary before completing the task!")
            
            # Generate summary
            summary = await self._generate_summary(args)
            self.final_summary = summary
            
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    async def _generate_summary(self, args=None) -> str:
        """Generate a final summary of all findings and information gathered"""
        try:
            # Get the original question (first user message)
            original_question = self.original_question
            if not original_question:
                for msg in self.memory.messages:
                    if msg.role == "user" and msg.content:
                        original_question = msg.content
                        break
            
            # Create a summary prompt - keep it simple to avoid context issues
            summary_prompt = f"""
            Please provide a clear answer to this question: "{original_question}"
            
            Keep your answer direct, informative, and under 200 words.
            """
            
            # Get summary from LLM
            llm = LLM()
            summary = await llm.ask(
                messages=[{"role": "user", "content": summary_prompt}],
                stream=False
            )
            
            # Add summary to memory
            self.memory.add_message(
                Message.assistant_message(f"FINAL ANSWER: {summary}")
            )
            
            # Format final output
            status = args.get("status", "success") if args else "success"
            
            # Print the summary to the console for user visibility
            print("\n" + "="*60)
            print(" "*20 + "FINAL ANSWER")
            print("="*60)
            print(summary)
            print("="*60 + "\n")
            
            return f"The interaction has been completed with status: {status}\n\n===== FINAL ANSWER =====\n\n{summary}\n\n======================="
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            await self._generate_fallback_summary()
            return f"The interaction has been completed."

    async def _generate_fallback_summary(self) -> None:
        """Generate a fallback summary if the normal summary generation fails"""
        try:
            # Use the original question to get a basic answer
            original_question = self.original_question
            fallback_prompt = f"{original_question} Keep your answer simple and under 150 words."
            
            # Get response directly with a simpler approach
            llm = LLM("default")  # Use default model
            summary = await llm.ask(
                messages=[{"role": "user", "content": fallback_prompt}],
                stream=False
            )
            
            # Print the fallback answer
            print("\n" + "="*60)
            print(" "*20 + "FINAL ANSWER")
            print("="*60)
            print(summary)
            print("="*60 + "\n")
            
            self.final_summary = f"===== FINAL ANSWER =====\n\n{summary}\n\n======================="
        except Exception as e:
            logger.error(f"Fallback summary generation failed: {e}")
            print("\n" + "="*60)
            print("COULD NOT GENERATE ANSWER")
            print("="*60)
            print(f"Sorry, I couldn't generate an answer to your question: '{self.original_question}'\nPlease try asking again or rephrasing your question.")
            print("="*60 + "\n")
            
            self.final_summary = f"===== FINAL ANSWER =====\n\nSorry, I couldn't generate an answer to your question: '{self.original_question}'\nPlease try asking again or rephrasing your question.\n\n======================="

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def run(self, question: str) -> str:
        """Run the agent to answer a question, with added summary at the end"""
        # Store the original question
        self.original_question = question
        self.tool_results = []
        
        try:
            # Call the parent run method
            result = await super().run(question)
            
            # If we never generated a summary (e.g., terminated early), do it now
            if not self.final_summary:
                self.final_summary = await self._generate_summary(None)
                
            return self.final_summary
        except Exception as e:
            logger.error(f"Error in run: {e}")
            await self._generate_fallback_summary()
            return self.final_summary