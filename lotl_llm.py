"""
LotL LLM Wrapper - browser-use 0.11.0 compatible BaseChatModel implementation.

This module provides a ChatLotL class that implements browser-use's native
BaseChatModel Protocol (NOT LangChain), routing all LLM calls through the
LotL Controller API which proxies through logged-in Google AI Studio.

Usage:
    from lotl_llm import ChatLotL, is_lotl_available, get_lotl_llm
    
    if is_lotl_available():
        llm = ChatLotL()
        # Use with browser-use Agent
        from browser_use import Agent
        agent = Agent(task="...", llm=llm)
"""

import json
import re
from typing import TypeVar, overload, Any

from pydantic import BaseModel

from browser_use.llm.messages import (
    BaseMessage,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ContentPartTextParam,
    ContentPartImageParam,
)
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

from lotl_client import LotLClient

T = TypeVar('T', bound=BaseModel)


def is_lotl_available(test_request: bool = True, test_timeout: float = 15.0) -> bool:
    """
    Check if LotL Controller is available and responsive.
    
    Args:
        test_request: If True, also send a quick test request to verify responsiveness
        test_timeout: Timeout for the test request (seconds)
    """
    try:
        client = LotLClient(timeout=test_timeout)
        if not client.is_available():
            return False
        
        if test_request:
            # Send a quick test to verify the AI Studio session is working
            response = client.chat("Reply with just: OK")
            return bool(response and len(response) > 0)
        
        return True
    except Exception as e:
        print(f"[LotL] Not available: {e}")
        return False


def get_lotl_llm(timeout: float = 300.0) -> "ChatLotL":
    """Get a ChatLotL instance with optional custom timeout."""
    return ChatLotL(timeout=timeout)


class ChatLotL:
    """
    browser-use 0.11.0 compatible LLM that routes through LotL Controller.
    
    Implements the BaseChatModel Protocol from browser_use.llm.base:
    - model: str attribute
    - provider property
    - name property  
    - ainvoke(messages, output_format) -> ChatInvokeCompletion
    """
    
    # Class-level attribute required by Protocol
    _verified_api_keys: bool = True
    
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        base_url: str = "http://localhost:3000",
        timeout: float = 300.0,
    ):
        """
        Initialize ChatLotL.
        
        Args:
            model: Model name (for logging only, actual model is set in AI Studio)
            base_url: LotL Controller URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = LotLClient(base_url=base_url, timeout=timeout)
    
    @property
    def provider(self) -> str:
        """Return the provider name."""
        return "lotl"
    
    @property
    def name(self) -> str:
        """Return the model name."""
        return self.model
    
    @property
    def model_name(self) -> str:
        """Legacy support - return model name."""
        return self.model
    
    def _messages_to_prompt(self, messages: list[BaseMessage]) -> tuple[str, list[str]]:
        """
        Convert browser-use messages to a single prompt string and extract images.
        
        NOTE: System messages are SKIPPED because the system prompt is already
        hardcoded in the AI Studio UI session. We only send the user content
        as a single combined request.
        
        Returns:
            Tuple of (prompt_text, list_of_base64_images)
        """
        parts = []
        images = []
        
        def clean_text(text: str) -> str:
            """Remove redundant markers that are already in system prompt."""
            # Remove closing browser_state tag (opening is fine for context)
            text = re.sub(r'</browser_state>\s*', '', text)
            # Remove "Current screenshot:" label (image is sent separately)
            text = re.sub(r'Current screenshot:\s*', '', text)
            return text.strip()
        
        for msg in messages:
            # SKIP SystemMessage - already hardcoded in AI Studio UI
            if isinstance(msg, SystemMessage):
                continue
                    
            elif isinstance(msg, UserMessage):
                if isinstance(msg.content, str):
                    cleaned = clean_text(msg.content)
                    if cleaned:
                        parts.append(cleaned)
                elif isinstance(msg.content, list):
                    text_parts = []
                    for p in msg.content:
                        if isinstance(p, ContentPartTextParam):
                            cleaned = clean_text(p.text)
                            if cleaned:
                                text_parts.append(cleaned)
                        elif isinstance(p, ContentPartImageParam):
                            # Extract image data URL
                            images.append(p.image_url.url)
                    if text_parts:
                        parts.append("\n".join(text_parts))
                        
            elif isinstance(msg, AssistantMessage):
                # Include assistant responses for conversation context (tool calls, etc.)
                if msg.content:
                    if isinstance(msg.content, str):
                        parts.append(f"[Previous Response]\n{msg.content}")
                    elif isinstance(msg.content, list):
                        text = "\n".join(
                            p.text for p in msg.content 
                            if hasattr(p, 'text')
                        )
                        if text:
                            parts.append(f"[Previous Response]\n{text}")
        
        # Combine into single prompt
        return "\n\n".join(parts), images
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        # Try to find JSON in code blocks first
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # Try to find raw JSON object or array
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            return json_match.group(1).strip()
        
        return text.strip()
    
    def _parse_structured_output(self, text: str, output_format: type[T]) -> T:
        """Parse LLM response into structured Pydantic model."""
        json_str = self._extract_json(text)
        
        try:
            data = json.loads(json_str)
            return output_format.model_validate(data)
        except json.JSONDecodeError as e:
            # Try to be more lenient - sometimes LLM adds extra text
            raise ValueError(f"Failed to parse JSON from response: {e}\nResponse: {text[:500]}")
        except Exception as e:
            raise ValueError(f"Failed to validate output format: {e}\nResponse: {text[:500]}")
    
    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: None = None
    ) -> ChatInvokeCompletion[str]: ...
    
    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T]
    ) -> ChatInvokeCompletion[T]: ...
    
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """
        Invoke the LLM with messages and optional structured output.
        
        Args:
            messages: List of browser-use message objects
            output_format: Optional Pydantic model class for structured output
            
        Returns:
            ChatInvokeCompletion with the response
        """
        import logging
        logger = logging.getLogger("lotl_llm")
        
        # Convert messages to prompt
        prompt, images = self._messages_to_prompt(messages)
        
        logger.info(f"[LotL] Sending request with {len(prompt)} chars, {len(images)} images")
        
        # NOTE: Output format schema is NOT appended here because it's already
        # hardcoded in the AI Studio system prompt. Adding it again causes confusion.
        
        # Call LotL Controller
        try:
            response_text = await self._client.achat(
                prompt=prompt,
                images=images if images else None,
                timeout=self._timeout
            )
            logger.info(f"[LotL] Got response: {len(response_text)} chars")
            # Log first 200 chars at INFO level to debug format issues
            logger.info(f"[LotL] Response preview: {response_text[:200]}...")
        except Exception as e:
            logger.error(f"[LotL] Request failed: {e}")
            raise
        
        # Create usage info (estimated since LotL doesn't return token counts)
        usage = ChatInvokeUsage(
            prompt_tokens=len(prompt) // 4,  # Rough estimate
            prompt_cached_tokens=None,
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=len(images) * 1000 if images else None,
            completion_tokens=len(response_text) // 4,
            total_tokens=(len(prompt) + len(response_text)) // 4,
        )
        
        # Parse output
        if output_format is not None:
            completion = self._parse_structured_output(response_text, output_format)
            return ChatInvokeCompletion[T](
                completion=completion,
                usage=usage,
                stop_reason="end_turn",
            )
        else:
            return ChatInvokeCompletion[str](
                completion=response_text,
                usage=usage,
                stop_reason="end_turn",
            )
    
    def __repr__(self) -> str:
        return f"ChatLotL(model='{self.model}', provider='lotl')"


# For Pydantic compatibility (allows ChatLotL to be used in Pydantic models)
def __get_pydantic_core_schema__(cls, source_type, handler):
    from pydantic_core import core_schema
    return core_schema.any_schema()


if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing ChatLotL...")
        print(f"LotL available: {is_lotl_available()}")
        
        if is_lotl_available():
            llm = ChatLotL()
            print(f"LLM: {llm}")
            print(f"Provider: {llm.provider}")
            print(f"Model: {llm.model}")
            
            # Test simple invoke
            from browser_use.llm.messages import UserMessage
            messages = [UserMessage(content="Say 'Hello from ChatLotL!' and nothing else.")]
            result = await llm.ainvoke(messages)
            print(f"Response: {result.completion}")
            print(f"Usage: {result.usage}")
        else:
            print("LotL Controller not available. Start it first.")
    
    asyncio.run(test())
