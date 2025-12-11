#!/usr/bin/env python3
"""
Browser-use agent to check Bing for "mstchrd" occurring in the next three hours.
"""
import asyncio
import os
import json
import requests
from datetime import datetime, timedelta
from browser_use import Agent, Browser, Controller
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class GeminiChatModel(BaseChatModel):
    """Custom LangChain chat model for Gemini API."""
    
    api_url: str = "https://key.ematthew477.workers.dev/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
    api_key: str = ""
    temperature: float = 0.0
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response from the Gemini API."""
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg.content}]
                })
            elif isinstance(msg, AIMessage):
                contents.append({
                    "role": "model",
                    "parts": [{"text": msg.content}]
                })
        
        # Prepare the payload
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192
            }
        }
        
        # Make the API request
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        # Return in LangChain format
        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self):
        """Return identifier for the LLM type."""
        return "gemini-chat"
    
    @property
    def _identifying_params(self):
        """Return identifying parameters."""
        return {
            "api_url": self.api_url,
            "temperature": self.temperature
        }


async def search_bing_for_mstchrd():
    """
    Use browser-use agent to search Bing for "mstchrd" in the next three hours.
    Returns the search results and saves them to a file.
    """
    # Get current time and target time window
    current_time = datetime.now()
    target_time = current_time + timedelta(hours=3)
    
    # Validate Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please set it as a GitHub secret or environment variable."
        )
    
    # Initialize browser
    browser = Browser()
    
    # Initialize LLM (using Gemini API)
    llm = GeminiChatModel(
        api_key=api_key,
        temperature=0.0
    )
    
    # Create controller for browser actions
    controller = Controller()
    
    # Define the search task
    task = f"""
    Go to Bing.com and search for "mstchrd" with time filter for the next 3 hours.
    Find all relevant results and extract the following information for each result:
    1. Title of each result
    2. URL of each result
    3. Brief description/snippet
    4. Timestamp if available
    
    Collect all results and note the total count.
    """
    
    try:
        # Create and run the agent
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            controller=controller,
        )
        
        history = await agent.run()
        
        # Extract relevant information from history
        # The browser-use agent returns a history object that may vary in structure
        # We'll safely extract what we can and preserve the raw data
        history_data = {
            "raw_history": str(history),
            "type": type(history).__name__
        }
        
        # Try to extract final result if available
        if hasattr(history, 'final_result'):
            try:
                history_data["final_result"] = str(history.final_result())
            except Exception:
                pass
        
        # Try to extract individual actions if iterable
        if hasattr(history, '__iter__') and not isinstance(history, str):
            try:
                history_data["actions"] = [str(item) for item in history]
            except Exception:
                pass
        
        # Prepare results
        results = {
            "timestamp": current_time.isoformat(),
            "search_term": "mstchrd",
            "time_window": {
                "from": current_time.isoformat(),
                "to": target_time.isoformat()
            },
            "agent_history": history_data,
            "status": "completed"
        }
        
        # Save results to file
        output_file = f"results_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        return results
        
    except Exception as e:
        error_results = {
            "timestamp": current_time.isoformat(),
            "search_term": "mstchrd",
            "status": "error",
            "error": str(e)
        }
        
        output_file = f"results_{current_time.strftime('%Y%m%d_%H%M%S')}_error.json"
        with open(output_file, 'w') as f:
            json.dump(error_results, f, indent=2)
        
        print(f"Error occurred: {e}")
        print(f"Error details saved to {output_file}")
        raise
    
    finally:
        # Clean up browser
        await browser.close()

if __name__ == "__main__":
    asyncio.run(search_bing_for_mstchrd())
