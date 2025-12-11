#!/usr/bin/env python3
"""
Browser-use agent to check Bing for "mstchrd" occurring in the next three hours.
"""
import asyncio
import os
import json
from datetime import datetime, timedelta
from browser_use import Agent, Browser, Controller
from langchain_openai import ChatOpenAI

async def search_bing_for_mstchrd():
    """
    Use browser-use agent to search Bing for "mstchrd" in the next three hours.
    Returns the search results and saves them to a file.
    """
    # Get current time and target time window
    current_time = datetime.now()
    target_time = current_time + timedelta(hours=3)
    
    # Validate OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it as a GitHub secret or environment variable."
        )
    
    # Initialize browser
    browser = Browser()
    
    # Initialize LLM (using OpenAI, but can be configured for other providers)
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key=api_key
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
