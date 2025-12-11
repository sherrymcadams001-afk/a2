#!/usr/bin/env python3
"""
Browser-use agent to find top accumulator bet matches with least risk occurring within 3 hours.
"""
import asyncio
import os
import json
from datetime import datetime, timedelta
from browser_use import Agent, Browser, Controller, ChatGoogle


async def search_accumulator_bets():
    """
    Use browser-use agent to find top accumulator bet matches with least risk within 3 hours.
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
    
    # Initialize LLM using browser-use's native Gemini support with custom endpoint
    llm = ChatGoogle(
        model='gemini-2.5-flash-preview-05-20',
        api_key=api_key,
        base_url='https://key.ematthew477.workers.dev/v1beta',
        temperature=0.0
    )
    
    # Create controller for browser actions
    controller = Controller()
    
    # Define the search task for accumulator bets
    task = f"""
    Go to Bing.com and search for top accumulator bet matches with the least risk occurring within the next 3 hours.
    
    Your goal is to find football/soccer matches that are:
    1. Starting within the next 3 hours (from {current_time.strftime('%H:%M')} to {target_time.strftime('%H:%M')})
    2. Have low risk factors (favorites with high probability of winning)
    3. Suitable for accumulator betting
    
    For each match found, extract:
    1. Match name (Team A vs Team B)
    2. Start time
    3. Recommended bet type (e.g., Home Win, Over/Under)
    4. Odds if available
    5. Risk assessment (Low/Medium/High)
    6. Source URL
    
    Organize the results by risk level (lowest risk first) and limit to top 5 matches.
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
            "search_type": "accumulator_bets",
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
            "search_type": "accumulator_bets",
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
    asyncio.run(search_accumulator_bets())
