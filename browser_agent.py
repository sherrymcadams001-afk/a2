#!/usr/bin/env python3
"""
Browser-use agent to find top accumulator bet matches with least risk occurring within 3 hours.
Uses OLBG as main aggregator source with stealth mode enabled.
"""
import asyncio
import os
import json
from datetime import datetime, timedelta
from browser_use import Agent, Browser, Controller, ChatGoogle
from stealth import create_stealth_config

# Configuration constants
PRIMARY_SOURCE_URL = "olbg.com"
PRIMARY_SOURCE_NAME = "OLBG"


async def search_accumulator_bets():
    """
    Use browser-use agent to find top accumulator bet matches with least risk within 3 hours.
    Uses OLBG as the main aggregator source with stealth mode enabled.
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
    
    # Initialize stealth configuration
    stealth = create_stealth_config(
        confidence=0.7,
        stress=0.2,
        emotional_state="focused"
    )
    
    # Initialize browser with stealth mode
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
    
    # Define the search task for accumulator bets using OLBG
    task = f"""
    IMPORTANT: Use {PRIMARY_SOURCE_NAME} (Online Betting Guide) at {PRIMARY_SOURCE_URL} as your PRIMARY aggregator source for betting tips and match information.
    
    Step 1: Navigate to {PRIMARY_SOURCE_NAME}
    - Go to {PRIMARY_SOURCE_URL} and look for football/soccer betting tips
    - Focus on matches happening within the next 3 hours (from {current_time.strftime('%H:%M')} to {target_time.strftime('%H:%M')})
    
    Step 2: Gather Initial Data from {PRIMARY_SOURCE_NAME}
    - Look for accumulator tips or "acca" recommendations
    - Identify matches with highest confidence ratings from {PRIMARY_SOURCE_NAME} tipsters
    - Note the recommended bets and odds
    
    Step 3: Research Further (if needed)
    - Cross-reference {PRIMARY_SOURCE_NAME} tips with additional sources if necessary
    - Verify match times and current odds
    - Confirm team form and recent performance
    
    Step 4: Compile Final List
    For each recommended match, extract:
    1. Match name (Team A vs Team B)
    2. Start time (must be within 3 hours)
    3. Recommended bet type (e.g., Home Win, Over 2.5, BTTS)
    4. Odds from {PRIMARY_SOURCE_NAME}
    5. Risk assessment (Low/Medium/High based on {PRIMARY_SOURCE_NAME} tipster confidence)
    6. {PRIMARY_SOURCE_NAME} tipster rating/confidence percentage
    7. Source URL from {PRIMARY_SOURCE_NAME}
    
    Step 5: Prioritize and Save
    - Organize results by risk level (lowest risk first)
    - Focus on matches suitable for accumulator betting
    - Limit to top 5-7 matches with best risk/reward ratio
    - Ensure all matches are within the 3-hour window
    
    STEALTH MODE: Operate with natural human-like behavior patterns including realistic delays, mouse movements, and reading pauses.
    """
    
    try:
        # Apply stealth delay before starting
        await stealth.before_action("navigate")
        
        # Create and run the agent
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            controller=controller,
        )
        
        history = await agent.run()
        
        # Apply stealth delay after completion
        await stealth.after_action("navigate", success=True)
        
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
            "primary_source": PRIMARY_SOURCE_NAME,
            "primary_source_url": PRIMARY_SOURCE_URL,
            "time_window": {
                "from": current_time.isoformat(),
                "to": target_time.isoformat()
            },
            "agent_history": history_data,
            "stealth_enabled": True,
            "psychological_state": {
                "confidence": stealth.profile.confidence,
                "stress": stealth.profile.stress,
                "emotional_state": stealth.profile.emotional_state.value
            },
            "status": "completed"
        }
        
        # Save results to file (will be committed by GitHub Actions)
        output_file = f"accumulator_bets_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        print(f"Stealth mode: Active (State: {stealth.profile.emotional_state.value})")
        return results
        
    except Exception as e:
        await stealth.after_action("navigate", success=False)
        
        error_results = {
            "timestamp": current_time.isoformat(),
            "search_type": "accumulator_bets",
            "primary_source": PRIMARY_SOURCE_NAME,
            "primary_source_url": PRIMARY_SOURCE_URL,
            "status": "error",
            "error": str(e),
            "stealth_enabled": True
        }
        
        output_file = f"accumulator_bets_{current_time.strftime('%Y%m%d_%H%M%S')}_error.json"
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
