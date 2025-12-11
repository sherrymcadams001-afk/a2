#!/usr/bin/env python3
"""
Browser-use agent to find top accumulator bet matches with least risk occurring within 3 hours.
Uses OLBG as main aggregator source with stealth mode enabled.
"""
import asyncio
import os
import json
import logging
import sys
from datetime import datetime, timedelta
from browser_use import Agent, Browser, Controller, ChatGoogle
from stealth import create_stealth_config

# Configuration constants
PRIMARY_SOURCE_URL = "olbg.com"
PRIMARY_SOURCE_NAME = "OLBG"
MAX_ACTIONS_TO_LOG = 10  # Limit number of actions logged to avoid excessive output

# Configure detailed logging
def setup_logging():
    """Set up logging with timestamped filename"""
    log_filename = f'agent_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename)
        ]
    )
    return logging.getLogger(__name__)


async def search_accumulator_bets():
    """
    Use browser-use agent to find top accumulator bet matches with least risk within 3 hours.
    Uses OLBG as the main aggregator source with stealth mode enabled.
    Returns the search results and saves them to a file.
    """
    # Set up logging with current timestamp
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("STARTING ACCUMULATOR BET SEARCH AGENT")
    logger.info("="*80)
    
    # Get current time and target time window
    current_time = datetime.now()
    target_time = current_time + timedelta(hours=3)
    logger.info(f"Time window: {current_time.strftime('%Y-%m-%d %H:%M:%S')} to {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set custom Gemini endpoint via environment variable
    os.environ['GOOGLE_API_BASE'] = 'https://key.ematthew477.workers.dev/v1beta'
    logger.info("Custom Gemini endpoint configured: https://key.ematthew477.workers.dev/v1beta")
    
    # Initialize stealth configuration
    logger.info("Initializing stealth mode...")
    stealth = create_stealth_config(
        confidence=0.7,
        stress=0.2,
        emotional_state="focused"
    )
    logger.info(f"✓ Stealth profile: confidence={stealth.profile.confidence}, "
                f"stress={stealth.profile.stress}, "
                f"state={stealth.profile.emotional_state.value}")
    
    # Initialize browser with stealth mode
    logger.info("Initializing browser...")
    browser = Browser()
    logger.info("✓ Browser initialized")
    
    # Initialize LLM using browser-use's native Gemini support
    logger.info("Connecting to Gemini endpoint...")
    llm = ChatGoogle(
        model='gemini-2.5-flash',
        api_key='custom-endpoint-key',  # Placeholder - custom endpoint handles auth
        temperature=0.0
    )
    logger.info("✓ Connected to Gemini via custom endpoint")
    
    # Create controller for browser actions
    logger.info("Setting up agent controller...")
    controller = Controller()
    logger.info("✓ Controller ready")
    
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
        logger.info("Applying pre-action stealth delay...")
        await stealth.before_action("navigate")
        logger.info("✓ Stealth delay applied")
        
        # Create and run the agent
        logger.info("-"*80)
        logger.info("AGENT TASK EXECUTION STARTING")
        logger.info("-"*80)
        logger.info(f"Primary source: {PRIMARY_SOURCE_NAME} ({PRIMARY_SOURCE_URL})")
        logger.info("Task steps:")
        logger.info("  1. Navigate to OLBG and find football/soccer tips")
        logger.info("  2. Gather accumulator recommendations with tipster ratings")
        logger.info("  3. Cross-reference data if needed")
        logger.info("  4. Extract match details (teams, odds, risk, start time)")
        logger.info("  5. Compile and prioritize top 5-7 low-risk matches")
        logger.info("-"*80)
        
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            controller=controller,
        )
        
        logger.info("Agent created. Beginning autonomous execution...")
        history = await agent.run()
        logger.info("✓ Agent execution completed")
        
        # Apply stealth delay after completion
        logger.info("Applying post-action stealth delay...")
        await stealth.after_action("navigate", success=True)
        logger.info("✓ Post-action delay applied")
        
        # Extract relevant information from history
        logger.info("-"*80)
        logger.info("PROCESSING AGENT RESULTS")
        logger.info("-"*80)
        
        # The browser-use agent returns a history object that may vary in structure
        # We'll safely extract what we can and preserve the raw data
        history_data = {
            "raw_history": str(history),
            "type": type(history).__name__
        }
        logger.info(f"History type: {type(history).__name__}")
        
        # Try to extract final result if available
        if hasattr(history, 'final_result'):
            try:
                final_result = str(history.final_result())
                history_data["final_result"] = final_result
                logger.info(f"Final result extracted ({len(final_result)} characters)")
            except Exception as e:
                logger.warning(f"Could not extract final_result: {e}")
        
        # Try to extract individual actions if iterable
        if hasattr(history, '__iter__') and not isinstance(history, str):
            try:
                actions = [str(item) for item in history]
                history_data["actions"] = actions
                total_actions = len(actions)
                logger.info(f"Extracted {total_actions} actions from history")
                
                # Log a limited number of actions for insight without excessive output
                actions_to_log = min(total_actions, MAX_ACTIONS_TO_LOG)
                if actions_to_log > 0:
                    logger.info(f"Logging first {actions_to_log} of {total_actions} actions:")
                    for idx in range(actions_to_log):
                        action = actions[idx]
                        action_preview = action[:200] + "..." if len(action) > 200 else action
                        logger.info(f"  Action {idx+1}: {action_preview}")
                    
                    if total_actions > MAX_ACTIONS_TO_LOG:
                        logger.info(f"  ... and {total_actions - MAX_ACTIONS_TO_LOG} more actions (see JSON for full history)")
            except Exception as e:
                logger.warning(f"Could not extract actions: {e}")
        
        # Prepare results
        logger.info("Compiling final results...")
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
        
        logger.info("="*80)
        logger.info("EXECUTION SUMMARY")
        logger.info("="*80)
        logger.info(f"✓ Results saved to: {output_file}")
        logger.info(f"✓ Status: {results['status']}")
        logger.info(f"✓ Stealth mode: Active")
        logger.info(f"✓ Psychological state: {stealth.profile.emotional_state.value}")
        logger.info(f"✓ Confidence level: {stealth.profile.confidence}")
        logger.info(f"✓ Stress level: {stealth.profile.stress}")
        logger.info("="*80)
        
        return results
        
    except Exception as e:
        logger.error("="*80)
        logger.error("EXECUTION FAILED")
        logger.error("="*80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.exception("Full traceback:")
        
        await stealth.after_action("navigate", success=False)
        
        error_results = {
            "timestamp": current_time.isoformat(),
            "search_type": "accumulator_bets",
            "primary_source": PRIMARY_SOURCE_NAME,
            "primary_source_url": PRIMARY_SOURCE_URL,
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "stealth_enabled": True
        }
        
        output_file = f"accumulator_bets_{current_time.strftime('%Y%m%d_%H%M%S')}_error.json"
        with open(output_file, 'w') as f:
            json.dump(error_results, f, indent=2)
        
        logger.error(f"Error details saved to {output_file}")
        logger.error("="*80)
        raise
    
    finally:
        # Clean up browser
        logger.info("Cleaning up browser resources...")
        await browser.close()
        logger.info("✓ Browser closed")
        logger.info("Agent execution finished.")

if __name__ == "__main__":
    asyncio.run(search_accumulator_bets())
