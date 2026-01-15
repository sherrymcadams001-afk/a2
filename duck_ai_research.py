#!/usr/bin/env python3
"""
Duck.ai research agent - Uses browser-use to query Duck.ai for match research.
Extracts full chat session and logs DOM IDs used at each step.
"""
import asyncio
import os
import logging
import sys
from datetime import datetime
from browser_use import Agent, Browser, Controller, ChatGoogle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# The match to research
MATCH = "Cunupia FC vs Central FC"

# Questions to ask Duck.ai
QUESTIONS = [
    f"What are the recent form and head-to-head statistics for {MATCH} in Trinidad and Tobago football?",
    f"What are the key players, injuries, and team news for {MATCH}?",
    f"Based on historical data, what is the likely outcome prediction for {MATCH} including over/under goals?"
]


async def run_duck_ai_research():
    """Run browser-use agent to query Duck.ai and extract research"""
    
    logger.info("=" * 80)
    logger.info(f"DUCK.AI RESEARCH AGENT - {MATCH}")
    logger.info("=" * 80)
    
    # Set Gemini endpoint
    os.environ['GOOGLE_API_BASE'] = 'https://key.ematthew477.workers.dev'
    
    # Import BrowserProfile for headless control
    from browser_use.browser import BrowserProfile, BrowserSession
    
    # Initialize browser with explicit headless=False for visibility
    browser = BrowserSession(
        browser_profile=BrowserProfile(
            headless=False,  # Show browser window
            disable_security=True,  # Allow cross-origin
        ),
        keep_alive=True
    )
    await browser.start()
    logger.info("✓ Browser started (HEADFUL mode)")
    
    # Initialize primary LLM
    llm = ChatGoogle(
        model='gemini-2.5-flash',
        temperature=0.0,
        api_key='DUMMY',
        http_options={'base_url': 'https://key.ematthew477.workers.dev'},
    )
    logger.info("✓ Primary LLM: gemini-2.5-flash")
    
    # Initialize fallback LLM for rate limits
    fallback_llm = ChatGoogle(
        model='gemini-2.5-flash-lite',
        temperature=0.0,
        api_key='DUMMY',
        http_options={'base_url': 'https://key.ematthew477.workers.dev'},
    )
    logger.info("✓ Fallback LLM: gemini-2.5-flash-lite")
    
    controller = Controller()
    
    # Build the comprehensive task prompt - simplified and using KNOWN selectors
    task = f"""
You are a research agent using Duck.ai to research a football match: {MATCH}

## KNOWN SELECTORS (Use these directly):
- Chat Input: textarea[name="user-prompt"]
- Cookie Button: Button with text "Agree and Continue"

## TASK:
1. Go to https://duck.ai
2. Click "Agree and Continue" if visible.
3. Click the chat input (textarea[name="user-prompt"]).
4. Type: "{QUESTIONS[0]}"
5. Press Enter, then WAIT 10 seconds for the answer to generate.
6. Type: "{QUESTIONS[1]}"
7. Press Enter, then WAIT 10 seconds.
8. Type: "{QUESTIONS[2]}"
9. Press Enter, then WAIT 10 seconds.
10. Extract ALL text content from the chat.

IMPORTANT:
- Do not get stuck waiting for "generation to finish" indicators. Just wait 10 seconds and proceed.
- Use the known selectors provided above.
"""
    
    try:
        # Create and run agent with fallback LLM for rate limits
        agent = Agent(
            task=task,
            llm=llm,
            fallback_llm=fallback_llm,
            browser_session=browser,
            controller=controller,
            max_actions_per_step=10,
            use_vision=True,  # Enabled as requested to help see state
        )
        
        logger.info("Starting Duck.ai research...")
        logger.info(f"Questions to ask: {len(QUESTIONS)}")
        logger.info("Vision: ENABLED")
        
        result = await agent.run(max_steps=50)  # Reduced max steps
        
        # Extract the final result
        final_output = result.final_result() if hasattr(result, 'final_result') else str(result)
        
        # Save to results.md
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open('results.md', 'a', encoding='utf-8') as f:
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write(f"# DUCK.AI RESEARCH SESSION - {MATCH}\n")
            f.write(f"**Timestamp:** {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            f.write("## Questions Asked:\n")
            for i, q in enumerate(QUESTIONS, 1):
                f.write(f"{i}. {q}\n")
            f.write("\n")
            f.write("## Agent Output:\n\n")
            f.write(str(final_output))
            f.write("\n\n")
            f.write("-" * 80 + "\n")
            f.write("## DOM ELEMENTS USED (from agent actions):\n\n")
            
            # Extract action history if available
            if hasattr(result, 'action_history'):
                for i, action in enumerate(result.action_history()):
                    f.write(f"- Step {i+1}: {action}\n")
            elif hasattr(result, 'history'):
                for i, item in enumerate(result.history):
                    f.write(f"- Step {i+1}: {item}\n")
            else:
                f.write("(Action history extraction - see agent logs for detailed DOM interactions)\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info("=" * 80)
        logger.info("RESEARCH COMPLETE")
        logger.info("=" * 80)
        logger.info("✓ Results appended to results.md")
        
        return final_output
        
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't close browser on error - let user see the results
        logger.warning("Browser kept open due to error. Close manually when done.")
        logger.warning("Press Ctrl+C to exit and close browser.")
        try:
            # Wait indefinitely until user presses Ctrl+C
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            pass
        finally:
            await browser.kill()
            logger.info("Browser closed")
        return None
    
    # Only close browser on successful completion
    await browser.kill()
    logger.info("Browser closed")


if __name__ == "__main__":
    asyncio.run(run_duck_ai_research())
