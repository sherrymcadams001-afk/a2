#!/usr/bin/env python3
"""
Browser-use agent for multi-purpose web scraping and data extraction.
Uses stealth browser automation with human-like behavior.

Supports two modes:
- WORKFLOW MODE (default): Uses workflow JSON for structured multi-step execution
- LEGACY MODE: Uses task_prompt.txt for single-task execution (set USE_WORKFLOW=false)

Available workflows:
- workflow.json: OLBG accumulator bets scraper
- workflows/sportybet_scraper.json: Sportybet live matches scraper

Set WORKFLOW_FILE env var to specify which workflow to use.
"""
import asyncio
import os
import json
import logging
import sys
from datetime import datetime, timedelta

# =============================================================================
# CRITICAL: Disable browser-use telemetry and cloud sync BEFORE imports
# These interfere with LLM request timing and cause sync issues
# =============================================================================
os.environ['ANONYMIZED_TELEMETRY'] = 'false'
os.environ['BROWSER_USE_CLOUD_SYNC'] = 'false'

from browser_use import Agent, Browser, Controller, ChatGoogle

# Configuration constants
PRIMARY_SOURCE_URL = "olbg.com"
PRIMARY_SOURCE_NAME = "OLBG"
MAX_ACTIONS_TO_LOG = 10
USE_WORKFLOW = os.environ.get('USE_WORKFLOW', 'true').lower() == 'true'
WORKFLOW_FILE = os.environ.get('WORKFLOW_FILE', 'workflow.json')

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
    
    # Stealth is now built into browser-use internals (mouse, keyboard, scroll actions)
    logger.info("✓ Stealth mode BUILT-IN: human-like delays for all browser actions")
    
    # Initialize browser with keep_alive=True for multi-step workflows
    logger.info("Initializing browser...")
    browser = Browser(keep_alive=True)
    await browser.start()  # Explicitly start browser session
    logger.info("✓ Browser initialized")
    
    # =========================================================================
    # LLM INITIALIZATION - LotL Controller only on VM, otherwise Gemini API
    # =========================================================================
    # LotL Controller routes prompts through a logged-in AI Studio session,
    # bypassing API quotas and rate limits. Only use when running on VM.
    
    from lotl_llm import is_lotl_available, get_lotl_llm
    
    llm = None
    fallback_llm = None
    
    # Detect if running on VM (Windows Administrator user)
    is_vm = os.getenv('USERNAME', '').lower() == 'administrator' or \
            os.path.exists(r'C:\Users\Administrator')
    
    # Skip test request, just check if controller is running (faster startup)
    if is_vm and is_lotl_available(test_request=False):
        logger.info("✓ VM detected + LotL Controller available - using local AI Studio session")
        llm = get_lotl_llm(timeout=600.0)  # 10 minutes - Gemini needs time for complex browser states
        fallback_llm = llm  # LotL is already robust, use same for fallback
    else:
        if is_vm:
            logger.info("VM detected but LotL Controller not available - using Gemini API mirror")
        else:
            logger.info("Not on VM - using Gemini API mirror directly")
        # Set custom Gemini endpoint via environment variable
        os.environ['GOOGLE_API_BASE'] = 'https://key.ematthew477.workers.dev'
        logger.info("Custom Gemini endpoint configured: https://key.ematthew477.workers.dev")
        
        # NOTE: The underlying `google-genai` client enforces that *some* auth input is provided
        # (api_key or Vertex config) even if you're pointing it at a custom mirror base_url.
        # The mirror endpoint accepts requests without a real key, so we pass a non-empty
        # placeholder to satisfy client initialization.
        llm = ChatGoogle(
            model='gemini-2.5-flash',
            temperature=0.0,
            api_key='DUMMY',
            http_options={'base_url': 'https://key.ematthew477.workers.dev'},
        )
        logger.info("✓ Connected to Gemini via custom endpoint")
        
        # Initialize Fallback LLM (Flash Lite)
        fallback_llm = ChatGoogle(
            model='gemini-2.0-flash-lite-preview-02-05',
            temperature=0.0,
            api_key='DUMMY',
            http_options={'base_url': 'https://key.ematthew477.workers.dev'},
        )
        logger.info("✓ Fallback LLM (Flash Lite) initialized")
    
    # Create controller for browser actions
    logger.info("Setting up agent controller...")
    controller = Controller()
    logger.info("✓ Controller ready")
    
    # Choose execution mode
    if USE_WORKFLOW:
        return await run_workflow_mode(logger, llm, browser, controller, current_time, target_time, fallback_llm)
    else:
        return await run_legacy_mode(logger, llm, browser, controller, current_time, target_time)


async def run_workflow_mode(logger, llm, browser, controller, current_time, target_time, fallback_llm=None):
    """Execute using structured workflow with validation and transforms"""
    from workflow_runner import WorkflowRunner
    
    logger.info("-"*80)
    logger.info("WORKFLOW MODE: Structured multi-step execution")
    logger.info("-"*80)
    
    # Resolve workflow file path
    workflow_file = WORKFLOW_FILE
    if not os.path.isabs(workflow_file):
        workflow_file = os.path.join(os.path.dirname(__file__), workflow_file)
    
    if not os.path.exists(workflow_file):
        logger.error(f"Workflow file not found: {workflow_file}")
        return None
    
    # Initialize workflow runner with source detection
    # Pass llm, controller, browser so runner can create fresh agents per step
    source = "sportybet" if "sportybet" in workflow_file.lower() else "olbg"
    runner = WorkflowRunner(
        workflow_file, 
        browser=browser, 
        llm=llm, 
        controller=controller, 
        source=source,
        fallback_llm=fallback_llm
    )
    
    # Set dynamic variables
    runner.set_variable('current_time', current_time.strftime('%H:%M'))
    runner.set_variable('target_time', target_time.strftime('%H:%M'))
    
    logger.info(f"✓ Workflow loaded from {workflow_file}")
    logger.info(f"  Steps: {len(runner.workflow.get('steps', []))}")
    
    try:
        # Execute workflow
        results = await runner.run()
        
        # Save results
        workflow_name = runner.workflow.get('name', 'workflow')
        primary_source_name = 'Sportybet' if source == 'sportybet' else PRIMARY_SOURCE_NAME
        primary_source_url = 'sportybet.com' if source == 'sportybet' else PRIMARY_SOURCE_URL

        output = {
            "timestamp": current_time.isoformat(),
            "search_type": workflow_name,
            "mode": "workflow",
            "primary_source": primary_source_name,
            "primary_source_url": primary_source_url,
            "time_window": {
                "from": current_time.isoformat(),
                "to": target_time.isoformat()
            },
            "results": results,
            "stealth_enabled": True,
            "stealth_mode": "built-in",
            "status": "completed"
        }
        
        output_file = f"{workflow_name}_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info("="*80)
        logger.info("PHASE 1 WORKFLOW COMPLETE")
        logger.info("="*80)
        logger.info(f"✓ Results saved to: {output_file}")
        logger.info(f"✓ Matches found: {results.get('count', 0)}")
        
        # ============ PHASE 2: Duck.ai Research ============
        phase2_enabled = os.environ.get('PHASE2_ENABLED', 'true').lower() == 'true'
        phase2_limit = int(os.environ.get('PHASE2_LIMIT', '5'))
        
        if phase2_enabled and source == 'sportybet' and results.get('count', 0) > 0:
            logger.info("")
            logger.info("="*80)
            logger.info("STARTING PHASE 2: Duck.ai Match Research")
            logger.info("="*80)
            logger.info(f"Researching up to {phase2_limit} matches via Duck.ai...")
            
            try:
                # Close Phase 1 browser before starting Phase 2 (clean state)
                await browser.kill()
                
                # Import and run Phase 2
                from phase2_duckai_batch import research_matches
                db_path = os.path.join(os.path.dirname(__file__), 'scraper_data.db')
                
                await research_matches(
                    db_path=db_path,
                    headless=False,  # Match Phase 1 visibility
                    limit=phase2_limit
                )
                
                logger.info("")
                logger.info("="*80)
                logger.info("PHASE 2 COMPLETE")
                logger.info("="*80)
                logger.info(f"✓ Research results stored in database (match_research table)")
                
            except Exception as e:
                logger.error(f"Phase 2 failed: {e}")
                import traceback
                traceback.print_exc()
        elif not phase2_enabled:
            logger.info("Phase 2 disabled (set PHASE2_ENABLED=true to enable)")
        # ====================================================
        
        return output
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise
    finally:
        # Clean up browser session (may already be killed if Phase 2 ran)
        try:
            await browser.kill()
        except:
            pass  # Already closed


async def run_legacy_mode(logger, llm, browser, controller, current_time, target_time):
    """Execute using single task prompt (legacy mode)"""
    # Load task prompt from external file
    task_file = os.path.join(os.path.dirname(__file__), 'task_prompt.txt')
    with open(task_file, 'r') as f:
        task_template = f.read()
    
    # Format the task with dynamic values
    task = task_template.format(
        PRIMARY_SOURCE_NAME=PRIMARY_SOURCE_NAME,
        PRIMARY_SOURCE_URL=PRIMARY_SOURCE_URL,
        current_time=current_time.strftime('%H:%M'),
        target_time=target_time.strftime('%H:%M')
    )
    logger.info(f"✓ Task prompt loaded from {task_file}")
    
    try:
        # Create and run the agent
        logger.info("-"*80)
        logger.info("LEGACY MODE: Single task execution")
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
            llm_timeout=180,  # Increase timeout for LotL with slower models
        )
        
        logger.info("Agent created. Beginning autonomous execution...")
        history = await agent.run(max_steps=10000)
        logger.info("✓ Agent execution completed")
        
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
            "stealth_mode": "built-in (mouse, keyboard, scroll delays)",
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
        logger.info(f"✓ Stealth mode: Built-in (human-like delays for all actions)")
        logger.info("="*80)
        
        return results
        
    except Exception as e:
        logger.error("="*80)
        logger.error("EXECUTION FAILED")
        logger.error("="*80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.exception("Full traceback:")
        
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
        try:
            if hasattr(browser, 'close'):
                await browser.close()
                logger.info("✓ Browser closed")
            else:
                logger.warning(f"Browser object {type(browser)} has no close() method")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
            
        logger.info("Agent execution finished.")

if __name__ == "__main__":
    asyncio.run(search_accumulator_bets())
