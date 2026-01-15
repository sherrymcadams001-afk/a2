import asyncio
import logging
import sys
import random
from datetime import datetime
from playwright.async_api import async_playwright

# Import database layer
from research_db import store_research, get_research, format_research_summary, get_predictions, format_prediction_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_match_config():
    """Get match to research - can be extended to read from queue/args."""
    return {
        'home_team': 'Cunupia FC',
        'away_team': 'Central FC',
        'league': 'Trinidad and Tobago Pro League',
        'match_date': None  # Will be filled if found in research
    }


def build_questions(home_team: str, away_team: str) -> list:
    """Build research questions for a match."""
    match = f"{home_team} vs {away_team}"
    return [
        f"What are the recent form and head-to-head statistics for {match} in Trinidad and Tobago football?",
        f"What are the key players, injuries, and team news for {match}?",
        f"Based on historical data, what is the likely outcome prediction for {match} including over/under goals?"
    ]

async def run_direct_research():
    config = get_match_config()
    home_team = config['home_team']
    away_team = config['away_team']
    match_display = f"{home_team} vs {away_team}"
    questions = build_questions(home_team, away_team)
    
    logger.info("=" * 80)
    logger.info(f"DIRECT PLAYWRIGHT RESEARCH - {match_display}")
    logger.info("=" * 80)

    async with async_playwright() as p:
        # Launch browser with stealth args
        browser = await p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                '--no-first-run',
                '--start-maximized',
            ]
        )
        
        # Create context with stealth settings
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
        )
        
        # Inject stealth scripts
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
            );
        """)

        page = await context.new_page()
        
        try:
            # 1. Navigate
            logger.info("Navigating to Duck.ai...")
            await page.goto("https://duck.ai", timeout=60000)
            await page.wait_for_load_state("networkidle")
            
            # 2. Handle Cookie/Terms
            try:
                # Try multiple potential selectors for the agree button
                agree_btn = page.locator('button:has-text("Agree and Continue"), button:has-text("Accept")').first
                if await agree_btn.is_visible(timeout=5000):
                    logger.info("Clicking 'Agree' button...")
                    await agree_btn.click()
                    await asyncio.sleep(1)
            except Exception as e:
                logger.info(f"No agree button found or error: {e}")

            # 3. Loop through questions
            input_selector = 'textarea[name="user-prompt"]'
            
            for i, question in enumerate(questions, 1):
                logger.info(f"Asking Question {i}: {question[:50]}...")
                
                # Wait for input to be ready
                await page.wait_for_selector(input_selector, state="visible")
                
                # Type and send
                await page.fill(input_selector, question)
                await page.press(input_selector, "Enter")
                
                # Wait for response generation
                logger.info("Waiting 15s for response...")
                await asyncio.sleep(15)
                
                # Optional: Check if generation is done (look for stop button disappearing or send button reappearing)
                # For now, fixed wait is safer as per previous success

            # 4. Extract Content
            logger.info("Extracting chat content...")
            # Get all text from the main chat container
            content = await page.evaluate("() => document.body.innerText")
            
            # 5. Save to Database
            logger.info("Saving research to database...")
            research_id = store_research(
                home_team=home_team,
                away_team=away_team,
                raw_response=content,
                league=config.get('league'),
                match_date=config.get('match_date'),
                research_source='duck.ai'
            )
            logger.info(f"✓ Research saved with ID: {research_id}")
            
            # 6. Display summary
            research_records = get_research(home_team=home_team, away_team=away_team, limit=1)
            if research_records:
                print("\n" + format_research_summary(research_records[0]))
            
            predictions = get_predictions(limit=1)
            if predictions:
                print("\n" + format_prediction_summary(predictions[0]))
            
        except Exception as e:
            logger.error(f"Error during research: {e}")
            # Take screenshot on error
            await page.screenshot(path="error_screenshot.png")
            logger.info("Saved error_screenshot.png")
        finally:
            logger.info("Closing browser...")
            await browser.close()

if __name__ == "__main__":
    asyncio.run(run_direct_research())
