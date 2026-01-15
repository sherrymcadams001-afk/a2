#!/usr/bin/env python3
"""Phase 2: Duck.ai research for each Sportybet live match.

Reads the latest Sportybet scrape session from SQLite, then runs Duck.ai
research (Playwright) per match and stores research keyed by live match ID.

This stage uses NO external LLM for navigation/orchestration.
"""

import argparse
import asyncio
import logging
import sys
from typing import Dict, List, Optional

import sqlite3
from playwright.async_api import async_playwright

from research_db import store_research


logger = logging.getLogger(__name__)


def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_latest_sportybet_session_id(db_path: str) -> Optional[int]:
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id
            FROM scrape_sessions
            WHERE source = 'sportybet'
            ORDER BY started_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


def get_matches_for_session(db_path: str, session_id: int, limit: int | None = None) -> List[Dict]:
    with _connect(db_path) as conn:
        cur = conn.cursor()
        q = """
            SELECT id, sport, league, home_team, away_team, match_time, status, home_score, away_score
            FROM live_matches
            WHERE source = 'sportybet' AND session_id = ?
            ORDER BY sport, league, id
        """
        params = [session_id]
        if limit:
            q += " LIMIT ?"
            params.append(limit)
        cur.execute(q, params)
        return [dict(r) for r in cur.fetchall()]


def questions_for_sport(sport: str, home: str, away: str) -> List[str]:
    match = f"{home} vs {away}"
    sport_norm = (sport or '').strip().lower()

    if sport_norm == 'football':
        return [
            f"Recent form and head-to-head for {match}. Include league position, last 5 results, and last H2H scorelines.",
            f"Key players, injuries, suspensions, and team news for {match}. If unknown, say what’s unknown.",
            f"Prediction for {match}: likely winner, correct score lean, and over/under 2.5 goals. Give brief reasoning.",
        ]

    if sport_norm == 'basketball':
        return [
            f"Recent form and head-to-head for {match}. Include recent scores and any notable trends.",
            f"Key injuries/rotations and team news for {match}. If unknown, state uncertainty.",
            f"Prediction for {match}: likely winner and total points lean (over/under), with brief reasoning.",
        ]

    if sport_norm == 'tennis':
        return [
            f"Recent form and head-to-head for {match}. Include surface context if available.",
            f"Player fitness/injury notes for {match}. If unknown, say unknown.",
            f"Prediction for {match}: likely winner and sets/total games lean, with brief reasoning.",
        ]

    # Default fallback
    return [
        f"Recent form and head-to-head for {match}.",
        f"Key injuries/team news for {match}.",
        f"Prediction for {match} with brief reasoning.",
    ]


async def research_matches(db_path: str, headless: bool, limit: int | None = None):
    session_id = get_latest_sportybet_session_id(db_path)
    if not session_id:
        raise RuntimeError("No Sportybet scrape session found in DB")

    matches = get_matches_for_session(db_path, session_id=session_id, limit=limit)
    if not matches:
        raise RuntimeError(f"No live matches found for Sportybet session_id={session_id}. Did Phase 1 store session_id?")

    logger.info(f"Phase2: loaded {len(matches)} matches from session {session_id}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
        )

        # minimal stealth
        await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")

        page = await context.new_page()

        for idx, m in enumerate(matches, 1):
            match_id = m['id']
            sport = m.get('sport')
            home = m.get('home_team') or ''
            away = m.get('away_team') or ''
            league = m.get('league')

            logger.info(f"[{idx}/{len(matches)}] Researching #{match_id}: {sport} | {home} vs {away}")

            await page.goto("https://duck.ai", timeout=60000)
            await page.wait_for_load_state("networkidle")

            # consent
            try:
                agree_btn = page.locator('button:has-text("Agree and Continue"), button:has-text("Accept")').first
                if await agree_btn.is_visible(timeout=3000):
                    await agree_btn.click()
                    await asyncio.sleep(1)
            except Exception:
                pass

            input_selector = 'textarea[name="user-prompt"]'
            await page.wait_for_selector(input_selector, state="visible")

            for q in questions_for_sport(sport, home, away):
                await page.fill(input_selector, q)
                await page.press(input_selector, "Enter")
                await asyncio.sleep(12)

            content = await page.evaluate("() => document.body.innerText")
            store_research(
                home_team=home,
                away_team=away,
                raw_response=content,
                league=league,
                research_source='duck.ai',
                live_match_id=match_id,
            )

        await browser.close()


def main():
    parser = argparse.ArgumentParser(description="Phase2: Duck.ai research batch for Sportybet matches")
    parser.add_argument('--db', default='scraper_data.db', help='Path to SQLite db')
    parser.add_argument('--headless', action='store_true', help='Run headless (recommended for GitHub Actions)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of matches to research')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    asyncio.run(research_matches(args.db, headless=args.headless, limit=args.limit))


if __name__ == '__main__':
    main()
