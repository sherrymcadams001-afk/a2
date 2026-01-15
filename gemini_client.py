"""
Gemini API Client - Uses google-genai library with mirror endpoint
Handles query generation and data extraction for match research
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Mirror endpoint - same as browser-use Phase 1
GEMINI_BASE_URL = "https://key.ematthew477.workers.dev"
MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-flash-lite"


@dataclass
class MatchInfo:
    """Match data for query generation"""
    number: int
    sport: str
    league: str
    team1: str
    team2: str
    odds: Dict[str, float]
    match_time: str


class GeminiClient:
    """
    Gemini API client using google-genai library with mirror endpoint
    Same approach as browser-use Phase 1
    Auto-fallback to lite model on rate limits
    """
    
    def __init__(self, base_url: str = GEMINI_BASE_URL, model: str = MODEL):
        self.base_url = base_url.rstrip('/')
        self.primary_model = model
        self.fallback_model = FALLBACK_MODEL
        self.model = model  # Current active model
        self._using_fallback = False
        self.client = genai.Client(
            api_key='DUMMY',
            http_options={'base_url': base_url}
        )
    
    def _switch_to_fallback(self):
        """Switch to fallback model for rest of run"""
        if not self._using_fallback:
            self._using_fallback = True
            self.model = self.fallback_model
            logger.warning(f"[GEMINI] Switching to fallback model: {self.fallback_model}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    async def generate(self, prompt: str, temperature: float = 0.7, retries: int = 3) -> Optional[str]:
        """
        Generate content from Gemini API
        Auto-falls back to lite model on rate limits
        
        Args:
            prompt: The prompt to send
            temperature: Creativity level (0.0-1.0)
            retries: Number of retries on rate limit
        
        Returns:
            Generated text or None if failed
        """
        for attempt in range(retries):
            try:
                # Use sync generate_content in thread (google-genai is sync)
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=8192,
                    )
                )
                
                # Extract text from response
                if response and response.text:
                    return response.text
                
                logger.warning(f"[GEMINI] Empty response")
                return None
                
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                    # Try fallback model immediately if not already using it
                    if not self._using_fallback:
                        self._switch_to_fallback()
                        # Retry immediately with fallback model
                        continue
                    
                    # Already on fallback, do regular retry with wait
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"[GEMINI] Rate limited on fallback, waiting {wait_time}s (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(wait_time)
                    continue
                
                logger.error(f"[GEMINI] Request failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                    continue
                return None
        
        logger.error("[GEMINI] All retries exhausted")
        return None
    
    async def generate_json(self, prompt: str, temperature: float = 0.3) -> Optional[Any]:
        """
        Generate JSON response from Gemini API
        
        Args:
            prompt: The prompt (should request JSON output)
            temperature: Lower for more deterministic JSON
        
        Returns:
            Parsed JSON or None if failed
        """
        response = await self.generate(prompt, temperature)
        
        if not response:
            return None
        
        # Try to extract JSON from response
        try:
            # Clean up common issues
            text = response.strip()
            
            # Remove markdown code blocks if present
            if text.startswith('```json'):
                text = text[7:]
            elif text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            
            text = text.strip()
            
            return json.loads(text)
            
        except json.JSONDecodeError as e:
            logger.error(f"[GEMINI] Failed to parse JSON: {e}")
            logger.debug(f"[GEMINI] Raw response: {response[:500]}")
            return None


# ============================================================================
# SPORT-SPECIFIC QUERY GENERATION PROMPTS (Optimized for Gemini 2.5 Pro)
# ============================================================================

FOOTBALL_QUERY_PROMPT = """<ROLE>
You are an elite sports intelligence analyst specializing in football/soccer data mining. Your task is to generate precision search queries that will yield maximum statistical intelligence for live betting research.
</ROLE>

<CONTEXT>
These are LIVE football matches currently in progress. The research goal is to gather predictive data: team form, head-to-head patterns, scoring trends, and any edge-giving statistics. Time is critical - queries must be efficient and high-yield.
</CONTEXT>

<STRATEGY>
For each match, construct queries that target:
1. **Current Form**: Last 5 match results, goals scored/conceded trends, home/away splits
2. **Head-to-Head**: Historical matchups between these exact teams, scoring patterns in previous meetings
3. **League Context**: Current standings, expected goals (xG), goal timing patterns (when do they score?)
4. **Live Intelligence**: If match is in progress, current stats (possession, shots, corners, cards)
5. **Edge Factors**: Injuries, suspensions, tactical patterns, set-piece efficiency

Query optimization rules:
- Use site-specific operators when beneficial: site:sofascore.com, site:flashscore.com, site:soccerway.com
- For youth leagues (U19, U20, Primavera), include age category explicitly
- For lesser-known leagues, include country name
- Prefer "vs" format for head-to-head: "Team A vs Team B"
- Include current year/season: "2024-25" or "2024"
</STRATEGY>

<MATCHES>
{matches}
</MATCHES>

<OUTPUT_FORMAT>
Return a JSON array. Each match gets 5 highly-targeted queries:
[
  {{
    "match_number": 1,
    "team1": "Exact Team Name",
    "team2": "Exact Team Name", 
    "league": "League Name",
    "queries": [
      "form and statistics query",
      "head to head history query",
      "league standings/context query",
      "live stats or recent goals query",
      "injury/news/edge factor query"
    ]
  }}
]
</OUTPUT_FORMAT>

Respond with ONLY the JSON array, no preamble or explanation."""


BASKETBALL_QUERY_PROMPT = """<ROLE>
You are an elite sports intelligence analyst specializing in basketball analytics. Your task is to generate precision search queries for live basketball betting research.
</ROLE>

<CONTEXT>
These are LIVE basketball matches, likely in Q1-Q3. Research goal: team scoring patterns, quarter-by-quarter trends, pace of play, and predictive statistics for in-game betting decisions.
</CONTEXT>

<STRATEGY>
For each match, construct queries targeting:
1. **Scoring Patterns**: Points per game, offensive/defensive ratings, pace, first quarter tendencies
2. **Head-to-Head**: Previous meetings, point differentials, which team covers spreads
3. **Quarter Analysis**: Q1/Q2/Q3/Q4 scoring averages, fast starters vs slow starters
4. **Key Metrics**: Rebounds, assists, turnovers, 3-point %, free throw %
5. **Situational**: Back-to-back games, home/away splits, recent form streak

Query optimization:
- Site operators: site:basketball-reference.com, site:flashscore.com, site:proballers.com, site:espn.com
- For international leagues (Turkey, Philippines, Australia NBL), include league name
- Use team abbreviations AND full names for better coverage
- Include "2024-25 season" or "2024" for current data
</STRATEGY>

<MATCHES>
{matches}
</MATCHES>

<OUTPUT_FORMAT>
Return a JSON array with 5 targeted queries per match:
[
  {{
    "match_number": 1,
    "team1": "Exact Team Name",
    "team2": "Exact Team Name",
    "league": "League Name",
    "queries": [
      "team scoring and offensive stats query",
      "head to head history query", 
      "quarter by quarter scoring patterns query",
      "recent form and results query",
      "key players and matchup analysis query"
    ]
  }}
]
</OUTPUT_FORMAT>

Respond with ONLY the JSON array, no preamble or explanation."""


TENNIS_QUERY_PROMPT = """<ROLE>
You are an elite sports intelligence analyst specializing in tennis analytics. Your task is to generate precision search queries for live tennis betting research.
</ROLE>

<CONTEXT>
These are LIVE tennis matches. Research goal: player form, head-to-head history, surface performance, serving/return statistics, and tournament context for in-play betting decisions.
</CONTEXT>

<STRATEGY>
For each match, construct queries targeting:
1. **Player Rankings & Form**: Current ranking, recent results, win/loss streak, current tournament run
2. **Head-to-Head**: Previous meetings between these players, surface-specific H2H
3. **Surface Performance**: Hard/clay/grass win rates, movement patterns on surface
4. **Service Stats**: First serve %, aces, double faults, break point conversion/save rates
5. **Tournament Context**: ITF/UTR/ATP/WTA level, draw size, seeding, path to current round

Query optimization:
- Site operators: site:flashscore.com, site:tennisabstract.com, site:atptour.com, site:wtatennis.com
- For ITF/UTR lower-tier: use flashscore or tennislive as primary sources
- Include full player names, not just surnames for unique identification
- For doubles: include both player names on each side
- Add tournament name and year for context
</STRATEGY>

<MATCHES>
{matches}
</MATCHES>

<OUTPUT_FORMAT>
Return a JSON array with 5 targeted queries per match:
[
  {{
    "match_number": 1,
    "player1": "Full Player Name",
    "player2": "Full Player Name",
    "tournament": "Tournament Name",
    "queries": [
      "player rankings and current form query",
      "head to head record query",
      "surface specific performance query",
      "serve and return statistics query",
      "tournament context and draw query"
    ]
  }}
]
</OUTPUT_FORMAT>

Respond with ONLY the JSON array, no preamble or explanation."""


# ============================================================================
# DATA EXTRACTION PROMPTS (Optimized for Gemini 2.5 Pro)
# ============================================================================

FOOTBALL_EXTRACTION_PROMPT = """<ROLE>
You are a sports data extraction specialist. Parse raw web content and extract structured football statistics with high precision.
</ROLE>

<TASK>
Extract all available football match data from the provided content for betting analysis.
</TASK>

<MATCH_CONTEXT>
Match: {team1} vs {team2}
League: {league}
</MATCH_CONTEXT>

<RAW_CONTENT>
{content}
</RAW_CONTENT>

<EXTRACTION_RULES>
1. Extract ONLY data explicitly present in the content - never invent or assume
2. Use null for any field where data is not available
3. Convert text-based form (WWLDW) to the actual string format
4. Normalize scores to "X-X" format
5. Include confidence notes in additional_info for uncertain extractions
</EXTRACTION_RULES>

<OUTPUT_SCHEMA>
{{
  "team1_stats": {{
    "goals_scored_total": null,
    "goals_conceded_total": null, 
    "goals_per_game_avg": null,
    "wins": null,
    "draws": null,
    "losses": null,
    "form_last_5": null,
    "league_position": null,
    "home_record": null,
    "clean_sheets": null
  }},
  "team2_stats": {{
    "goals_scored_total": null,
    "goals_conceded_total": null,
    "goals_per_game_avg": null,
    "wins": null,
    "draws": null,
    "losses": null,
    "form_last_5": null,
    "league_position": null,
    "away_record": null,
    "clean_sheets": null
  }},
  "head_to_head": {{
    "total_matches": null,
    "team1_wins": null,
    "draws": null,
    "team2_wins": null,
    "avg_goals_per_match": null,
    "both_teams_scored_pct": null,
    "recent_results": []
  }},
  "live_stats": {{
    "current_score": null,
    "minute": null,
    "possession": null,
    "shots_on_target": null,
    "corners": null,
    "yellow_cards": null
  }},
  "betting_relevant": {{
    "over_2_5_trend": null,
    "btts_trend": null,
    "first_half_goals_avg": null,
    "late_goals_pattern": null
  }},
  "additional_info": [],
  "data_quality": "high|medium|low",
  "source_recency": null
}}
</OUTPUT_SCHEMA>

Return ONLY valid JSON matching this schema."""


BASKETBALL_EXTRACTION_PROMPT = """<ROLE>
You are a sports data extraction specialist. Parse raw web content and extract structured basketball statistics with high precision.
</ROLE>

<TASK>
Extract all available basketball match data from the provided content for live betting analysis.
</TASK>

<MATCH_CONTEXT>
Match: {team1} vs {team2}
League: {league}
</MATCH_CONTEXT>

<RAW_CONTENT>
{content}
</RAW_CONTENT>

<EXTRACTION_RULES>
1. Extract ONLY data explicitly present - never invent statistics
2. Use null for unavailable fields
3. Quarter scores as array: [Q1, Q2, Q3, Q4] with null for unplayed quarters
4. Convert percentages to decimal (85% → 0.85) or keep as string "85%"
5. Note data freshness in additional_info
</EXTRACTION_RULES>

<OUTPUT_SCHEMA>
{{
  "team1_stats": {{
    "points_per_game": null,
    "rebounds_per_game": null,
    "assists_per_game": null,
    "field_goal_pct": null,
    "three_point_pct": null,
    "free_throw_pct": null,
    "wins": null,
    "losses": null,
    "current_streak": null,
    "home_record": null,
    "pace_rating": null
  }},
  "team2_stats": {{
    "points_per_game": null,
    "rebounds_per_game": null,
    "assists_per_game": null,
    "field_goal_pct": null,
    "three_point_pct": null,
    "free_throw_pct": null,
    "wins": null,
    "losses": null,
    "current_streak": null,
    "away_record": null,
    "pace_rating": null
  }},
  "head_to_head": {{
    "total_matches": null,
    "team1_wins": null,
    "team2_wins": null,
    "avg_total_points": null,
    "avg_point_diff": null,
    "recent_results": []
  }},
  "live_stats": {{
    "current_score": null,
    "quarter": null,
    "time_remaining": null,
    "quarter_scores": [],
    "lead_changes": null,
    "largest_lead": null
  }},
  "betting_relevant": {{
    "over_under_trend": null,
    "first_quarter_trend": null,
    "third_quarter_trend": null,
    "avg_total_points": null,
    "covers_spread_pct": null
  }},
  "additional_info": [],
  "data_quality": "high|medium|low",
  "source_recency": null
}}
</OUTPUT_SCHEMA>

Return ONLY valid JSON matching this schema."""


TENNIS_EXTRACTION_PROMPT = """<ROLE>
You are a sports data extraction specialist. Parse raw web content and extract structured tennis statistics with high precision.
</ROLE>

<TASK>
Extract all available tennis match data from the provided content for live betting analysis.
</TASK>

<MATCH_CONTEXT>
Match: {player1} vs {player2}
Tournament: {tournament}
</MATCH_CONTEXT>

<RAW_CONTENT>
{content}
</RAW_CONTENT>

<EXTRACTION_RULES>
1. Extract ONLY data explicitly present - never invent statistics
2. Use null for unavailable fields
3. Set scores as array: ["6-4", "7-6", ...] 
4. Rankings as integers, remove "ATP" or "WTA" prefix
5. Surface should be: "hard", "clay", "grass", or "indoor hard"
</EXTRACTION_RULES>

<OUTPUT_SCHEMA>
{{
  "player1_stats": {{
    "current_ranking": null,
    "ranking_trend": null,
    "ytd_wins": null,
    "ytd_losses": null,
    "current_surface_record": null,
    "titles_this_year": null,
    "first_serve_pct": null,
    "aces_per_match": null,
    "break_points_saved_pct": null
  }},
  "player2_stats": {{
    "current_ranking": null,
    "ranking_trend": null,
    "ytd_wins": null,
    "ytd_losses": null,
    "current_surface_record": null,
    "titles_this_year": null,
    "first_serve_pct": null,
    "aces_per_match": null,
    "break_points_saved_pct": null
  }},
  "head_to_head": {{
    "total_matches": null,
    "player1_wins": null,
    "player2_wins": null,
    "surface_specific_h2h": null,
    "recent_results": []
  }},
  "live_stats": {{
    "current_score": null,
    "set_scores": [],
    "current_set": null,
    "current_game": null,
    "serving": null,
    "break_points": null
  }},
  "betting_relevant": {{
    "tiebreak_record": null,
    "deciding_set_record": null,
    "retirement_history": null,
    "fatigue_factor": null,
    "surface_preference": null
  }},
  "additional_info": [],
  "data_quality": "high|medium|low",
  "source_recency": null
}}
</OUTPUT_SCHEMA>

Return ONLY valid JSON matching this schema."""


class QueryGenerator:
    """Generate sport-specific search queries using Gemini"""
    
    def __init__(self, client: GeminiClient):
        self.client = client
    
    def _format_matches(self, matches: List[MatchInfo]) -> str:
        """Format matches for prompt"""
        lines = []
        for m in matches:
            odds_str = ", ".join(f"{k}: {v}" for k, v in m.odds.items())
            lines.append(f"{m.number}. {m.team1} vs {m.team2} | League: {m.league} | Odds: {odds_str}")
        return "\n".join(lines)
    
    async def generate_football_queries(self, matches: List[MatchInfo]) -> List[Dict]:
        """Generate queries for football matches"""
        if not matches:
            return []
        
        prompt = FOOTBALL_QUERY_PROMPT.format(matches=self._format_matches(matches))
        result = await self.client.generate_json(prompt)
        
        if result:
            logger.info(f"[QUERY] Generated queries for {len(matches)} football matches")
            return result
        return []
    
    async def generate_basketball_queries(self, matches: List[MatchInfo]) -> List[Dict]:
        """Generate queries for basketball matches"""
        if not matches:
            return []
        
        prompt = BASKETBALL_QUERY_PROMPT.format(matches=self._format_matches(matches))
        result = await self.client.generate_json(prompt)
        
        if result:
            logger.info(f"[QUERY] Generated queries for {len(matches)} basketball matches")
            return result
        return []
    
    async def generate_tennis_queries(self, matches: List[MatchInfo]) -> List[Dict]:
        """Generate queries for tennis matches"""
        if not matches:
            return []
        
        prompt = TENNIS_QUERY_PROMPT.format(matches=self._format_matches(matches))
        result = await self.client.generate_json(prompt)
        
        if result:
            logger.info(f"[QUERY] Generated queries for {len(matches)} tennis matches")
            return result
        return []


class DataExtractor:
    """Extract structured data from raw HTML using Gemini"""
    
    def __init__(self, client: GeminiClient):
        self.client = client
    
    async def extract_football_data(
        self,
        team1: str,
        team2: str,
        league: str,
        content: str
    ) -> Optional[Dict]:
        """Extract football match data"""
        # Truncate content to avoid token limits
        content = content[:15000]
        
        prompt = FOOTBALL_EXTRACTION_PROMPT.format(
            team1=team1,
            team2=team2,
            league=league,
            content=content
        )
        
        return await self.client.generate_json(prompt)
    
    async def extract_basketball_data(
        self,
        team1: str,
        team2: str,
        league: str,
        content: str
    ) -> Optional[Dict]:
        """Extract basketball match data"""
        content = content[:15000]
        
        prompt = BASKETBALL_EXTRACTION_PROMPT.format(
            team1=team1,
            team2=team2,
            league=league,
            content=content
        )
        
        return await self.client.generate_json(prompt)
    
    async def extract_tennis_data(
        self,
        player1: str,
        player2: str,
        tournament: str,
        content: str
    ) -> Optional[Dict]:
        """Extract tennis match data"""
        content = content[:15000]
        
        prompt = TENNIS_EXTRACTION_PROMPT.format(
            player1=player1,
            player2=player2,
            tournament=tournament,
            content=content
        )
        
        return await self.client.generate_json(prompt)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def test_client():
    """Test the Gemini client"""
    async with GeminiClient() as client:
        response = await client.generate("Say 'API working' in exactly 2 words.")
        print(f"Response: {response}")
        return response is not None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(test_client())
    print(f"Test passed: {result}")
