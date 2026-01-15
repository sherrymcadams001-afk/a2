"""
Phase 2: Stealth Match Research Scraper
Asynchronously scrapes statistical data for all matches from results.md
Uses global stealth with Google->Bing fallback
"""

import asyncio
import re
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from pathlib import Path
from bs4 import BeautifulSoup

import database as db
from http_stealth import get_stealth_client, close_stealth_client, StealthHTTPClient

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("match_research")
MAX_CONCURRENT_SCRAPES = 5  # Limit concurrent requests


class SportType(Enum):
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    TENNIS = "tennis"
    UNKNOWN = "unknown"


# ============================================================================
# MATCH DATA STRUCTURES
# ============================================================================

@dataclass
class Match:
    """Parsed match from results.md"""
    number: int
    sport: SportType
    league: str
    teams: str
    team1: str
    team2: str
    match_time: str
    odds: Dict[str, float] = field(default_factory=dict)
    match_type: str = "Real"
    raw_text: str = ""


@dataclass
class MatchResearch:
    """Research data collected for a match"""
    match: Match
    team1_stats: Dict[str, Any] = field(default_factory=dict)
    team2_stats: Dict[str, Any] = field(default_factory=dict)
    head_to_head: List[str] = field(default_factory=list)
    recent_form: Dict[str, List[str]] = field(default_factory=dict)
    live_stats: Dict[str, Any] = field(default_factory=dict)
    injuries_suspensions: Dict[str, List[str]] = field(default_factory=dict)
    weather: str = ""
    venue: str = ""
    additional_info: List[str] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)


# ============================================================================
# RESULTS.MD PARSER
# ============================================================================

class ResultsParser:
    """Parse results.md into structured Match objects"""
    
    def __init__(self, filepath: str = "results.md"):
        self.filepath = filepath
        self.matches: List[Match] = []
    
    def parse(self) -> List[Match]:
        """Parse results.md and return list of matches"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.matches = []
        match_number = 0
        
        # Split into sections
        sections = self._split_sections(content)
        
        for section_name, section_content in sections.items():
            sport = self._detect_sport(section_name)
            matches = self._parse_section(section_content, sport)
            
            for match in matches:
                match_number += 1
                match.number = match_number
                self.matches.append(match)
        
        logger.info(f"[PARSER] Parsed {len(self.matches)} matches from results.md")
        return self.matches
    
    def _split_sections(self, content: str) -> Dict[str, str]:
        """Split content into sport sections"""
        sections = {}
        current_section = "Football"
        current_content = []
        
        for line in content.split('\n'):
            # Detect section headers
            if line.startswith('# ') and 'Basketball' in line:
                sections[current_section] = '\n'.join(current_content)
                current_section = "Basketball"
                current_content = []
            elif line.startswith('# ') and 'Tennis' in line:
                sections[current_section] = '\n'.join(current_content)
                current_section = "Tennis"
                current_content = []
            elif 'ITF' in line or 'UTR' in line:
                # Tennis matches detected
                if current_section != "Tennis":
                    sections[current_section] = '\n'.join(current_content)
                    current_section = "Tennis"
                    current_content = []
                current_content.append(line)
            else:
                current_content.append(line)
        
        sections[current_section] = '\n'.join(current_content)
        return sections
    
    def _detect_sport(self, section_name: str) -> SportType:
        """Detect sport type from section name"""
        section_lower = section_name.lower()
        if 'basketball' in section_lower:
            return SportType.BASKETBALL
        elif 'tennis' in section_lower:
            return SportType.TENNIS
        else:
            return SportType.FOOTBALL
    
    def _parse_section(self, content: str, sport: SportType) -> List[Match]:
        """Parse matches from a section"""
        if sport == SportType.TENNIS:
            return self._parse_tennis_matches(content)
        elif sport == SportType.BASKETBALL:
            return self._parse_basketball_matches(content)
        else:
            return self._parse_football_matches(content)
    
    def _parse_football_matches(self, content: str) -> List[Match]:
        """Parse football match entries"""
        matches = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and headers
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Check if this is a league name (no colon, not Match Time, Teams, Odds)
            if not any(x in line for x in ['Match Time:', 'Teams:', 'Odds:', 'Match Type:']):
                league = line
                match_data = {'league': league, 'sport': SportType.FOOTBALL}
                
                # Parse following lines for match details
                j = i + 1
                while j < len(lines) and j < i + 6:
                    detail_line = lines[j].strip()
                    
                    if 'Match Time:' in detail_line:
                        match_data['match_time'] = detail_line.replace('Match Time:', '').strip()
                    elif 'Teams:' in detail_line:
                        teams = detail_line.replace('Teams:', '').strip()
                        match_data['teams'] = teams
                        if ' vs ' in teams:
                            parts = teams.split(' vs ')
                            match_data['team1'] = parts[0].strip()
                            match_data['team2'] = parts[1].strip() if len(parts) > 1 else ""
                    elif 'Odds:' in detail_line:
                        odds_str = detail_line.replace('Odds:', '').strip()
                        match_data['odds'] = self._parse_odds(odds_str)
                    elif 'Match Type:' in detail_line:
                        match_data['match_type'] = detail_line.replace('Match Type:', '').strip()
                    elif detail_line and not any(x in detail_line for x in ['Match Time:', 'Teams:', 'Odds:', 'Match Type:']):
                        # This is likely the next league, stop parsing
                        break
                    
                    j += 1
                
                # Create match if we have essential data
                if 'teams' in match_data:
                    match = Match(
                        number=0,
                        sport=SportType.FOOTBALL,
                        league=match_data.get('league', 'Unknown'),
                        teams=match_data.get('teams', ''),
                        team1=match_data.get('team1', ''),
                        team2=match_data.get('team2', ''),
                        match_time=match_data.get('match_time', ''),
                        odds=match_data.get('odds', {}),
                        match_type=match_data.get('match_type', 'Real'),
                        raw_text='\n'.join(lines[i:j])
                    )
                    matches.append(match)
                    i = j
                    continue
            
            i += 1
        
        return matches
    
    def _parse_basketball_matches(self, content: str) -> List[Match]:
        """Parse basketball match entries"""
        matches = []
        lines = content.split('\n')
        
        i = 0
        current_league = ""
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Check for league name with colon at end
            if line.endswith(':') and 'Match Time' not in line and 'Teams' not in line:
                current_league = line.rstrip(':')
                i += 1
                continue
            
            # Check for match entry
            if 'Match Time:' in line or 'Teams:' in line:
                match_data = {'league': current_league, 'sport': SportType.BASKETBALL}
                
                # Parse this and following lines
                j = i
                while j < len(lines) and j < i + 8:
                    detail_line = lines[j].strip()
                    
                    if 'Match Time:' in detail_line:
                        match_data['match_time'] = detail_line.replace('Match Time:', '').strip()
                    elif 'Teams:' in detail_line:
                        teams = detail_line.replace('Teams:', '').strip()
                        match_data['teams'] = teams
                        if ' vs ' in teams:
                            parts = teams.split(' vs ')
                            match_data['team1'] = parts[0].strip()
                            match_data['team2'] = parts[1].strip() if len(parts) > 1 else ""
                    elif 'Odds:' in detail_line:
                        odds_str = detail_line.replace('Odds:', '').strip()
                        match_data['odds'] = self._parse_basketball_odds(odds_str)
                    elif 'Current Quarter:' in detail_line:
                        match_data['quarter'] = detail_line.replace('Current Quarter:', '').strip()
                    elif detail_line.endswith(':') and 'Match Time' not in detail_line:
                        # Next league
                        break
                    
                    j += 1
                
                if 'teams' in match_data:
                    match = Match(
                        number=0,
                        sport=SportType.BASKETBALL,
                        league=match_data.get('league', 'Unknown'),
                        teams=match_data.get('teams', ''),
                        team1=match_data.get('team1', ''),
                        team2=match_data.get('team2', ''),
                        match_time=match_data.get('match_time', ''),
                        odds=match_data.get('odds', {}),
                        match_type='Real',
                        raw_text='\n'.join(lines[i:j])
                    )
                    matches.append(match)
                    i = j
                    continue
            
            i += 1
        
        return matches
    
    def _parse_tennis_matches(self, content: str) -> List[Match]:
        """Parse tennis match entries"""
        matches = []
        lines = content.split('\n')
        
        current_tournament = ""
        current_players = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Tournament name (ITF, UTR, etc.)
            if line.startswith('ITF') or line.startswith('UTR'):
                # Save previous match if exists
                if current_tournament and current_players:
                    for player_pair in self._group_tennis_players(current_players):
                        raw_players = '\n'.join([f"{p[0]}: {p[1]}" for p in current_players])
                        match = Match(
                            number=0,
                            sport=SportType.TENNIS,
                            league=current_tournament.rstrip(':'),
                            teams=f"{player_pair[0]} vs {player_pair[1]}",
                            team1=player_pair[0],
                            team2=player_pair[1],
                            match_time="Live",
                            odds={'player1': player_pair[2], 'player2': player_pair[3]},
                            match_type='Real',
                            raw_text=f"{current_tournament}\n{raw_players}"
                        )
                        matches.append(match)
                
                current_tournament = line
                current_players = []
            
            # Player line with odds
            elif line.startswith('*'):
                # Format: *   Player Name: 1.50
                player_match = re.match(r'\*\s+(.+?):\s*([\d.]+)', line)
                if player_match:
                    current_players.append((player_match.group(1), float(player_match.group(2))))
        
        # Process last tournament
        if current_tournament and current_players:
            for player_pair in self._group_tennis_players(current_players):
                raw_players = '\n'.join([f"{p[0]}: {p[1]}" for p in current_players])
                match = Match(
                    number=0,
                    sport=SportType.TENNIS,
                    league=current_tournament.rstrip(':'),
                    teams=f"{player_pair[0]} vs {player_pair[1]}",
                    team1=player_pair[0],
                    team2=player_pair[1],
                    match_time="Live",
                    odds={'player1': player_pair[2], 'player2': player_pair[3]},
                    match_type='Real',
                    raw_text=f"{current_tournament}\n{raw_players}"
                )
                matches.append(match)
        
        return matches
    
    def _group_tennis_players(self, players: List[tuple]) -> List[tuple]:
        """Group tennis players into pairs for matches"""
        grouped = []
        for i in range(0, len(players) - 1, 2):
            p1_name, p1_odds = players[i]
            p2_name, p2_odds = players[i + 1] if i + 1 < len(players) else ("Unknown", 0.0)
            grouped.append((p1_name, p2_name, p1_odds, p2_odds))
        return grouped
    
    def _parse_odds(self, odds_str: str) -> Dict[str, float]:
        """Parse odds string for football"""
        odds = {}
        parts = odds_str.replace(',', '').split()
        try:
            if len(parts) >= 3:
                odds['home'] = float(parts[0])
                odds['draw'] = float(parts[1])
                odds['away'] = float(parts[2])
        except ValueError:
            pass
        return odds
    
    def _parse_basketball_odds(self, odds_str: str) -> Dict[str, float]:
        """Parse odds string for basketball"""
        odds = {}
        # Format: Team1 1.10, Team2 7.25
        parts = re.findall(r'[\d.]+', odds_str)
        try:
            if len(parts) >= 2:
                odds['team1'] = float(parts[0])
                odds['team2'] = float(parts[1])
        except ValueError:
            pass
        return odds


# ============================================================================
# SPORT-SPECIFIC SCRAPERS
# ============================================================================

class BaseScraper:
    """Base scraper with common functionality"""
    
    def __init__(self, client: StealthHTTPClient):
        self.client = client
    
    async def search_and_extract(self, query: str) -> List[str]:
        """Search for query and extract relevant snippets"""
        html = await self.client.search(query)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        snippets = []
        
        # Extract search result snippets
        for result in soup.select('.g, .b_algo, .result'):
            text = result.get_text(separator=' ', strip=True)
            if len(text) > 50:
                snippets.append(text[:500])
        
        return snippets[:5]  # Top 5 results
    
    async def get_page_content(self, url: str) -> Optional[str]:
        """Fetch and extract text from a page"""
        html = await self.client.get(url, referer_type="direct")
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove scripts and styles
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        
        return soup.get_text(separator='\n', strip=True)


class FootballScraper(BaseScraper):
    """Football-specific stats scraper"""
    
    async def research_match(self, match: Match) -> MatchResearch:
        """Gather comprehensive research for a football match"""
        research = MatchResearch(match=match)
        
        tasks = [
            self._get_team_stats(match.team1, research, 'team1'),
            self._get_team_stats(match.team2, research, 'team2'),
            self._get_head_to_head(match, research),
            self._get_recent_form(match, research),
            self._get_live_stats(match, research),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        research.sources_used.append(f"Search engine: {self.client.stealth.current_engine}")
        return research
    
    async def _get_team_stats(self, team: str, research: MatchResearch, key: str):
        """Get team statistics"""
        queries = [
            f"{team} football statistics 2024",
            f"{team} goals scored conceded season",
        ]
        
        stats = {}
        for query in queries:
            snippets = await self.search_and_extract(query)
            if snippets:
                stats['search_results'] = snippets
                research.sources_used.append(query)
                break
        
        if key == 'team1':
            research.team1_stats = stats
        else:
            research.team2_stats = stats
    
    async def _get_head_to_head(self, match: Match, research: MatchResearch):
        """Get head-to-head history"""
        query = f"{match.team1} vs {match.team2} head to head history"
        snippets = await self.search_and_extract(query)
        research.head_to_head = snippets
        research.sources_used.append(query)
    
    async def _get_recent_form(self, match: Match, research: MatchResearch):
        """Get recent form for both teams"""
        for team in [match.team1, match.team2]:
            query = f"{team} recent results last 5 matches"
            snippets = await self.search_and_extract(query)
            research.recent_form[team] = snippets
            research.sources_used.append(query)
    
    async def _get_live_stats(self, match: Match, research: MatchResearch):
        """Get live match statistics if available"""
        query = f"{match.team1} vs {match.team2} live score stats today"
        snippets = await self.search_and_extract(query)
        research.live_stats = {'search_results': snippets}
        research.sources_used.append(query)


class BasketballScraper(BaseScraper):
    """Basketball-specific stats scraper"""
    
    async def research_match(self, match: Match) -> MatchResearch:
        """Gather comprehensive research for a basketball match"""
        research = MatchResearch(match=match)
        
        tasks = [
            self._get_team_stats(match.team1, research, 'team1'),
            self._get_team_stats(match.team2, research, 'team2'),
            self._get_head_to_head(match, research),
            self._get_live_stats(match, research),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        research.sources_used.append(f"Search engine: {self.client.stealth.current_engine}")
        return research
    
    async def _get_team_stats(self, team: str, research: MatchResearch, key: str):
        """Get basketball team statistics"""
        queries = [
            f"{team} basketball statistics 2024",
            f"{team} points per game rebounds assists",
        ]
        
        stats = {}
        for query in queries:
            snippets = await self.search_and_extract(query)
            if snippets:
                stats['search_results'] = snippets
                research.sources_used.append(query)
                break
        
        if key == 'team1':
            research.team1_stats = stats
        else:
            research.team2_stats = stats
    
    async def _get_head_to_head(self, match: Match, research: MatchResearch):
        """Get head-to-head history"""
        query = f"{match.team1} vs {match.team2} basketball head to head"
        snippets = await self.search_and_extract(query)
        research.head_to_head = snippets
        research.sources_used.append(query)
    
    async def _get_live_stats(self, match: Match, research: MatchResearch):
        """Get live basketball statistics"""
        query = f"{match.team1} vs {match.team2} live score basketball"
        snippets = await self.search_and_extract(query)
        research.live_stats = {'search_results': snippets}
        research.sources_used.append(query)


class TennisScraper(BaseScraper):
    """Tennis-specific stats scraper"""
    
    async def research_match(self, match: Match) -> MatchResearch:
        """Gather comprehensive research for a tennis match"""
        research = MatchResearch(match=match)
        
        tasks = [
            self._get_player_stats(match.team1, research, 'team1'),
            self._get_player_stats(match.team2, research, 'team2'),
            self._get_head_to_head(match, research),
            self._get_surface_stats(match, research),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        research.sources_used.append(f"Search engine: {self.client.stealth.current_engine}")
        return research
    
    async def _get_player_stats(self, player: str, research: MatchResearch, key: str):
        """Get tennis player statistics"""
        queries = [
            f"{player} tennis statistics ranking 2024",
            f"{player} serve percentage win rate",
        ]
        
        stats = {}
        for query in queries:
            snippets = await self.search_and_extract(query)
            if snippets:
                stats['search_results'] = snippets
                research.sources_used.append(query)
                break
        
        if key == 'team1':
            research.team1_stats = stats
        else:
            research.team2_stats = stats
    
    async def _get_head_to_head(self, match: Match, research: MatchResearch):
        """Get head-to-head history"""
        query = f"{match.team1} vs {match.team2} tennis head to head"
        snippets = await self.search_and_extract(query)
        research.head_to_head = snippets
        research.sources_used.append(query)
    
    async def _get_surface_stats(self, match: Match, research: MatchResearch):
        """Get surface-specific performance"""
        # Extract tournament surface from league name
        surface = "hard"  # default
        if 'clay' in match.league.lower():
            surface = "clay"
        elif 'grass' in match.league.lower():
            surface = "grass"
        
        for player in [match.team1, match.team2]:
            query = f"{player} tennis {surface} court record"
            snippets = await self.search_and_extract(query)
            research.additional_info.extend(snippets[:2])
            research.sources_used.append(query)


# ============================================================================
# OUTPUT HANDLER
# ============================================================================

class ResearchOutputHandler:
    """Save research to numbered files"""
    
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def save_research(self, research: MatchResearch) -> str:
        """Save research to numbered file"""
        filename = f"match_{research.match.number:03d}.md"
        filepath = self.output_dir / filename
        
        content = self._format_research(research)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"[OUTPUT] Saved research to {filename}")
        return str(filepath)
    
    def _format_research(self, research: MatchResearch) -> str:
        """Format research as markdown"""
        m = research.match
        
        lines = [
            f"# Match {m.number}: {m.teams}",
            "",
            "## Match Information",
            f"- **Sport**: {m.sport.value.title()}",
            f"- **League**: {m.league}",
            f"- **Match Time**: {m.match_time}",
            f"- **Odds**: {m.odds}",
            "",
            "---",
            "",
            f"## {m.team1} Statistics",
            "",
        ]
        
        if research.team1_stats.get('search_results'):
            for snippet in research.team1_stats['search_results']:
                lines.append(f"> {snippet[:300]}...")
                lines.append("")
        else:
            lines.append("*No statistics found*")
            lines.append("")
        
        lines.extend([
            "---",
            "",
            f"## {m.team2} Statistics",
            "",
        ])
        
        if research.team2_stats.get('search_results'):
            for snippet in research.team2_stats['search_results']:
                lines.append(f"> {snippet[:300]}...")
                lines.append("")
        else:
            lines.append("*No statistics found*")
            lines.append("")
        
        lines.extend([
            "---",
            "",
            "## Head-to-Head History",
            "",
        ])
        
        if research.head_to_head:
            for item in research.head_to_head:
                lines.append(f"> {item[:300]}...")
                lines.append("")
        else:
            lines.append("*No head-to-head data found*")
            lines.append("")
        
        lines.extend([
            "---",
            "",
            "## Recent Form",
            "",
        ])
        
        for team, form in research.recent_form.items():
            lines.append(f"### {team}")
            if form:
                for item in form[:2]:
                    lines.append(f"> {item[:200]}...")
            else:
                lines.append("*No recent form data*")
            lines.append("")
        
        lines.extend([
            "---",
            "",
            "## Live Statistics",
            "",
        ])
        
        if research.live_stats.get('search_results'):
            for snippet in research.live_stats['search_results']:
                lines.append(f"> {snippet[:300]}...")
                lines.append("")
        else:
            lines.append("*No live statistics available*")
            lines.append("")
        
        if research.additional_info:
            lines.extend([
                "---",
                "",
                "## Additional Information",
                "",
            ])
            for info in research.additional_info:
                lines.append(f"- {info[:200]}...")
            lines.append("")
        
        lines.extend([
            "---",
            "",
            "## Research Sources",
            "",
        ])
        for source in research.sources_used:
            lines.append(f"- {source}")
        
        lines.extend([
            "",
            "---",
            f"*Generated: {datetime.now().isoformat()}*",
        ])
        
        return '\n'.join(lines)


# ============================================================================
# MAIN SCRAPER ENGINE
# ============================================================================

class MatchResearchEngine:
    """Main engine for async match research"""
    
    def __init__(self, results_file: str = "results.md"):
        self.results_file = results_file
        self.parser = ResultsParser(results_file)
        self.output_handler = ResearchOutputHandler()
        self.client: Optional[StealthHTTPClient] = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)
    
    async def run(self) -> List[str]:
        """Run full research pipeline"""
        logger.info("[ENGINE] Starting Phase 2: Match Research")
        
        # Parse matches
        matches = self.parser.parse()
        
        if not matches:
            logger.warning("[ENGINE] No matches found in results.md")
            return []
        
        # Initialize stealth client
        self.client = await get_stealth_client()
        
        try:
            # Create scrapers
            scrapers = {
                SportType.FOOTBALL: FootballScraper(self.client),
                SportType.BASKETBALL: BasketballScraper(self.client),
                SportType.TENNIS: TennisScraper(self.client),
            }
            
            # Research all matches concurrently (with semaphore limit)
            tasks = [
                self._research_match_with_limit(match, scrapers)
                for match in matches
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Save all research
            output_files = []
            for result in results:
                if isinstance(result, MatchResearch):
                    filepath = self.output_handler.save_research(result)
                    output_files.append(filepath)
                elif isinstance(result, Exception):
                    logger.error(f"[ENGINE] Research failed: {result}")
            
            # Save to database for Phase 3
            self._save_to_database(output_files)
            
            logger.info(f"[ENGINE] Phase 2 complete. Researched {len(output_files)} matches.")
            return output_files
            
        finally:
            await close_stealth_client()
    
    async def _research_match_with_limit(
        self,
        match: Match,
        scrapers: Dict[SportType, BaseScraper]
    ) -> MatchResearch:
        """Research a match with concurrency limit"""
        async with self.semaphore:
            logger.info(f"[ENGINE] Researching match {match.number}: {match.teams}")
            
            scraper = scrapers.get(match.sport, scrapers[SportType.FOOTBALL])
            return await scraper.research_match(match)
    
    def _save_to_database(self, output_files: List[str]):
        """Save research results to database for Phase 3"""
        session_id = db.start_session('research', 'phase2')
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create research results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_research (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    match_number INTEGER,
                    filepath TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed INTEGER DEFAULT 0
                )
            ''')
            
            for filepath in output_files:
                # Extract match number from filename
                match_num = int(Path(filepath).stem.split('_')[1])
                cursor.execute('''
                    INSERT INTO match_research (session_id, match_number, filepath)
                    VALUES (?, ?, ?)
                ''', (session_id, match_num, filepath))
        
        db.complete_session(session_id, 'completed', len(output_files))
        logger.info(f"[DATABASE] Saved {len(output_files)} research files to database")


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    engine = MatchResearchEngine()
    output_files = await engine.run()
    
    print(f"\n{'='*60}")
    print(f"Phase 2 Complete: {len(output_files)} matches researched")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    return output_files


if __name__ == "__main__":
    asyncio.run(main())
