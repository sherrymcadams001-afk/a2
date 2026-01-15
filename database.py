#!/usr/bin/env python3
"""
Database layer for storing scraped match data.
Uses SQLite for simplicity, can be swapped for PostgreSQL/MySQL.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

DB_PATH = os.environ.get('SCRAPER_DB', os.path.join(os.path.dirname(__file__), 'scraper_data.db'))


@contextmanager
def get_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """Initialize database with required tables"""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                category TEXT NOT NULL,
                live_count INTEGER DEFAULT 0,
                session_id INTEGER,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, category)
            )
        ''')
        
        # Live matches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                sport TEXT NOT NULL,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER DEFAULT 0,
                away_score INTEGER DEFAULT 0,
                match_time TEXT,
                status TEXT DEFAULT 'Live',
                extra_data TEXT,
                session_id INTEGER,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, sport, home_team, away_team, scraped_at)
            )
        ''')
        
        # Scrape sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scrape_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                workflow TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT DEFAULT 'running',
                total_matches INTEGER DEFAULT 0,
                error TEXT
            )
        ''')
        
        # Match research table - stores AI research for specific matches
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS match_research (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                live_match_id INTEGER,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                league TEXT,
                match_date TEXT,
                research_source TEXT DEFAULT 'duck.ai',
                
                -- Form data
                home_form TEXT,
                away_form TEXT,
                home_position INTEGER,
                away_position INTEGER,
                home_points INTEGER,
                away_points INTEGER,
                
                -- Head to head
                h2h_total_matches INTEGER,
                h2h_home_wins INTEGER,
                h2h_away_wins INTEGER,
                h2h_draws INTEGER,
                h2h_avg_goals REAL,
                last_meeting_result TEXT,
                
                -- Key players / injuries
                home_key_players TEXT,
                away_key_players TEXT,
                home_injuries TEXT,
                away_injuries TEXT,
                
                -- Raw response text
                raw_response TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Predictions table - stores outcome predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                research_id INTEGER,
                live_match_id INTEGER,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                
                -- Prediction data
                predicted_winner TEXT,
                predicted_home_score INTEGER,
                predicted_away_score INTEGER,
                over_under_line REAL DEFAULT 2.5,
                over_under_prediction TEXT,
                confidence TEXT,
                
                -- Betting recommendations
                recommended_bet TEXT,
                odds_value TEXT,
                
                -- Tracking
                actual_home_score INTEGER,
                actual_away_score INTEGER,
                prediction_correct INTEGER,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (research_id) REFERENCES match_research(id)
            )
        ''')
        
        # Lightweight migrations for existing DBs (ADD COLUMN if missing)
        def _has_column(table: str, col: str) -> bool:
            cursor.execute(f"PRAGMA table_info({table})")
            return any(r[1] == col for r in cursor.fetchall())

        def _add_column(table: str, col: str, col_type: str):
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

        try:
            if not _has_column('live_matches', 'session_id'):
                _add_column('live_matches', 'session_id', 'INTEGER')
        except Exception:
            pass

        try:
            if not _has_column('categories', 'session_id'):
                _add_column('categories', 'session_id', 'INTEGER')
        except Exception:
            pass

        try:
            if not _has_column('match_research', 'live_match_id'):
                _add_column('match_research', 'live_match_id', 'INTEGER')
        except Exception:
            pass

        try:
            if not _has_column('predictions', 'live_match_id'):
                _add_column('predictions', 'live_match_id', 'INTEGER')
        except Exception:
            pass

        # Create indexes (after migrations so columns exist)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_sport ON live_matches(sport)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_source ON live_matches(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_scraped ON live_matches(scraped_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_research_teams ON match_research(home_team, away_team)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_research_date ON match_research(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_research ON predictions(research_id)')

        # Optional indexes for newly-added columns (may fail on older SQLite states)
        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_session ON live_matches(session_id)')
        except Exception:
            pass

        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_research_match ON match_research(live_match_id)')
        except Exception:
            pass


def start_session(source: str, workflow: str) -> int:
    """Start a new scrape session, return session ID"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO scrape_sessions (source, workflow) VALUES (?, ?)',
            (source, workflow)
        )
        return cursor.lastrowid


def complete_session(session_id: int, total_matches: int, error: Optional[str] = None):
    """Mark session as complete"""
    status = 'error' if error else 'completed'
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''UPDATE scrape_sessions 
               SET completed_at = CURRENT_TIMESTAMP, status = ?, total_matches = ?, error = ?
               WHERE id = ?''',
            (status, total_matches, error, session_id)
        )


def store_categories(source: str, categories: List[Dict], session_id: Optional[int] = None) -> int:
    """Store sport categories"""
    with get_connection() as conn:
        cursor = conn.cursor()
        count = 0
        for cat in categories:
            try:
                cursor.execute(
                    '''INSERT OR REPLACE INTO categories (source, category, live_count, session_id, scraped_at)
                       VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                    (source, cat.get('category', ''), cat.get('live_count', 0), session_id)
                )
                count += 1
            except Exception as e:
                print(f"Error storing category {cat}: {e}")
        return count


def store_matches(source: str, matches: List[Dict], mode: str = 'append', session_id: Optional[int] = None) -> int:
    """Store match data. Mode: 'append' or 'replace'"""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        if mode == 'replace':
            # Clear existing matches for this source from today
            cursor.execute(
                "DELETE FROM live_matches WHERE source = ? AND date(scraped_at) = date('now')",
                (source,)
            )
        
        count = 0
        for match in matches:
            try:
                # Extract extra data as JSON (exclude known fields from all naming conventions)
                known_fields = {'sport', 'league', 'league_name', 'tournament',
                               'home_team', 'away_team', 'team1', 'team2', 'team1_name', 'team2_name',
                               'player1', 'player2', 'home_score', 'away_score', 'score1', 'score2',
                               'team1_score', 'team2_score', 'match_time', 'status', 'team_names'}
                extra_fields = {k: v for k, v in match.items() if k not in known_fields}
                extra_json = json.dumps(extra_fields) if extra_fields else None
                
                # Map various field naming conventions to standard DB fields
                # Handle team_names field (can be string "Team A vs Team B" or array ["Team A", "Team B"])
                team_names = match.get('team_names')
                parsed_home = ''
                parsed_away = ''
                if team_names:
                    if isinstance(team_names, list) and len(team_names) >= 2:
                        # Array format: ["Team A", "Team B"]
                        parsed_home = str(team_names[0]).strip()
                        parsed_away = str(team_names[1]).strip()
                    elif isinstance(team_names, str) and ' vs ' in team_names:
                        # String format: "Team A vs Team B"
                        parts = team_names.split(' vs ', 1)
                        if len(parts) == 2:
                            parsed_home = parts[0].strip()
                            parsed_away = parts[1].strip()
                
                # Priority: home_team > team1_name > team1 > player1 > parsed from team_names
                home_team = (match.get('home_team') or match.get('team1_name') or 
                            match.get('team1') or match.get('player1') or parsed_home or '')
                away_team = (match.get('away_team') or match.get('team2_name') or 
                            match.get('team2') or match.get('player2') or parsed_away or '')
                
                # Priority: home_score > team1_score > score1
                home_score = match.get('home_score') or match.get('team1_score') or match.get('score1') or 0
                away_score = match.get('away_score') or match.get('team2_score') or match.get('score2') or 0
                
                # Handle None scores (convert to 0 for DB)
                home_score = home_score if home_score is not None else 0
                away_score = away_score if away_score is not None else 0
                
                # League: league > league_name > tournament
                league = match.get('league') or match.get('league_name') or match.get('tournament') or ''
                
                # Try to infer sport from league name if not provided
                sport = match.get('sport', 'Unknown')
                if sport == 'Unknown' and league:
                    league_lower = league.lower()
                    # Basketball keywords
                    if any(kw in league_lower for kw in ['nba', 'ncaa', 'basketball', 'nbb', 'euroleague', 
                                                          'nbl', 'wnbl', 'kbl', 'bskt', 'lakers', 'sakers']):
                        sport = 'Basketball'
                    # Football/Soccer keywords  
                    elif any(kw in league_lower for kw in ['premier', 'liga', 'serie', 'football', 'soccer', 
                                                            'bundesliga', 'nusantara', 'league', 'fc ', ' fc',
                                                            'united', 'trophy', 'elites']):
                        sport = 'Football'
                    # Tennis keywords
                    elif any(kw in league_lower for kw in ['atp', 'wta', 'tennis', 'grand slam', 'itf', 
                                                           'utr', 'doubles', 'singles']):
                        sport = 'Tennis'
                
                cursor.execute(
                    '''INSERT INTO live_matches 
                       (source, sport, league, home_team, away_team, home_score, away_score, match_time, status, extra_data, session_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        source,
                        sport,
                        league,
                        home_team,
                        away_team,
                        home_score,
                        away_score,
                        match.get('match_time', ''),
                        match.get('status', 'Live'),
                        extra_json,
                        session_id
                    )
                )
                count += 1
            except Exception as e:
                print(f"Error storing match {match}: {e}")
        
        return count


def get_latest_matches(source: Optional[str] = None, sport: Optional[str] = None, limit: int = 100) -> List[Dict]:
    """Get latest scraped matches"""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        query = 'SELECT * FROM live_matches WHERE 1=1'
        params = []
        
        if source:
            query += ' AND source = ?'
            params.append(source)
        
        if sport:
            query += ' AND sport = ?'
            params.append(sport)
        
        query += ' ORDER BY scraped_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]


def get_match_counts_by_sport(source: Optional[str] = None) -> Dict[str, int]:
    """Get match counts grouped by sport"""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        if source:
            cursor.execute(
                '''SELECT sport, COUNT(*) as count FROM live_matches 
                   WHERE source = ? AND date(scraped_at) = date('now')
                   GROUP BY sport''',
                (source,)
            )
        else:
            cursor.execute(
                '''SELECT sport, COUNT(*) as count FROM live_matches 
                   WHERE date(scraped_at) = date('now')
                   GROUP BY sport'''
            )
        
        return {row['sport']: row['count'] for row in cursor.fetchall()}


def get_session_stats() -> Dict[str, Any]:
    """Get statistics about scrape sessions"""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Total sessions today
        cursor.execute(
            "SELECT COUNT(*) FROM scrape_sessions WHERE date(started_at) = date('now')"
        )
        sessions_today = cursor.fetchone()[0]
        
        # Latest session
        cursor.execute(
            'SELECT * FROM scrape_sessions ORDER BY started_at DESC LIMIT 1'
        )
        latest = cursor.fetchone()
        
        # Total matches today
        cursor.execute(
            "SELECT COUNT(*) FROM live_matches WHERE date(scraped_at) = date('now')"
        )
        matches_today = cursor.fetchone()[0]
        
        return {
            'sessions_today': sessions_today,
            'matches_today': matches_today,
            'latest_session': dict(latest) if latest else None
        }


# Initialize database on import
init_database()
