import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "scraper_data.db"


def main() -> None:
    print("db_path:", DB_PATH)
    print("db_exists:", DB_PATH.exists())
    if not DB_PATH.exists():
        raise SystemExit("scraper_data.db not found")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    def cols(table: str) -> list[str]:
        cur.execute(f"PRAGMA table_info({table})")
        return [r[1] for r in cur.fetchall()]

    print("live_matches cols:", cols("live_matches"))
    print("match_research cols:", cols("match_research"))
    print("predictions cols:", cols("predictions"))
    print("scrape_sessions cols:", cols("scrape_sessions"))

    desired_session_cols = [
        "id",
        "source",
        "workflow",
        "started_at",
        "status",
        "total_matches",
        "error_message",
        "error",
    ]
    session_cols_available = set(cols("scrape_sessions"))
    session_cols = [c for c in desired_session_cols if c in session_cols_available]
    if not session_cols:
        raise SystemExit("scrape_sessions exists but has no expected columns")

    cur.execute(
        f"SELECT {', '.join(session_cols)} FROM scrape_sessions WHERE source=? ORDER BY id DESC LIMIT 5",
        ("sportybet",),
    )
    print("latest sportybet sessions:")
    for row in cur.fetchall():
        print(dict(row))

    cur.execute(
        "SELECT id, session_id, sport, league, home_team, away_team, match_time, status "
        "FROM live_matches WHERE source=? ORDER BY id DESC LIMIT 10",
        ("sportybet",),
    )
    print("latest sportybet matches:")
    match_rows = [dict(r) for r in cur.fetchall()]
    for row in match_rows:
        print(row)

    match_ids = [r["id"] for r in match_rows if "id" in r]
    if match_ids:
        placeholders = ",".join(["?"] * len(match_ids))
        cur.execute(
            f"SELECT id, live_match_id, home_team, away_team, created_at FROM match_research WHERE live_match_id IN ({placeholders}) ORDER BY id DESC LIMIT 20",
            match_ids,
        )
        print("recent match_research for these match IDs:")
        for row in cur.fetchall():
            print(dict(row))


if __name__ == "__main__":
    main()
