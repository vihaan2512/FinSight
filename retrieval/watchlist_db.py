import sqlite3
import os
from loguru import logger

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.db")


def get_db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_watchlist_db():
    """Initialize SQLite tables for user management, watchlist, and portfolio."""
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            user_id INTEGER,
            ticker TEXT,
            PRIMARY KEY (user_id, ticker),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            user_id INTEGER,
            ticker TEXT,
            weight REAL,
            PRIMARY KEY (user_id, ticker),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Watchlist and portfolio database initialized successfully.")


def add_user(username: str) -> int:
    """Create a new user or return existing user_id."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username.lower().strip(),))
        conn.commit()
        user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username.lower().strip(),))
        user_id = cursor.fetchone()[0]
    conn.close()
    return user_id


def get_user_id(username: str) -> int:
    """Get user_id for a username, creating it if it doesn't exist."""
    return add_user(username)


def add_to_watchlist(user_id: int, ticker: str) -> bool:
    """Add a ticker to a user's watchlist."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR IGNORE INTO watchlist (user_id, ticker) VALUES (?, ?)",
            (user_id, ticker.upper().strip())
        )
        conn.commit()
        success = True
    except Exception as e:
        logger.error(f"Failed to add {ticker} to watchlist for user {user_id}: {e}")
        success = False
    conn.close()
    return success


def remove_from_watchlist(user_id: int, ticker: str) -> bool:
    """Remove a ticker from a user's watchlist."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND ticker = ?",
            (user_id, ticker.upper().strip())
        )
        conn.commit()
        success = True
    except Exception as e:
        logger.error(f"Failed to remove {ticker} from watchlist for user {user_id}: {e}")
        success = False
    conn.close()
    return success


def get_watchlist(user_id: int) -> list[str]:
    """Retrieve all tickers in a user's watchlist."""
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM watchlist WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def add_to_portfolio(user_id: int, ticker: str, weight: float) -> bool:
    """Add or update a ticker with its allocation weight in a user's portfolio."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO portfolio (user_id, ticker, weight) VALUES (?, ?, ?)",
            (user_id, ticker.upper().strip(), float(weight))
        )
        conn.commit()
        success = True
    except Exception as e:
        logger.error(f"Failed to add/update {ticker} to portfolio for user {user_id}: {e}")
        success = False
    conn.close()
    return success


def remove_from_portfolio(user_id: int, ticker: str) -> bool:
    """Remove a ticker from a user's portfolio."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM portfolio WHERE user_id = ? AND ticker = ?",
            (user_id, ticker.upper().strip())
        )
        conn.commit()
        success = True
    except Exception as e:
        logger.error(f"Failed to remove {ticker} from portfolio for user {user_id}: {e}")
        success = False
    conn.close()
    return success


def get_portfolio(user_id: int) -> dict[str, float]:
    """Retrieve all tickers and their weights in a user's portfolio."""
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, weight FROM portfolio WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


# Initialize table schemas
init_watchlist_db()