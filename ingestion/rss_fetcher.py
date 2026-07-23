"""
Fetches finance news from TRUSTED, high-credibility sources only.

Sources:    Economic Times, Moneycontrol, Business Standard,
            Livemint, The Hindu BusinessLine, Financial Express, ZEE Business, SEBI, RBI Press Releases,
            NSE, BSE Corporate Announcements
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import re
from datetime import datetime, timezone
from typing import Optional

import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from loguru import logger
import concurrent.futures
from ingestion.company_seed import COMMON_ENTITIES


# ── Trusted RSS feeds ─────────────────────────────────────────────────────────

INDIAN_FEEDS = {
    "economic_times_markets": "https://news.google.com/rss/search?q=when:24h+site:economictimes.indiatimes.com+markets&hl=en-IN&gl=IN&ceid=IN:en",
    "economic_times_stocks": "https://news.google.com/rss/search?q=when:24h+site:economictimes.indiatimes.com+stocks&hl=en-IN&gl=IN&ceid=IN:en",
    "economic_times_economy": "https://news.google.com/rss/search?q=when:24h+site:economictimes.indiatimes.com+economy&hl=en-IN&gl=IN&ceid=IN:en",
    "moneycontrol_news": "https://www.moneycontrol.com/rss/latestnews.xml",
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "business_standard_market": "https://news.google.com/rss/search?q=when:24h+site:business-standard.com+markets&hl=en-IN&gl=IN&ceid=IN:en",
    "business_standard_economy": "https://news.google.com/rss/search?q=when:24h+site:business-standard.com+economy&hl=en-IN&gl=IN&ceid=IN:en",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "livemint_companies": "https://www.livemint.com/rss/companies",
    "hindu_businessline": "https://www.thehindubusinessline.com/markets/?service=rss",
    "livemint_money": "https://www.livemint.com/rss/money",
    "livemint_industry": "https://www.livemint.com/rss/industry",
    "financial_express_market": "https://news.google.com/rss/search?q=when:24h+site:financialexpress.com+market&hl=en-IN&gl=IN&ceid=IN:en",
    "financial_express_economy": "https://news.google.com/rss/search?q=when:24h+site:financialexpress.com+economy&hl=en-IN&gl=IN&ceid=IN:en",
    "cnbctv18_market": "https://news.google.com/rss/search?q=when:24h+site:cnbctv18.com+market&hl=en-IN&gl=IN&ceid=IN:en",
    "cnbctv18_economy": "https://news.google.com/rss/search?q=when:24h+site:cnbctv18.com+economy&hl=en-IN&gl=IN&ceid=IN:en",
    "zeebusiness_market": "https://news.google.com/rss/search?q=when:24h+site:zeebiz.com+markets&hl=en-IN&gl=IN&ceid=IN:en",
    "sebi_press_releases": "https://www.sebi.gov.in/sebirss.xml",
    "rbi_press_releases": "https://news.google.com/rss/search?q=when:24h+site:rbi.org.in&hl=en-IN&gl=IN&ceid=IN:en",
    "nse_corporate_announcements": "https://news.google.com/rss/search?q=when:24h+site:economictimes.indiatimes.com+%22nse%22+%22announcement%22+OR+%22board+meeting%22+OR+%22dividend%22+OR+%22earnings%22&hl=en-IN&gl=IN&ceid=IN:en",
    "bse_corporate_announcements": "https://news.google.com/rss/search?q=when:24h+site:business-standard.com+%22bse%22+%22announcement%22+OR+%22filing%22+OR+%22dividend%22+OR+%22earnings%22&hl=en-IN&gl=IN&ceid=IN:en",
    "chittorgarh_ipo": "https://www.chittorgarh.com/rss/ipo_rss.xml",
    "moneycontrol_sme_ipo": "https://www.moneycontrol.com/rss/sme_ipo.xml",
    "livemint_sector_indices": "https://www.livemint.com/rss/market/sector-news",
    "bse_sme_announcements": "https://news.google.com/rss/search?q=when:24h+site:business-standard.com+%22BSE+SME%22+OR+%22BSE+announcement%22&hl=en-IN&gl=IN&ceid=IN:en",
    "ndtv_profit": "https://news.google.com/rss/search?q=when:24h+site:ndtvprofit.com&hl=en-IN&gl=IN&ceid=IN:en",
}

ALL_TRUSTED_FEEDS = INDIAN_FEEDS

TICKER_STOPWORDS = {
    "A", "I", "IT", "OR", "AS", "AT", "BY", "BE", "IS", "IN",
    "ON", "TO", "UP", "US", "WE", "AI", "FY", "AM", "PM",
    "Q1", "Q2", "Q3", "Q4", "H1", "H2",
    "CEO", "CFO", "COO", "CTO", "IPO", "GDP", "CPI", "WPI",
    "ETF", "RBI", "NSE", "BSE",
    "FOR", "NEW", "NOW", "ALL", "NOT", "BUT", "AND", "THE",
    "ARE", "INC", "LTD", "PLC", "PTY", "LLC", "PVT",
    "YOY", "QOQ", "MOM", "PAT", "EPS", "PE", "PB",
    "ECONOMIC TIMES", "LIVE MINT", "BUSINESS STANDARD", "FINANCIAL EXPRESS", 
    "ZEE BUSINESS", "SEBI", "GOVERNMENT", "MINISTRY", "CENTRAL BANK", 
    "MARKET", "MARKETS", "NEWS", "STOCK", "STOCKS", "ECONOMY", "MODI", 
    "NIFTY", "SENSEX", "WAR", "INFLATION", "INTEREST RATES", "GOLD", "OIL", 
    "RUPEE", "TECH", "APP", "GOOGLE NEWS", "YAHOO", "YAHOO FINANCE"
}


def extract_tickers(text: str) -> list[str]:
    """
    Extract ALL plausible ticker symbols from text — not limited to
    any hardcoded list.  This means articles about Zomato, Paytm,
    Snowflake, or any other company are correctly tagged at ingest
    time and retrievable later.

    Strategy 1 — $TICKER  (most reliable, always include)
    Strategy 2 — ALL-CAPS words 2-6 chars, filtered by stopwords
    Strategy 3 — known exchange-suffix patterns e.g. "ZOMATO.NS"
    Strategy 4 — Exchange prefix patterns e.g. (NSE: WIPRO), (NASDAQ: AAPL)
    Strategy 5 — Common entity name matching (e.g. "Tata Motors" -> TATAMOTORS)
    """
    found = set()

    # S1: explicit $TICKER notation
    found.update(re.findall(r'\$([A-Z]{1,6})\b', text))

    # S2: standalone ALL-CAPS tokens (the bulk of real tickers)
    for word in re.findall(r'\b([A-Z]{2,6})\b', text):
        if word not in TICKER_STOPWORDS:
            found.add(word)

    # S3: NSE/BSE suffix pattern  e.g. ZOMATO.NS  HDFCBANK.BO
    for match in re.findall(r'\b([A-Z]{2,10})\.(NS|BO|BSE|NSE)\b', text):
        found.add(match[0])

    # S4: Exchange prefix patterns e.g. (NSE: WIPRO), (BSE: RELIANCE)
    for match in re.findall(r'\b(?:NSE|BSE)\s*:\s*([A-Z0-9]{1,10})\b', text, re.IGNORECASE):
        found.add(match.upper())

    # S5: Search for common company names from seeds
    text_lower = text.lower()
    for alias, name, ticker, sector, industry in COMMON_ENTITIES:
        norm_alias = normalize_company_name(alias)
        norm_name = normalize_company_name(name)
        
        # Check alias (e.g. "tata motors" or "reliance")
        if len(norm_alias) >= 3:
            if re.search(r'\b' + re.escape(norm_alias) + r'\b', text_lower):
                t_clean = ticker.split('.')[0] if '.' in ticker else ticker
                found.add(t_clean.upper())
                continue
                
        if len(norm_name) >= 3:
            if re.search(r'\b' + re.escape(norm_name) + r'\b', text_lower):
                t_clean = ticker.split('.')[0] if '.' in ticker else ticker
                found.add(t_clean.upper())

    return sorted(found)


def detect_region(source_name: str, title: str = "", summary: str = "") -> str:
    text = (title + " " + summary).lower()
    global_triggers = [
        "us stocks", "spacex", "wall street", "us fed", "nasdaq", "dow jones",
        "s&p 500", "global markets", "brent crude", "global economy", "us inflation",
        "federal reserve", "world bank", "imf", "european union", "white house",
        "geopolitical tensions", "us-iran", "global triggers", "middle east tension",
        "fed meeting", "fed's policy", "fed interest rate", "us-iran deal"
    ]
    for trigger in global_triggers:
        if trigger in text:
            return "global"
    return "india"



def clean_html(raw: str) -> str:
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "lxml")
    return " ".join(soup.get_text(separator=" ").split())[:3000]


def parse_date(date_str: Optional[str]) -> str:
    if not date_str:
        return datetime.now(timezone.utc).isoformat()
    try:
        from dateutil import parser as dp
        dt = dp.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def make_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


SOURCE_QUALITY = {
    # Indian Feeds
    "economic_times_markets":    {"source_tier": 2, "credibility_score": 0.8, "document_type": "news"},
    "economic_times_stocks":     {"source_tier": 2, "credibility_score": 0.8, "document_type": "news"},
    "economic_times_economy":    {"source_tier": 2, "credibility_score": 0.8, "document_type": "news"},
    "moneycontrol_news":         {"source_tier": 2, "credibility_score": 0.85, "document_type": "news"},
    "moneycontrol_markets":      {"source_tier": 2, "credibility_score": 0.85, "document_type": "news"},
    "business_standard_market":  {"source_tier": 2, "credibility_score": 0.8, "document_type": "news"},
    "business_standard_economy": {"source_tier": 2, "credibility_score": 0.8, "document_type": "news"},
    "livemint_markets":          {"source_tier": 2, "credibility_score": 0.8, "document_type": "news"},
    "livemint_companies":        {"source_tier": 2, "credibility_score": 0.8, "document_type": "news"},
    "hindu_businessline":        {"source_tier": 2, "credibility_score": 0.75, "document_type": "news"},
    "ndtv_profit":               {"source_tier": 2, "credibility_score": 0.8, "document_type": "news"},
    "sebi_press_releases":        {"source_tier": 0, "credibility_score": 1.0, "document_type": "regulator"},
    "rbi_press_releases":         {"source_tier": 0, "credibility_score": 1.0, "document_type": "central_bank"},
    "nse_corporate_announcements":{"source_tier": 0, "credibility_score": 1.0, "document_type": "exchange_announcement"},
    "bse_corporate_announcements":{"source_tier": 0, "credibility_score": 1.0, "document_type": "exchange_announcement"},
    "chittorgarh_ipo":            {"source_tier": 2, "credibility_score": 0.8, "document_type": "news"},
    "moneycontrol_sme_ipo":       {"source_tier": 2, "credibility_score": 0.85, "document_type": "news"},
    "livemint_sector_indices":    {"source_tier": 2, "credibility_score": 0.8, "document_type": "news"},
    "bse_sme_announcements":      {"source_tier": 0, "credibility_score": 1.0, "document_type": "exchange_announcement"},
}

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entity_cache.db")


def get_db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entity_cache (
            alias TEXT PRIMARY KEY,
            company_name TEXT,
            ticker TEXT,
            sector TEXT,
            industry TEXT
        )
    """)
    conn.commit()

    # Pre-seed top Indian market-cap companies from separate seed file
    for alias, name, ticker, sector, industry in COMMON_ENTITIES:
        normalized_alias = normalize_company_name(alias)
        cursor.execute(
            "INSERT OR IGNORE INTO entity_cache VALUES (?, ?, ?, ?, ?)",
            (normalized_alias, name, ticker, sector, industry)
        )
    conn.commit()
    conn.close()


def normalize_company_name(name: str) -> str:
    """Normalize company name by converting to lowercase and stripping common corporate suffixes."""
    name = name.lower().strip()
    suffixes = [
        "pvt ltd", "pvt. ltd.", "pvt.ltd.", "private limited", "ltd.", "ltd", "limited",
        "inc.", "inc", "llc", "corp.", "corp", "corporation", "co.", "co", "company", "plc"
    ]
    for s in suffixes:
        if name.endswith(" " + s) or name.endswith("." + s) or name.endswith(" " + s + "."):
            name = name.rsplit(s, 1)[0].strip()
            break
    # Clean any trailing punctuation/whitespace
    name = re.sub(r'[\s.,\-&()]+$', '', name).strip()
    return name


init_db()


# Build in-memory lookup map for pre-seeded company entities
SEED_MAP = {}
for alias, name, ticker, sector, industry in COMMON_ENTITIES:
    normalized_alias = normalize_company_name(alias)
    normalized_name = normalize_company_name(name)
    info = {
        "company_name": name,
        "ticker": ticker,
        "sector": sector,
        "industry": industry
    }
    SEED_MAP[normalized_alias] = info
    SEED_MAP[normalized_name] = info


def find_in_seeds(normalized_name: str) -> Optional[dict]:
    # 1. Exact match in SEED_MAP
    if normalized_name in SEED_MAP:
        return SEED_MAP[normalized_name]

    # 2. Word boundary substring matching to find in seeds
    for alias, name, ticker, sector, industry in COMMON_ENTITIES:
        norm_alias = normalize_company_name(alias)
        norm_name = normalize_company_name(name)
        if len(norm_alias) >= 3:
            # check if norm_alias is a word in normalized_name
            if re.search(r'\b' + re.escape(norm_alias) + r'\b', normalized_name):
                return {
                    "company_name": name,
                    "ticker": ticker,
                    "sector": sector,
                    "industry": industry
                }
            # check if normalized_name is a word in norm_name
            if re.search(r'\b' + re.escape(normalized_name) + r'\b', norm_name):
                return {
                    "company_name": name,
                    "ticker": ticker,
                    "sector": sector,
                    "industry": industry
                }
    return None


def resolve_ticker_via_api(alias: str, conn=None) -> dict:
    """Query Yahoo Finance Search API to match company alias with its ticker and industry."""
    normalized = normalize_company_name(alias)
    if not normalized:
        return {}

    # 1. Search in-memory company seeds first
    seed_match = find_in_seeds(normalized)
    if seed_match:
        SEED_MAP[normalized] = seed_match
        return seed_match

    # 2. Check SQLite cache using normalized alias
    should_close = False
    if conn is None:
        conn = get_db_conn()
        should_close = True

    cursor = conn.cursor()
    cursor.execute("SELECT company_name, ticker, sector, industry FROM entity_cache WHERE alias = ?", (normalized,))
    cached = cursor.fetchone()
    if cached:
        if should_close:
            conn.close()
        res = {
            "company_name": cached[0],
            "ticker": cached[1],
            "sector": cached[2],
            "industry": cached[3]
        }
        SEED_MAP[normalized] = res
        return res

    # Fetch from API with robust error fallback
    result = {}
    try:
        import requests
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={normalized}&quotesCount=1&newsCount=0"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=1.5)
        if res.status_code == 200:
            data = res.json()
            quotes = data.get("quotes", [])
            if quotes:
                quote = quotes[0]
                result = {
                    "company_name": quote.get("longname") or quote.get("shortname") or alias,
                    "ticker": quote.get("symbol"),
                    "sector": quote.get("sector", ""),
                    "industry": quote.get("industry", "")
                }
    except Exception as e:
        logger.warning(f"Yahoo Search API error or rate-limit for {normalized}: {e}")
    
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO entity_cache VALUES (?, ?, ?, ?, ?)",
            (normalized, result.get("company_name", ""), result.get("ticker", ""), result.get("sector", ""), result.get("industry", ""))
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to commit entity cache for {normalized}: {e}")
    if should_close:
        conn.close()
    
    # Also save in the in-memory SEED_MAP to avoid DB lookup next time
    SEED_MAP[normalized] = result
    
    return result



_nlp = None

def get_spacy_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' is missing. Please download it: python -m spacy download en_core_web_sm")
            raise ImportError(
                "Required spaCy model 'en_core_web_sm' is not installed. "
                "Please run: python -m spacy download en_core_web_sm"
            )
    return _nlp


def match_company_master(text: str) -> dict:
    """
    Left as a compatibility stub or helper for single article matching.
    For performance, fetch_all_articles uses nlp.pipe batch processing.
    """
    companies = set()
    tickers = set()
    sectors = set()
    industries = set()
    
    nlp = get_spacy_nlp()
    if not nlp:
        return {
            "companies": [],
            "tickers": [],
            "sectors": [],
            "industries": [],
        }

    doc = nlp(text)
    # Extract ORG entities
    orgs = {ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"}
    
    for name in orgs:
        # Clean name checks
        if name.upper() in TICKER_STOPWORDS or len(name) < 3 or name.isdigit() or not name[0].isalnum():
            continue
        resolved = resolve_ticker_via_api(name)
        if resolved and resolved.get("ticker"):
            companies.add(resolved["company_name"])
            tickers.add(resolved["ticker"])
            if resolved.get("sector"):
                sectors.add(resolved["sector"])
            if resolved.get("industry"):
                industries.add(resolved["industry"])

    return {
        "companies": list(companies),
        "tickers": list(tickers),
        "sectors": list(sectors),
        "industries": list(industries),
    }


def classify_event(title: str, text: str) -> str:
    content = (title + " " + text).lower()
    # Keywords prioritized from specific/high-value to general
    event_triggers = {
        "earnings": ["quarterly results", "q4 results", "q3 results", "q2 results", "q1 results", "net profit", "profit jumps", "profit drops", "earnings report", "earnings"],
        "dividend": ["interim dividend", "dividend yield", "dividend declaration", "dividend"],
        "buyback": ["share buyback", "buyback offer", "buyback"],
        "acquisition": ["acquisition", "acquires", "acquired", "takeover"],
        "merger": ["merger", "amalgamation", "merges"],
        "ipo": ["initial public offering", "ipo price", "public issue", "listing date", "ipo"],
        "fundraising": ["raising funds", "raise capital", "debt issue", "bonds issue", "fundraising", "fundraise"],
        "management_change": ["management change", "appoints ceo", "appoints cfo", "ceo resigns", "cfo resigns", "resigns as ceo", "resigns as cfo", "stepping down as ceo"],
        "litigation": ["sebi penalty", "sebi fine", "court order", "lawsuit", "litigation", "sues", "sued"],
        "regulation": ["policy change", "regulatory action", "guidelines", "compliance", "ban on", "regulation", "regulatory"],
        "macro": ["repo rate", "inflation rate", "gdp growth", "interest rate", "central bank", "monetary policy", "macroeconomic"],
        "guidance": ["earnings outlook", "future guidance", "outlook", "projection", "target price", "forecast"],
        "product_launch": ["product launch", "unveils new", "introduces new", "launching new"]
    }
    
    # Score each event type based on keyword matches, select the one with highest matches
    scores = {}
    for event, keywords in event_triggers.items():
        score = 0
        for kw in keywords:
            if kw in content:
                # Use length as weight to favor specific phrases over general single words
                score += len(kw)
        if score > 0:
            scores[event] = score
            
    if scores:
        return max(scores, key=scores.get)
    return "news"


def scrape_full_text(url: str) -> Optional[str]:
    """
    Attempts to download and extract full text from a news URL.
    Handles bot detection and checks for premium paywalls, returning None on failure/paywall.
    """
    lower_url = url.lower()
    premium_indicators = ["/premium/", "/prime/", "/subscription/", "/paywall/"]
    if any(ind in lower_url for ind in premium_indicators):
        logger.debug(f"Skipping premium URL: {url}")
        return None

    import requests
    from bs4 import BeautifulSoup
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            logger.debug(f"Failed to fetch {url}: HTTP {response.status_code}")
            return None
        
        final_url = response.url.lower()
        if any(ind in final_url for ind in ["login", "register", "subscribe", "signin"]):
            logger.debug(f"Redirected to auth/paywall URL: {response.url}")
            return None

        html = response.text
        html_lower = html.lower()
        paywall_phrases = [
            "subscribe to read", "subscriber only", "premium article", 
            "sign in to read the full", "story is subscriber exclusive", 
            "join now to read", "unlock this article", "members only",
            "to read the full story", "become a member"
        ]
        if any(phrase in html_lower for phrase in paywall_phrases):
            logger.debug(f"Paywall text detected in HTML for {url}")
            return None

        soup = BeautifulSoup(html, "lxml")
        
        # Remove garbage tags
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe"]):
            element.decompose()

        paragraphs = []
        body_container = None
        for selector in ["article", ".story-body", ".article-body", ".article-content", ".story_details", ".content-area"]:
            body_container = soup.select_one(selector)
            if body_container:
                break
        
        target = body_container if body_container else soup
        
        for p in target.find_all("p"):
            p_class = " ".join(p.get("class", [])).lower()
            p_id = p.get("id", "").lower()
            if any(term in p_class or term in p_id for term in ["comment", "ad", "sidebar", "footer", "nav", "recommend", "related", "widget", "social", "share", "copyright"]):
                continue
            
            p_text = p.get_text().strip()
            if len(p_text) > 30:
                paragraphs.append(p_text)

        if not paragraphs:
            return None
            
        full_text = "\n\n".join(paragraphs)
        if len(full_text) < 400:
            logger.debug(f"Extracted content too short ({len(full_text)} chars) for {url}")
            return None

        return full_text
    except Exception as e:
        logger.debug(f"Error scraping {url}: {e}")
        return None


def fetch_nse_announcements_via_google_news(feed_url: str, limit: int = 30) -> list[dict]:
    """
    Fetches real-time corporate announcements referencing NSE from high-quality sources.
    """
    logger.info("Fetching live NSE announcements...")
    articles = []
    try:
        import feedparser
        import time
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:limit]:
            title = clean_html(entry.get("title", ""))
            summary = clean_html(entry.get("summary", "") or entry.get("description", ""))
            url = entry.get("link", "")
            if not url:
                continue
            
            published_iso = parse_date(entry.get("published", entry.get("updated", "")))
            try:
                published_ts = datetime.fromisoformat(published_iso).timestamp()
            except Exception:
                published_ts = time.time()

            full_text = scrape_full_text(url) or f"{title}. {summary}"

            articles.append({
                "title":        title,
                "summary":      summary,
                "published":    published_iso,
                "published_ts": published_ts,
                "source":       "nse_corporate_announcements",
                "region":       "india",
                "url":          url,
                "author":       "NSE Board",
                "text":         full_text,
            })
    except Exception as e:
        logger.warning(f"Error fetching live NSE announcements: {e}")
    return articles


def fetch_bse_announcements_via_google_news(feed_url: str, limit: int = 30) -> list[dict]:
    """
    Fetches corporate announcements referencing BSE.
    """
    logger.info("Fetching live BSE announcements...")
    articles = []
    try:
        import feedparser
        import time
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:limit]:
            title = clean_html(entry.get("title", ""))
            summary = clean_html(entry.get("summary", "") or entry.get("description", ""))
            url = entry.get("link", "")
            if not url:
                continue

            published_iso = parse_date(entry.get("published", entry.get("updated", "")))
            try:
                published_ts = datetime.fromisoformat(published_iso).timestamp()
            except Exception:
                published_ts = time.time()

            full_text = scrape_full_text(url) or f"{title}. {summary}"

            articles.append({
                "title":        title,
                "summary":      summary,
                "published":    published_iso,
                "published_ts": published_ts,
                "source":       "bse_corporate_announcements",
                "region":       "india",
                "url":          url,
                "author":       "BSE Board",
                "text":         full_text,
            })
    except Exception as e:
        logger.warning(f"Error fetching live BSE announcements: {e}")
    return articles


import socket
socket.setdefaulttimeout(10)


def fetch_from_feed(source_name: str, feed_url: str, limit: int = 30) -> list[dict]:
    if "announcements" in source_name:
        return []
    articles = []
    try:
        feed = feedparser.parse(feed_url)
        logger.info(f"[{source_name}] {len(feed.entries)} entries")

        for entry in feed.entries[:limit]:
            url = entry.get("link", "")
            if not url:
                continue

            title   = clean_html(entry.get("title", ""))
            summary = clean_html(entry.get("summary", "") or entry.get("description", ""))

            published_iso = parse_date(entry.get("published", entry.get("updated", "")))
            try:
                published_ts = datetime.fromisoformat(published_iso).timestamp()
            except Exception:
                published_ts = datetime.now(timezone.utc).timestamp()

            author = entry.get("author", "")
            if not author and entry.get("author_detail"):
                author = entry.get("author_detail", {}).get("name", "")
            if not author:
                author = "Unknown"

            full_text = scrape_full_text(url) or f"{title}. {summary}"

            articles.append({
                "title":        title,
                "summary":      summary,
                "published":    published_iso,
                "published_ts": published_ts,
                "source":       source_name,
                "region":       detect_region(source_name, title, summary),
                "url":          url,
                "author":       author,
                "text":         full_text,
            })
    except Exception as e:
        logger.warning(f"[{source_name}] Feed error: {e}")

    return articles


def fetch_all_articles(
    region: str = "india",
    deduplicate: bool = True,
    days: int = 1,
) -> list[dict]:
    """
    Aggregated fetching + processing pipeline with batch spaCy processing.
    """
    time_filter = f"when:{days}d" if days > 1 else "when:24h"
    adjusted_feeds = {}
    for name, url in INDIAN_FEEDS.items():
        if "when:24h" in url:
            adjusted_feeds[name] = url.replace("when:24h", time_filter)
        else:
            adjusted_feeds[name] = url

    raw_articles = []
    
    # Set dynamic limit based on the query time window (None/no-limit for multi-day backfills, 30 for daily schedules)
    limit = None if days > 1 else 30

    # 1. Ingestion Phase (Fetch feeds and announcements concurrently)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit RSS feed fetching tasks
        future_to_source = {
            executor.submit(fetch_from_feed, source_name, feed_url, limit): source_name
            for source_name, feed_url in adjusted_feeds.items()
            if "announcements" not in source_name
        }
        
        # Submit corporate announcements tasks
        future_nse = executor.submit(fetch_nse_announcements_via_google_news, adjusted_feeds["nse_corporate_announcements"], limit)
        future_bse = executor.submit(fetch_bse_announcements_via_google_news, adjusted_feeds["bse_corporate_announcements"], limit)
        
        # We fetch the new bse_sme_announcements via the same helper, mapping its source to bse_sme_announcements
        def fetch_bse_sme(url):
            res = fetch_bse_announcements_via_google_news(url, limit)
            for r in res:
                r["source"] = "bse_sme_announcements"
            return res
        future_bse_sme = executor.submit(fetch_bse_sme, adjusted_feeds["bse_sme_announcements"])
        
        # Gather feed results
        for future in concurrent.futures.as_completed(future_to_source):
            try:
                raw_articles.extend(future.result())
            except Exception as e:
                source_name = future_to_source[future]
                logger.error(f"Task for source {source_name} failed: {e}")
                
        # Gather corporate announcements
        try:
            raw_articles.extend(future_nse.result())
        except Exception as e:
            logger.error(f"NSE announcements fetch failed: {e}")
            
        try:
            raw_articles.extend(future_bse.result())
        except Exception as e:
            logger.error(f"BSE announcements fetch failed: {e}")

        try:
            raw_articles.extend(future_bse_sme.result())
        except Exception as e:
            logger.error(f"BSE SME announcements fetch failed: {e}")

    # 2. Processing & Enrichment Phase
    enriched_articles = []
    seen_ids = set()

    # Pre-load spaCy model
    nlp = get_spacy_nlp()
    
    # Extract text strings for batch pipeline execution
    texts = [a["text"] for a in raw_articles]
    logger.info(f"Running spaCy batch NER on {len(texts)} articles...")

    batch_resolved_cache = {}
    
    conn = get_db_conn()
    try:
        # Iterate batch processed documents
        for i, doc in enumerate(nlp.pipe(texts, batch_size=50)):
            raw_article = raw_articles[i]
            url = raw_article["url"]
            art_id = make_id(url)

            if deduplicate and art_id in seen_ids:
                continue
            seen_ids.add(art_id)

            title = raw_article["title"]
            summary = raw_article["summary"]
            full_text = texts[i]
            source_name = raw_article["source"]

            # Parse organizations from batch document
            orgs = {ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"}
            companies = set()
            tickers = set()
            sectors = set()
            industries = set()

            # Local cache batch lookup
            for name in orgs:
                if name.upper() in TICKER_STOPWORDS or len(name) < 3 or name.isdigit() or not name[0].isalnum():
                    continue
                
                # Check batch in-memory cache first to avoid redundant SQLite/API queries
                normalized_key = normalize_company_name(name)
                if normalized_key in batch_resolved_cache:
                    resolved = batch_resolved_cache[normalized_key]
                else:
                    resolved = resolve_ticker_via_api(name, conn=conn)
                    batch_resolved_cache[normalized_key] = resolved

                if resolved and resolved.get("ticker"):
                    companies.add(resolved["company_name"])
                    tickers.add(resolved["ticker"])
                    if resolved.get("sector"):
                        sectors.add(resolved["sector"])
                    if resolved.get("industry"):
                        industries.add(resolved["industry"])

            regex_tickers = extract_tickers(full_text)
            all_tickers = sorted(list(set(regex_tickers + list(tickers))))
            event_type = classify_event(title, full_text)
            quality = SOURCE_QUALITY.get(source_name, {"source_tier": 2, "credibility_score": 0.7, "document_type": "news"})

            enriched_articles.append({
                "id":                art_id,
                "title":             title,
                "summary":           summary,
                "published":         raw_article["published"],
                "published_ts":      raw_article["published_ts"],
                "source":            source_name,
                "region":            raw_article["region"],
                "url":               url,
                "tickers":           all_tickers,
                "text":              full_text,
                "source_tier":       quality["source_tier"],
                "credibility_score": quality["credibility_score"],
                "document_type":     quality["document_type"],
                "author":            raw_article["author"],
                "companies":         list(companies),
                "sector":            list(sectors)[0] if sectors else "",
                "industry":          list(industries)[0] if industries else "",
                "event_type":        event_type,
            })
        conn.commit()
    except Exception as e:
        logger.error(f"SQLite batch error: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()

    logger.success(
        f"Fetched and processed {len(enriched_articles)} Indian articles"
    )
    return enriched_articles


if __name__ == "__main__":
    from ingestion.embedder import get_vector_store
    articles = fetch_all_articles()
    print(f"\nFetched: {len(articles)} articles. Embedding and uploading to Qdrant Cloud...")
    store = get_vector_store()
    stored = store.embed_and_store(articles)
    print(f"\nSuccessfully stored {stored} chunks in Qdrant Cloud!")