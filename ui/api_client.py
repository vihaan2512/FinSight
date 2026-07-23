"""
Single place for all HTTP calls from Streamlit → FastAPI.
"""


import os
import requests
import streamlit as st
from typing import Optional, Iterator

def _get_api_base() -> str:
    if os.getenv("FINSIGHT_API_URL"):
        return os.getenv("FINSIGHT_API_URL").rstrip("/")
    try:
        if hasattr(st, "secrets"):
            if "FINSIGHT_API_URL" in st.secrets:
                return str(st.secrets["FINSIGHT_API_URL"]).rstrip("/")
            if "finsight_api_url" in st.secrets:
                return str(st.secrets["finsight_api_url"]).rstrip("/")
    except Exception:
        pass
    return "http://127.0.0.1:8000"

API_BASE = _get_api_base()
TIMEOUT  = 60   


@st.cache_resource
def get_session() -> requests.Session:
    return requests.Session()


def _get(path: str, params: dict = None) -> dict:
    resp = get_session().get(f"{API_BASE}{path}", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict) -> dict:
    resp = get_session().post(f"{API_BASE}{path}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _post_stream(path: str, body: dict) -> Iterator[str]:
    """POST and yield text chunks as they stream in."""
    with get_session().post(
        f"{API_BASE}{path}", json=body,
        stream=True, timeout=TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")


# ── Health / Stats ────────────────────────────────────────────────────────────

def health() -> dict:
    """GET /health — returns status + DB stats."""
    return _get("/health")


def stats() -> dict:
    """GET /stats — returns collection stats."""
    return _get("/stats")


def doc_count() -> int:
    try:
        return stats().get("total_documents", 0)
    except Exception:
        return 0


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest(region: str = "india", days: int = 1) -> dict:
    """POST /ingest — trigger news ingestion."""
    return _post(f"/ingest?region={region}&days={days}", {})


# ── Ask (RAG) ─────────────────────────────────────────────────────────────────

def ask(
    question: str,
    ticker:     Optional[str] = None,
    days:       int           = 1,
    top_k:      int           = 8,
    region:     Optional[str] = None,
    model:      Optional[str] = None,
    use_hybrid: bool          = True,
    context:    Optional[str] = None,
) -> dict:
    """
    POST /ask — returns {"answer": str, "sources": list, "docs_count": int}
    """
    return _post("/ask", {
        "question":   question,
        "ticker":     ticker,
        "days":       days,
        "top_k":      top_k,
        "region":     region,
        "model":      model,
        "use_hybrid": use_hybrid,
        "stream":     False,
        "context":    context,
    })


def ask_stream(
    question: str,
    ticker:     Optional[str] = None,
    days:       int           = 1,
    top_k:      int           = 8,
    region:     Optional[str] = None,
    model:      Optional[str] = None,
    use_hybrid: bool          = True,
    context:    Optional[str] = None,
) -> Iterator[str]:
    """
    POST /ask with stream=True — yields text chunks for Streamlit write_stream.
    """
    return _post_stream("/ask", {
        "question":   question,
        "ticker":     ticker,
        "days":       days,
        "top_k":      top_k,
        "region":     region,
        "model":      model,
        "use_hybrid": use_hybrid,
        "stream":     True,
        "context":    context,
    })


def retrieve(
    question: str,
    ticker:     Optional[str] = None,
    days:       int           = 1,
    top_k:      int           = 8,
    region:     Optional[str] = None,
    use_hybrid: bool          = True,
) -> dict:
    """
    POST /retrieve — returns {"docs": list, "sources": list, "context": str}
    """
    return _post("/retrieve", {
        "question":   question,
        "ticker":     ticker,
        "days":       days,
        "top_k":      top_k,
        "region":     region,
        "use_hybrid": use_hybrid,
    })


def warmup() -> dict:
    """POST /warmup — triggers cross-encoder model loading and prediction."""
    return _post("/warmup", {})


# ── Sentiment ─────────────────────────────────────────────────────────────────

def sentiment_for_ticker(ticker: str, docs: list[dict]) -> dict:
    """POST /sentiment — returns sentiment score for one ticker."""
    result = _post("/sentiment", {"ticker": ticker, "docs": docs})
    return result.get("sentiment", {})


def sentiment_for_docs(docs: list[dict]) -> dict:
    """POST /sentiment — returns top-5 ticker sentiments from a set of docs."""
    result = _post("/sentiment", {"docs": docs})
    return result.get("sentiments", {})



# ── Market summaries ──────────────────────────────────────────────────────────

def india_summary(watchlist_only: bool = False, username: Optional[str] = None) -> dict:
    """GET /india — summary + sentiments + articles for Indian market."""
    return _get("/india", params={"watchlist_only": watchlist_only, "username": username})



# ── News feed ─────────────────────────────────────────────────────────────────

def feed(
    query:  str           = "latest financial market news",
    days:   int           = 1,
    top_k:  int           = 20,
    region: Optional[str] = None,
    watchlist_only: bool  = False,
    username: Optional[str] = None,
) -> dict:
    """POST /feed — browse/search indexed articles."""
    return _post("/feed", {
        "query":  query,
        "days":   days,
        "top_k":  top_k,
        "region": region,
        "watchlist_only": watchlist_only,
        "username": username,
    })


# ── Recent ticker ─────────────────────────────────────────────────────────────

def recent_ticker(ticker: str, limit: int = 5) -> dict:
    """GET /recent/{ticker} — recent articles + summary for a ticker."""
    return _get(f"/recent/{ticker}", params={"limit": limit})


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_eval(quick: bool = True, days: int = 30) -> dict:
    """GET /evaluate — runs eval suite, returns metrics."""
    return _get("/evaluate", params={"quick": quick, "days": days})


# ── Watchlist Management & Brief Client ───────────────────────────────────────

def _delete(path: str, body: dict) -> dict:
    resp = get_session().delete(f"{API_BASE}{path}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_watchlist(username: str) -> list[str]:
    """GET /watchlist — retrieve watchlist for user."""
    try:
        res = _get("/watchlist", params={"username": username})
        return res.get("watchlist", [])
    except Exception:
        return []


def add_to_watchlist(username: str, ticker: str) -> bool:
    """POST /watchlist — add ticker to watchlist."""
    try:
        res = _post("/watchlist", {"username": username, "ticker": ticker})
        return res.get("status") == "success"
    except Exception:
        return False


def remove_from_watchlist(username: str, ticker: str) -> bool:
    """DELETE /watchlist — remove ticker from watchlist."""
    try:
        res = _delete("/watchlist", {"username": username, "ticker": ticker})
        return res.get("status") == "success"
    except Exception:
        return False


def get_watchlist_brief(
    username: str,
    model: Optional[str] = None,
    stream: bool = False,
    hours: int = 24,
):
    """POST /watchlist/brief — generate custom brief for user watchlist."""
    if stream:
        return _post_stream("/watchlist/brief", {
            "username": username,
            "model": model,
            "stream": True,
            "hours": hours,
        })
    else:
        return _post("/watchlist/brief", {
            "username": username,
            "model": model,
            "stream": False,
            "hours": hours,
        })


# ── Portfolio API ─────────────────────────────────────────────────────────────

def get_portfolio(username: str) -> dict[str, float]:
    """GET /portfolio — retrieve portfolio for user."""
    try:
        res = _get("/portfolio", params={"username": username})
        return res.get("portfolio", {})
    except Exception:
        return {}


def add_to_portfolio(username: str, ticker: str, weight: float) -> bool:
    """POST /portfolio — add/update ticker weight in portfolio."""
    try:
        res = _post("/portfolio", {"username": username, "ticker": ticker, "weight": weight})
        return res.get("status") == "success"
    except Exception:
        return False


def remove_from_portfolio(username: str, ticker: str) -> bool:
    """DELETE /portfolio — remove ticker from portfolio."""
    try:
        res = _delete("/portfolio", {"username": username, "ticker": ticker})
        return res.get("status") == "success"
    except Exception:
        return False


def get_portfolio_analysis(username: str) -> dict:
    """GET /portfolio/analysis — run portfolio exposure and estimated impact analysis."""
    try:
        return _get("/portfolio/analysis", params={"username": username})
    except Exception as e:
        return {"status": "error", "portfolio_impact": "0.0%", "exposure": [], "reason": str(e)}


def get_event_impact(event_query: str) -> list[str]:
    """POST /event-impact — generate macro event causal chain."""
    try:
        res = _post("/event-impact", {"event_query": event_query})
        return res.get("chain", [])
    except Exception:
        return [event_query, "Market shifts", "Operational costs rise", "Profits compress"]