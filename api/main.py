"""
FinSight FastAPI backend.
Endpoints:
  GET  /health                  — health + DB stats
  GET  /stats                   — collection stats
  POST /ingest                  — trigger ingestion
  POST /ask                     — RAG answer (streaming or JSON)
  POST /sentiment               — sentiment scores for retrieved docs
  GET  /india                   — Indian market summary
  GET  /feed                    — browse/search indexed articles
  GET  /evaluate                — run evaluation suite
"""


from contextlib import asynccontextmanager
from typing import Optional
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from config import get_settings
from ingestion.rss_fetcher import fetch_all_articles
from ingestion.embedder import get_vector_store
from retrieval.vector_store import retrieve, get_recent_by_ticker, format_context
from retrieval.hybrid_search import hybrid_retrieve
from retrieval.sentiment import score_sentiment, score_multiple_tickers
from api.llm import ask_groq, summarize_articles
from evaluation.evaluator import run_evaluation

settings = get_settings()
scheduler = AsyncIOScheduler()


import datetime

_cached_doc_count = 0
_last_ingest_time = None
try:
    if os.path.exists("last_ingest.txt"):
        with open("last_ingest.txt", "r") as f:
            _last_ingest_time = f.read().strip() or None
except Exception:
    pass

def update_cached_doc_count():
    global _cached_doc_count
    try:
        store = get_vector_store()
        stats = store.get_collection_stats()
        _cached_doc_count = stats.get("total_documents", 0)
    except Exception as e:
        logger.warning(f"Failed to update cached doc count: {e}")

def run_ingestion(region: str = "india", days: int = 1):
    global _last_ingest_time
    logger.info(f"Ingestion starting (days={days})...")
    try:
        articles = fetch_all_articles(region=region, days=days)
        store = get_vector_store()
        stored = store.embed_and_store(articles)
        update_cached_doc_count()
        # Update last ingestion time
        now = datetime.datetime.now()
        _last_ingest_time = now.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open("last_ingest.txt", "w") as f:
                f.write(_last_ingest_time)
        except Exception:
            pass

        try:
            from retrieval.hybrid_search import hybrid_retrieve
            from retrieval.vector_store import retrieve
            hybrid_retrieve.cache_clear()
            retrieve.cache_clear()
            logger.info("Cleared retrieval caches post-ingestion.")
        except Exception as ce:
            logger.warning(f"Failed to clear cache: {ce}")

        return len(articles), stored
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 0, 0


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FinSight API starting...")
    update_cached_doc_count()
    logger.info(f"DB initial doc count cached: {_cached_doc_count}")

    if _cached_doc_count == 0:
        logger.info("Empty DB — running initial ingestion in background...")
        import asyncio
        asyncio.create_task(asyncio.to_thread(run_ingestion))

    try:
        logger.info("Warming up cross-encoder model...")
        from retrieval.hybrid_search import get_cross_encoder
        ce = get_cross_encoder()
        ce.predict([("warmup query", "warmup document text")])
        logger.info("Cross-encoder model warmed up successfully.")
    except Exception as e:
        logger.warning(f"Failed to warm up cross-encoder model: {e}")

    scheduler.add_job(run_ingestion, "interval", minutes=settings.ingest_interval_minutes)
    scheduler.start()
    yield
    scheduler.shutdown()
    logger.info("FinSight API stopped")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinSight API",
    description="Real-Time Finance News RAG — India",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    ticker:   Optional[str] = None
    days:     int           = 1
    top_k:    int           = 8
    region:   Optional[str] = None
    stream:   bool          = False   
    model:    Optional[str] = None
    use_hybrid: bool        = True    
    context:  Optional[str] = None


class RetrieveRequest(BaseModel):
    question: str
    ticker:   Optional[str] = None
    days:     int           = 1
    top_k:    int           = 8
    region:   Optional[str] = None
    use_hybrid: bool        = True


class SentimentRequest(BaseModel):
    docs: list[dict]
    ticker: Optional[str] = None     
    

class TickerRequest(BaseModel):
    ticker: str
    days:   int           = 1
    top_k:  int           = 8
    region: Optional[str] = None
    model:  Optional[str] = None


class FeedRequest(BaseModel):
    query:  str           = "latest financial market news"
    days:   int           = 1
    top_k:  int           = 20
    region: Optional[str] = None
    watchlist_only: bool  = False
    username: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    doc_count = _cached_doc_count
    try:
        store = get_vector_store()
        stats = store.get_collection_stats()
        doc_count = stats.get("total_documents", 0)
    except Exception:
        pass
    return {
        "status": "ok",
        "groq_model": settings.groq_model,
        "db": {
            "total_documents": doc_count,
            "last_ingest_time": _last_ingest_time
        }
    }


@app.get("/stats")
async def stats():
    return get_vector_store().get_collection_stats()


@app.post("/ingest")
async def ingest(background_tasks: BackgroundTasks, region: str = "india", days: int = 1):
    """Trigger ingestion."""
    background_tasks.add_task(run_ingestion, region, days)
    return {"status": "ok", "message": "Ingestion started in the background"}


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Main RAG endpoint. Uses hybrid search by default.
    Returns streaming text (if stream=True) or JSON.
    """
    if req.context is not None:
        context = req.context
        sources = []
        docs_count = 0
    else:
        # Retrieve
        if req.use_hybrid:
            docs = hybrid_retrieve(
                req.question, ticker=req.ticker, days=req.days,
                top_k=req.top_k, region=req.region, use_reranker=True,
            )
        else:
            docs = retrieve(req.question, req.ticker, req.days, req.top_k, req.region)

        if not docs:
            return {
                "answer": "No relevant news found. Try increasing the days window or rephrasing.",
                "sources": [],
                "docs_count": 0,
            }

        context = format_context(docs)
        sources = []
        seen_urls = set()
        for d in docs:
            for src in d.get("sources", []):
                url = src.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "title":     src.get("title", ""),
                        "source":    src.get("source", ""),
                        "region":    src.get("region", ""),
                        "url":       url,
                        "published": src.get("published", "")[:10],
                        "tickers":   src.get("tickers", ""),
                    })
        docs_count = len(docs)

    if req.stream:
        return StreamingResponse(
            ask_groq(req.question, context, req.ticker, req.model, stream=True),
            media_type="text/plain",
        )

    answer = ask_groq(req.question, context, req.ticker, req.model, stream=False)
    return {"answer": answer, "sources": sources, "docs_count": docs_count}


@app.post("/retrieve")
async def retrieve_endpoint(req: RetrieveRequest):
    """Retrieve articles without running LLM generation."""
    if req.use_hybrid:
        docs = hybrid_retrieve(
            req.question, ticker=req.ticker, days=req.days,
            top_k=req.top_k, region=req.region, use_reranker=True,
        )
    else:
        docs = retrieve(req.question, req.ticker, req.days, req.top_k, req.region)

    sources = []
    seen_urls = set()
    for d in docs:
        for src in d.get("sources", []):
            url = src.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title":     src.get("title", ""),
                    "source":    src.get("source", ""),
                    "region":    src.get("region", ""),
                    "url":       url,
                    "published": src.get("published", "")[:10],
                    "tickers":   src.get("tickers", ""),
                })
    context = format_context(docs)
    return {"docs": docs, "sources": sources, "context": context}


@app.post("/warmup")
async def warmup():
    """Warm up the cross-encoder."""
    try:
        from retrieval.hybrid_search import get_cross_encoder
        ce = get_cross_encoder()
        ce.predict([("warmup query", "warmup document text")])
        return {"status": "ok", "message": "Cross-encoder warmed up"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/sentiment")
async def sentiment(req: SentimentRequest):
    """
    Score sentiment for retrieved docs.
    Pass ticker to score a specific one, or get top-5 tickers.
    """
    if req.ticker:
        result = score_sentiment(req.ticker, req.docs)
        return {"ticker": req.ticker, "sentiment": result}
    else:
        results = score_multiple_tickers(req.docs)
        return {"sentiments": results}



@app.get("/india")
async def india_summary(watchlist_only: bool = False, username: Optional[str] = None):
    if watchlist_only and username:
        from retrieval.watchlist_db import get_user_id
        from retrieval.vector_store import retrieve_watchlist_stories
        user_id = get_user_id(username)
        raw_docs = retrieve_watchlist_stories(user_id, hours=24, top_k=25)
        docs = [d for d in raw_docs if d.get("region") == "india"]
    else:
        docs = retrieve(
            "Indian stock market NSE BSE Nifty Sensex",
            region="india", days=1, top_k=25,
        )
    if not docs:
        return {
            "region": "india", "articles_found": 0,
            "summary": "No Indian market news found.",
            "articles": []
        }

    summary = summarize_articles(docs)
    articles = [
        {"title": d.get("title",""), "source": d.get("source",""),
         "published": d.get("published","")[:10], "url": d.get("url","")}
        for d in docs[:8]
    ]
    return {
        "region": "india",
        "articles_found": len(docs),
        "summary": summary,
        "articles": articles,
    }


@app.post("/feed")
async def feed(req: FeedRequest):
    """Browse/search indexed articles."""
    if req.watchlist_only and req.username:
        from retrieval.watchlist_db import get_user_id
        from retrieval.vector_store import retrieve_watchlist_stories
        user_id = get_user_id(req.username)
        docs = retrieve_watchlist_stories(user_id, hours=req.days * 24, top_k=req.top_k)
        if req.region:
            docs = [d for d in docs if d.get("region") == req.region]
    else:
        docs = retrieve(req.query, days=req.days, top_k=req.top_k, region=req.region)
        
    return {
        "docs_found": len(docs),
        "articles": [
            {
                "title":    d.get("title",""),
                "source":   d.get("source") or (d["sources"][0].get("source", "") if d.get("sources") else ""),
                "region":   d.get("region",""),
                "published":d.get("published","")[:10],
                "url":      d.get("url") or (d["sources"][0].get("url", "") if d.get("sources") else ""),
                "summary":  d.get("summary","")[:200],
                "tickers":  d.get("tickers",""),
            }
            for d in docs
        ],
    }


@app.get("/evaluate")
async def evaluate(
    quick: bool = Query(False),
    days:  int  = Query(30),
    top_k: int  = Query(8),
):
    agg, results = run_evaluation(quick=quick, days=days, top_k=top_k, save=True)
    return {"status": "ok", "summary": agg, "results": results, "n_queries": len(results)}


# ── Watchlist Management & Brief Endpoints ────────────────────────────────────

class WatchlistRequest(BaseModel):
    username: str
    ticker: Optional[str] = None


class WatchlistBriefRequest(BaseModel):
    username: str
    model: Optional[str] = None
    stream: bool = False
    hours: int = 24


@app.get("/watchlist")
async def get_user_watchlist(username: str):
    from retrieval.watchlist_db import get_user_id, get_watchlist
    user_id = get_user_id(username)
    watchlist = get_watchlist(user_id)
    return {"watchlist": watchlist}


@app.post("/watchlist")
async def add_to_user_watchlist(req: WatchlistRequest):
    from retrieval.watchlist_db import get_user_id, add_to_watchlist
    if not req.ticker:
        return {"status": "error", "message": "Ticker is required"}
    user_id = get_user_id(req.username)
    
    # Resolve company name or alias to official Yahoo ticker symbol
    from ingestion.rss_fetcher import resolve_ticker_via_api
    resolved = resolve_ticker_via_api(req.ticker)
    ticker_to_save = req.ticker
    if resolved and resolved.get("ticker"):
        ticker_to_save = resolved["ticker"]
        
    success = add_to_watchlist(user_id, ticker_to_save)
    return {"status": "success" if success else "error"}


@app.delete("/watchlist")
async def remove_from_user_watchlist(req: WatchlistRequest):
    from retrieval.watchlist_db import get_user_id, remove_from_watchlist
    if not req.ticker:
        return {"status": "error", "message": "Ticker is required"}
    user_id = get_user_id(req.username)
    success = remove_from_watchlist(user_id, req.ticker)
    return {"status": "success" if success else "error"}


@app.post("/watchlist/brief")
async def get_watchlist_brief(req: WatchlistBriefRequest):
    from retrieval.watchlist_db import get_user_id
    from retrieval.vector_store import retrieve_watchlist_stories
    
    user_id = get_user_id(req.username)
    docs = retrieve_watchlist_stories(user_id, hours=req.hours)
    
    if not docs:
        if req.stream:
            async def empty_generator():
                yield "No watchlist updates found today. Watchlist might be empty, or there is no recent news."
            return StreamingResponse(empty_generator(), media_type="text/plain")
        return {
            "answer": "No watchlist updates found today. Watchlist might be empty, or there is no recent news.",
            "sources": [],
            "docs_count": 0
        }
        
    context = format_context(docs)
    
    # Flatten sources for UI compatibility
    sources = []
    seen_urls = set()
    for d in docs:
        for src in d.get("sources", []):
            url = src.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title":     src.get("title", ""),
                    "source":    src.get("source", ""),
                    "region":    src.get("region", ""),
                    "url":       url,
                    "published": src.get("published", "")[:10],
                    "tickers":   src.get("tickers", ""),
                })

    watchlist_str = ", ".join(docs[0].get("tickers", [])) if docs else ""
    from retrieval.watchlist_db import get_watchlist
    watchlist_tickers = get_watchlist(user_id)
    watchlist_str = ", ".join(watchlist_tickers)

    brief_query = (
        f"Provide a daily brief for the following specific watchlist companies: {watchlist_str}.\n"
        f"For each company that has updates in the text, output EXACTLY the following structure and nothing else:\n\n"
        f"### [Company Ticker/Name]\n"
        f"* **Key Points**:\n"
        f"  - [Key factual point 1]\n"
        f"  - [Key factual point 2]\n"
        f"* **Sentiment**: [Positive/Negative/Neutral]\n"
        f"* **Sources**: [[Source Name](URL)] (Use the exact URL provided for the article in the context)\n\n"
        f"Do NOT include any conversational intro, outro, direct answer remarks, or generic explanation.\n"
        f"Do NOT include any sections about 'risks', 'catalysts', or other generic analysis.\n"
        f"Do NOT include updates, key points, or information for any other companies not in the watchlist: {watchlist_str}."
    )
    
    if req.stream:
        return StreamingResponse(
            ask_groq(brief_query, context, model=req.model, stream=True),
            media_type="text/plain"
        )
        
    answer = ask_groq(brief_query, context, model=req.model, stream=False)
    return {"answer": answer, "sources": sources, "docs_count": len(docs)}


# ── Portfolio Management & Exposure Endpoints ─────────────────────────────────

class PortfolioRequest(BaseModel):
    username: str
    ticker:   Optional[str] = None
    weight:   Optional[float] = None


class EventImpactRequest(BaseModel):
    event_query: str


@app.get("/portfolio")
async def get_user_portfolio(username: str):
    from retrieval.watchlist_db import get_user_id, get_portfolio
    user_id = get_user_id(username)
    portfolio = get_portfolio(user_id)
    return {"portfolio": portfolio}


@app.post("/portfolio")
async def add_to_user_portfolio(req: PortfolioRequest):
    from retrieval.watchlist_db import get_user_id, add_to_portfolio
    if not req.ticker or req.weight is None:
        return {"status": "error", "message": "Ticker and weight are required"}
    user_id = get_user_id(req.username)
    
    # Resolve company name or alias to official Yahoo ticker symbol
    from ingestion.rss_fetcher import resolve_ticker_via_api
    resolved = resolve_ticker_via_api(req.ticker)
    ticker_to_save = req.ticker
    if resolved and resolved.get("ticker"):
        ticker_to_save = resolved["ticker"]
        
    success = add_to_portfolio(user_id, ticker_to_save, req.weight)
    return {"status": "success" if success else "error"}


@app.delete("/portfolio")
async def remove_from_user_portfolio(req: PortfolioRequest):
    from retrieval.watchlist_db import get_user_id, remove_from_portfolio
    if not req.ticker:
        return {"status": "error", "message": "Ticker is required"}
    user_id = get_user_id(req.username)
    success = remove_from_portfolio(user_id, req.ticker)
    return {"status": "success" if success else "error"}


@app.get("/portfolio/analysis")
async def get_portfolio_analysis(username: str):
    from retrieval.watchlist_db import get_user_id, get_portfolio
    from retrieval.vector_store import retrieve
    from retrieval.sentiment import score_sentiment
    from groq import Groq
    import re
    import json

    user_id = get_user_id(username)
    portfolio = get_portfolio(user_id)
    if not portfolio:
        return {
            "status": "success",
            "portfolio_impact": 0.0,
            "exposure": [],
            "reason": "Portfolio is empty. Please add stocks and weights to analyze."
        }

    exposure = []
    weighted_sentiment_sum = 0.0
    total_weight = 0.0

    for ticker, weight in portfolio.items():
        from retrieval.vector_store import retrieve_ticker_stories
        docs = retrieve_ticker_stories(ticker, days=1, top_k=10)

        sentiment = score_sentiment(ticker, docs)
        score = sentiment["score"]

        if score >= 0.15:
            impact = "affected positively 🟢"
        elif score <= -0.15:
            impact = "affected negatively 🔴"
        else:
            impact = "neutral 🟡"

        exposure.append({
            "ticker": ticker,
            "weight": weight,
            "sentiment_score": score,
            "impact": impact,
            "reason": sentiment.get("reason", "No recent major news.")
        })

        weighted_sentiment_sum += score * weight
        total_weight += weight

    portfolio_impact_shift = (weighted_sentiment_sum / total_weight) if total_weight > 0 else 0.0

    reason = "No data"
    if settings.groq_api_key and exposure:
        client = Groq(api_key=settings.groq_api_key)
        exposure_summary_text = "\n".join([
            f"- {item['ticker']} ({item['weight']:.0%}): score {item['sentiment_score']:.2f} ({item['impact']}) - {item['reason']}"
            for item in exposure
        ])

        prompt = f"""You are a portfolio intelligence advisor. 
Here is a client's portfolio exposure analysis:
{exposure_summary_text}

Overall estimated sentiment shift: {portfolio_impact_shift:+.2%}

Provide a brief, professional two-sentence summary of the portfolio's current exposure risk and the main driver of its sentiment shift."""
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0
            )
            reason = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Portfolio exposure LLM summary failed: {e}")
            reason = f"Weighted sentiment shift is {portfolio_impact_shift:+.2%}. Main drivers are your allocated assets."
    else:
        reason = f"Weighted sentiment shift is {portfolio_impact_shift:+.2%}."

    formatted_shift = f"{portfolio_impact_shift * 10:+.1f}%"

    return {
        "status": "success",
        "portfolio_impact": formatted_shift,
        "exposure": exposure,
        "reason": reason
    }


@app.post("/event-impact")
async def get_event_impact(req: EventImpactRequest):
    """Dynamically generate a causal chain of impact for a macro event or ticker."""
    from groq import Groq
    import re
    import json
    
    if not settings.groq_api_key:
        return {"chain": [req.event_query, "Market shifts", "Operational costs rise", "Profits compress"]}
        
    client = Groq(api_key=settings.groq_api_key)
    prompt = f"""You are a macroeconomic analyst.
Generate a logical causal impact chain for the following event or company query: '{req.event_query}'.

For example, for 'Oil Price Increase', the chain is:
Oil Price ↑ ➔ Airline costs ↑ ➔ Margins ↓ ➔ Airline stocks ↓

For 'NVIDIA demand', the chain is:
AI chip demand ↑ ➔ NVIDIA revenue ↑ ➔ GPU supply constraint ➔ High ASPs ➔ Net Margins ↑

Provide exactly 4 steps in the chain. Each step should be a very short phrase (1-3 words) with arrows (↑ or ↓) indicating direction if appropriate.
Respond ONLY with a JSON object containing a list of strings:
{{
  "chain": ["Step 1", "Step 2", "Step 3", "Step 4"]
}}"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        return {"chain": data.get("chain", [])}
    except Exception as e:
        logger.warning(f"Failed to generate event impact chain: {e}")
        return {"chain": [req.event_query, "Market impact", "Operational costs change", "Asset price shift"]}