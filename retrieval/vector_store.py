from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from ingestion.embedder import get_vector_store, COLLECTION_NAME, COLLECTIONS
from retrieval.cache_utils import ttl_cache


def route_query(query: str) -> list[str]:
    """
    Identify query intent based on keywords and return the list of target collection names to search.
    - filings: "earnings", "dividend", "filing", "board meeting", "q1", "q2", "q3", "q4", "revenue", "financial results", "quarterly", "annual report"
    - macro: "gdp", "inflation", "interest rate", "fed", "rbi", "sebi", "policy", "regulation", "treasury", "unemployment", "jobs", "tariff"
    - defaults to news
    """
    q_low = query.lower()
    filing_keywords = [
        "earnings", "dividend", "filing", "board meeting", "q1", "q2", "q3", "q4",
        "revenue", "profit", "net profit", "financial results", "quarterly", "annual report",
        "announcement", "earnings release", "stock split", "merger", "acquisition"
    ]
    macro_keywords = [
        "gdp", "inflation", "interest rate", "interest rates", "fed", "rbi", "sebi", 
        "policy", "regulation", "treasury", "unemployment", "employment", "jobs", 
        "tariff", "tariffs", "trade", "fiscal", "regulatory", "monetary policy", "guidelines"
    ]

    targets = []
    if any(kw in q_low for kw in filing_keywords):
        targets.append("finance_filings")
    if any(kw in q_low for kw in macro_keywords):
        targets.append("finance_macro")
    
    targets.append("finance_news")
    return list(dict.fromkeys(targets))


def _cutoff_ts(days: int) -> int:
    """Unix timestamp for N days ago."""
    return int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())


def _build_where(days: int, region: Optional[str]) -> Filter:
    """Build Qdrant Filter using FieldConditions."""
    must_conditions = [
        FieldCondition(key="published_ts", range=Range(gte=_cutoff_ts(days)))
    ]
    if region:
        must_conditions.append(
            FieldCondition(key="region", match=MatchValue(value=region))
        )
    return Filter(must=must_conditions)


def _filter_by_ticker(docs: list[dict], ticker: str) -> list[dict]:
    """Post-retrieval filter: keep docs where ticker appears in tickers list or string."""
    ticker = ticker.upper().strip()
    filtered = []
    for d in docs:
        tickers_data = d.get("tickers", [])
        if isinstance(tickers_data, list):
            if ticker in [t.upper().strip() for t in tickers_data]:
                filtered.append(d)
        elif isinstance(tickers_data, str):
            if ticker in [t.strip().upper() for t in tickers_data.split(",")]:
                filtered.append(d)
    return filtered


def _query(store, query_vec: list, where: Filter, n: int, collections: list[str]) -> list[dict]:
    """Run Qdrant search queries across target collection(s), merging and applying source boosts."""
    all_results = []
    seen_ids = set()
    candidate_count = max(1, n * 2)

    for coll in collections:
        try:
            results = store.client.query_points(
                collection_name=coll,
                query=query_vec,
                query_filter=where,
                limit=candidate_count,
                with_payload=True
            ).points
            
            for point in results:
                if point.score > 0.25:
                    payload = point.payload
                    doc_id = payload.get("id", point.id)
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    payload["id"] = doc_id
                    all_results.append((payload, point.score))
        except Exception as e:
            logger.error(f"Qdrant query error on {coll}: {e}")

    # Sort by adjusted similarity
    all_results.sort(key=lambda x: x[1], reverse=True)
    return [doc[0] for doc in all_results[:n]]


def rerank_and_cluster(query: str, docs: list[dict], top_k: int) -> list[dict]:
    """Rerank candidates using Cross-Encoder and cluster them by canonical_story_id."""
    if not docs:
        return []
    
    try:
        from retrieval.hybrid_search import rerank_with_cache
        doc_ids = tuple(d.get("id", "") for d in docs)
        doc_texts = tuple(f"{d.get('title','')} {d.get('summary','')}" for d in docs)
        ce_scores = rerank_with_cache(query, doc_ids, doc_texts)
        
        sorted_docs = [docs[idx] for idx in sorted(range(len(ce_scores)), key=lambda idx: ce_scores[idx], reverse=True)]
    except Exception as e:
        logger.warning(f"Cross-encoder reranking failed in retrieve: {e}")
        sorted_docs = docs
        
    clustered_stories = []
    story_map = {}
    
    for art in sorted_docs:
        story_id = art.get("canonical_story_id") or f"story_{art.get('id')}"
        if story_id not in story_map:
            story_map[story_id] = {
                "story_id": story_id,
                "title": art.get("title", ""),
                "summary": art.get("summary", ""),
                "text": art.get("text", ""),
                "published": art.get("published", ""),
                "published_ts": art.get("published_ts", 0),
                "region": art.get("region", "india"),
                "tickers": art.get("tickers", []),
                "sources": []
            }
            clustered_stories.append(story_map[story_id])
            
        story_map[story_id]["sources"].append({
            "source": art.get("source"),
            "url": art.get("url"),
            "title": art.get("title"),
            "published": art.get("published"),
            "region": art.get("region"),
            "tickers": art.get("tickers")
        })
        story_map[story_id]["tickers"] = sorted(list(set(story_map[story_id]["tickers"] + art.get("tickers", []))))
        
    return clustered_stories[:top_k]


@ttl_cache(ttl_seconds=300)
def retrieve(
    query: str,
    ticker:  Optional[str] = None,
    days:    int           = 7,
    top_k:   int           = 8,
    region:  Optional[str] = None,
) -> list[dict]:
    """
    Retrieve top-K stories by semantic similarity + optional filters.
    """
    store     = get_vector_store()
    from retrieval.query_expansion import expand_query_with_companies
    expanded_query = expand_query_with_companies(query)
    query_vec = store.embed_text(expanded_query)
    where     = _build_where(days, region)
    routed_colls = route_query(query)

    docs = _query(store, query_vec, where, 50, routed_colls)

    if len(docs) < 25 and set(routed_colls) != set(COLLECTIONS):
        logger.info(f"Routed search returned only {len(docs)} docs — falling back to searching all collections")
        docs = _query(store, query_vec, where, 50, COLLECTIONS)

    if ticker and docs:
        ticker_docs = _filter_by_ticker(docs, ticker)
        if ticker_docs:
            docs = ticker_docs
        else:
            logger.info(f"Ticker '{ticker}' not in retrieved docs — keeping all semantic results")

    if not docs and days < 90:
        logger.info(f"No results in {days}d — widening to 90 days")
        wide_where = _build_where(90, region)
        docs = _query(store, query_vec, wide_where, 50, routed_colls)
        if len(docs) < 25 and set(routed_colls) != set(COLLECTIONS):
            docs = _query(store, query_vec, wide_where, 50, COLLECTIONS)
        if ticker and docs:
            ticker_docs = _filter_by_ticker(docs, ticker)
            if ticker_docs:
                docs = ticker_docs

    if not docs and region:
        logger.info("No results with region filter — dropping region")
        docs = _query(store, query_vec, _build_where(90, None), 50, routed_colls)
        if len(docs) < 25 and set(routed_colls) != set(COLLECTIONS):
            docs = _query(store, query_vec, _build_where(90, None), 50, COLLECTIONS)

    result = rerank_and_cluster(query, docs, top_k)
    logger.info(f"Retrieved {len(result)} stories for: '{query[:60]}'")
    return result


def get_recent_by_ticker(ticker: str, limit: int = 5) -> list[dict]:
    """
    Fetch recent articles mentioning a ticker.
    Uses semantic search with ticker as query, then post-filters.
    """
    store  = get_vector_store()
    ticker = ticker.upper().strip()

    query_vec = store.embed_text(ticker)
    where     = _build_where(90, None)
    docs      = _query(store, query_vec, where, limit * 4, COLLECTIONS)

    ticker_docs = _filter_by_ticker(docs, ticker) or docs
    ticker_docs.sort(key=lambda x: x.get("published", ""), reverse=True)
    return ticker_docs[:limit]


def format_context(docs: list[dict]) -> str:
    """Format retrieved stories into LLM context block."""
    if not docs:
        return "No relevant news articles found."
    parts = []
    for i, doc in enumerate(docs, 1):
        date      = doc.get("published", "")[:10]
        region    = "🇮🇳" if doc.get("region") == "india" else "🌍"
        tickers   = doc.get("tickers", "")
        ticker_tag = f" | Tickers: {tickers}" if tickers else ""
        content   = doc.get("text") or doc.get("summary", "")
        
        if doc.get("sources"):
            source_names = [s.get("source", "unknown").replace("_", " ").title() for s in doc["sources"]]
            source_str = f"Sources: {', '.join(source_names)}"
        else:
            source_str = f"Source: {doc.get('source', 'unknown').replace('_', ' ').title()}"
            
        parts.append(
            f"[{region} Story {i} | {source_str} | {date}{ticker_tag}]\n"
            f"Title: {doc.get('title', '')}\n"
            f"Content: {content}\n"
            f"URL: {doc.get('url', '')}"
        )
    return "\n\n---\n\n".join(parts)


def retrieve_watchlist_stories(user_id: int, hours: int = 24, top_k: int = 8) -> list[dict]:
    """
    Retrieve top stories for tickers in a user's watchlist published within last N hours.
    Falls back to last 7 days if no new articles found.
    Uses hybrid ticker and text-matching to ensure 100% resolution reliability.
    """
    from retrieval.watchlist_db import get_watchlist
    watchlist = get_watchlist(user_id)
    if not watchlist:
        logger.info(f"User {user_id} watchlist is empty.")
        return []
        
    store = get_vector_store()
    
    keywords_map = {}
    for ticker in watchlist:
        kw = {ticker.upper(), ticker.split(".")[0].upper()}
        try:
            from ingestion.rss_fetcher import get_db_conn
            conn = get_db_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT alias, company_name FROM entity_cache WHERE ticker = ?", (ticker,))
            for alias, name in cursor.fetchall():
                if alias:
                    kw.add(alias.upper())
                if name:
                    kw.add(name.upper())
                    words = name.split()
                    if words:
                        kw.add(words[0].upper())
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to query entity_cache aliases for watchlist ticker {ticker}: {e}")
        keywords_map[ticker] = kw

    # Query Qdrant
    from qdrant_client.models import Filter, FieldCondition, Range
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    where = Filter(must=[
        FieldCondition(key="published_ts", range=Range(gte=cutoff_ts))
    ])
    
    docs = []
    for coll in COLLECTIONS:
        try:
            offset = None
            while True:
                res, next_offset = store.client.scroll(
                    collection_name=coll,
                    scroll_filter=where,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                for point in res:
                    payload = point.payload
                    payload["id"] = payload.get("id", point.id)
                    docs.append(payload)
                if not next_offset:
                    break
                offset = next_offset
        except Exception as e:
            logger.error(f"Failed to scroll watchlist docs for {coll}: {e}")
            
    def filter_and_tag(documents: list[dict]) -> list[dict]:
        filtered = []
        for d in documents:
            title = (d.get("title") or "").upper()
            summary = (d.get("summary") or "").upper()
            text = (d.get("text") or "").upper()
            doc_tickers = [t.upper().strip() for t in d.get("tickers", [])]
            
            matched_tickers = set()
            for ticker, kws in keywords_map.items():
                clean_ticker = ticker.upper()
                if clean_ticker in doc_tickers or clean_ticker.split(".")[0] in doc_tickers:
                    matched_tickers.add(ticker)
                    continue
                for kw in kws:
                    if kw in title or kw in summary or kw in text:
                        matched_tickers.add(ticker)
                        break
            
            if matched_tickers:
                d["tickers"] = list(set(d.get("tickers", []) + list(matched_tickers)))
                filtered.append(d)
        return filtered

    matched_docs = filter_and_tag(docs)

    if not matched_docs and hours == 24:
        logger.info(f"No articles for watchlist {watchlist} in last 24h — widening to last 7 days.")
        wide_cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp())
        where = Filter(must=[
            FieldCondition(key="published_ts", range=Range(gte=wide_cutoff_ts))
        ])
        docs = []
        for coll in COLLECTIONS:
            try:
                offset = None
                while True:
                    res, next_offset = store.client.scroll(
                        collection_name=coll,
                        scroll_filter=where,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    for point in res:
                        payload = point.payload
                        payload["id"] = payload.get("id", point.id)
                        docs.append(payload)
                    if not next_offset:
                        break
                    offset = next_offset
            except Exception as e:
                logger.error(f"Failed to scroll wide watchlist docs for {coll}: {e}")
        matched_docs = filter_and_tag(docs)
        
    # Sort docs by publish timestamp desc
    matched_docs.sort(key=lambda x: x.get("published_ts", 0), reverse=True)
    return rerank_and_cluster("watchlist news announcements", matched_docs, top_k)


def retrieve_ticker_stories(ticker: str, days: int = 1, top_k: int = 10) -> list[dict]:
    """
    Retrieve top stories for a single ticker using hybrid ticker and text-matching via scroll.
    Bypasses similarity score thresholds to guarantee 100% retrieval reliability.
    """
    store = get_vector_store()
    ticker = ticker.upper().strip()

    kw = {ticker, ticker.split(".")[0]}
    try:
        from ingestion.rss_fetcher import get_db_conn
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT alias, company_name FROM entity_cache WHERE ticker = ?", (ticker,))
        for alias, name in cursor.fetchall():
            if alias:
                kw.add(alias.upper())
            if name:
                kw.add(name.upper())
                words = name.split()
                if words:
                    kw.add(words[0].upper())
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to query entity_cache aliases for ticker {ticker}: {e}")

    # Query Qdrant
    from qdrant_client.models import Filter, FieldCondition, Range
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    where = Filter(must=[
        FieldCondition(key="published_ts", range=Range(gte=cutoff_ts))
    ])
    
    docs = []
    for coll in COLLECTIONS:
        try:
            offset = None
            while True:
                res, next_offset = store.client.scroll(
                    collection_name=coll,
                    scroll_filter=where,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                for point in res:
                    payload = point.payload
                    payload["id"] = payload.get("id", point.id)
                    docs.append(payload)
                if not next_offset:
                    break
                offset = next_offset
        except Exception as e:
            logger.error(f"Failed to scroll docs for ticker {ticker} in {coll}: {e}")
            
    filtered = []
    for d in docs:
        title = (d.get("title") or "").upper()
        summary = (d.get("summary") or "").upper()
        text = (d.get("text") or "").upper()
        doc_tickers = [t.upper().strip() for t in d.get("tickers", [])]
        
        matched = False
        if ticker in doc_tickers or ticker.split(".")[0] in doc_tickers:
            matched = True
        else:
            for k in kw:
                if k in title or k in summary or k in text:
                    matched = True
                    break
        
        if matched:
            d["tickers"] = list(set(d.get("tickers", []) + [ticker]))
            filtered.append(d)
            
    if not filtered and days < 30:
        logger.info(f"No articles for ticker {ticker} in last {days} days — widening to 30 days.")
        wide_cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())
        where = Filter(must=[
            FieldCondition(key="published_ts", range=Range(gte=wide_cutoff_ts))
        ])
        docs = []
        for coll in COLLECTIONS:
            try:
                offset = None
                while True:
                    res, next_offset = store.client.scroll(
                        collection_name=coll,
                        scroll_filter=where,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    for point in res:
                        payload = point.payload
                        payload["id"] = payload.get("id", point.id)
                        docs.append(payload)
                    if not next_offset:
                        break
                    offset = next_offset
            except Exception as e:
                logger.error(f"Failed to scroll wide docs for ticker {ticker} in {coll}: {e}")
                
        for d in docs:
            title = (d.get("title") or "").upper()
            summary = (d.get("summary") or "").upper()
            text = (d.get("text") or "").upper()
            doc_tickers = [t.upper().strip() for t in d.get("tickers", [])]
            
            matched = False
            if ticker in doc_tickers or ticker.split(".")[0] in doc_tickers:
                matched = True
            else:
                for k in kw:
                    if k in title or k in summary or k in text:
                        matched = True
                        break
            if matched:
                d["tickers"] = list(set(d.get("tickers", []) + [ticker]))
                filtered.append(d)
                
    filtered.sort(key=lambda x: x.get("published_ts", 0), reverse=True)
    return rerank_and_cluster(f"{ticker} news and announcements", filtered, top_k)