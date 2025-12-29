"""
Hybrid BM25 + dense vector search with cross-encoder re-ranking.
"""

import math
from typing import Optional
from functools import lru_cache

from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from ingestion.embedder import get_vector_store, COLLECTION_NAME
from retrieval.query_expansion import expand_query_with_companies

from retrieval.cache_utils import ttl_cache

_cross_encoder: Optional[CrossEncoder] = None

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_POOL = 50   
RRF_K = 60            


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            import os
            torch.set_num_threads(min(4, os.cpu_count() or 4))
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512, device=device)
    return _cross_encoder


@lru_cache(maxsize=1024)
def rerank_with_cache(query: str, doc_ids: tuple[str, ...], doc_texts: tuple[str, ...]) -> tuple[float, ...]:
    ce = get_cross_encoder()
    pairs = [(query, text) for text in doc_texts]
    ce_scores = ce.predict(pairs)
    return tuple(float(s) for s in ce_scores)


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def _build_bm25(docs: list[dict]) -> BM25Okapi:
    """Build a BM25 index over a list of article dicts."""
    corpus = [
        _tokenize(f"{d.get('title','')} {d.get('summary','')}")
        for d in docs
    ]
    return BM25Okapi(corpus)



# ── Main hybrid retrieve ──────────────────────────────────────────────────────

@ttl_cache(ttl_seconds=300)
def hybrid_retrieve(
    query: str,
    ticker: Optional[str] = None,
    days: int = 1,
    top_k: int = 8,
    region: Optional[str] = None,
    use_reranker: bool = True,
) -> list[dict]:
    import time
    from datetime import datetime, timedelta, timezone
    
    store = get_vector_store()
    from retrieval.vector_store import route_query
    from ingestion.embedder import COLLECTIONS

    routed_colls = route_query(query)
    
    total = sum(store.client.get_collection(name).points_count for name in COLLECTIONS)
    if total == 0:
        logger.warning("Collections empty — run ingestion first")
        return []

    # ── 1. Dense retrieval candidates and distances ──────────────────────
    expanded_query = expand_query_with_companies(query)
    query_vec = store.embed_text(expanded_query)
    
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    conditions = [FieldCondition(key="published_ts", range=Range(gte=cutoff_ts))]
    if region:
        conditions.append(FieldCondition(key="region", match=MatchValue(value=region)))
    if ticker:
        ticker = ticker.upper().strip()
        base_ticker = ticker.split('.')[0] if '.' in ticker else ticker
        conditions.append(
            Filter(should=[
                FieldCondition(key="tickers", match=MatchValue(value=ticker)),
                FieldCondition(key="tickers", match=MatchValue(value=base_ticker))
            ])
        )
    where = Filter(must=conditions)
    
    def fetch_candidates(colls: list[str]) -> list[dict]:
        seen_ids = set()
        candidates = []
        for coll in colls:
            try:
                results = store.client.query_points(
                    collection_name=coll,
                    query=query_vec,
                    query_filter=where,
                    limit=min(CANDIDATE_POOL, total),
                    with_payload=True
                ).points
                for point in results:
                    payload = point.payload
                    doc_id = payload.get("id", point.id)
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    payload["id"] = doc_id
                    score = point.score
                    candidates.append((payload, 1.0 - score))
            except Exception as e:
                logger.error(f"Dense query error on {coll}: {e}")
        return candidates

    dense_candidates = fetch_candidates(routed_colls)

    if len(dense_candidates) < (CANDIDATE_POOL / 2) and set(routed_colls) != set(COLLECTIONS):
        logger.info(f"Routed query got only {len(dense_candidates)} candidates — trying all collections")
        dense_candidates = fetch_candidates(COLLECTIONS)

    if not dense_candidates and days < 90:
        logger.info("0 dense candidates — trying 90 days wide window")
        wide_cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=90)).timestamp()
        wide_conditions = [FieldCondition(key="published_ts", range=Range(gte=wide_cutoff_ts))]
        if region:
            wide_conditions.append(FieldCondition(key="region", match=MatchValue(value=region)))
        if ticker:
            ticker = ticker.upper().strip()
            base_ticker = ticker.split('.')[0] if '.' in ticker else ticker
            from qdrant_client.models import Filter as QFilter
            wide_conditions.append(
                QFilter(should=[
                    FieldCondition(key="tickers", match=MatchValue(value=ticker)),
                    FieldCondition(key="tickers", match=MatchValue(value=base_ticker))
                ])
            )
        where = Filter(must=wide_conditions)
        dense_candidates = fetch_candidates(routed_colls)
        if len(dense_candidates) < (CANDIDATE_POOL / 2) and set(routed_colls) != set(COLLECTIONS):
            dense_candidates = fetch_candidates(COLLECTIONS)

    if not dense_candidates:
        return []

    # ── 2. BM25 over the dense candidate pool ────────────────────────────
    docs_for_bm25 = [item[0] for item in dense_candidates]
    bm25 = _build_bm25(docs_for_bm25)
    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    max_bm25 = max(1e-5, max(bm25_scores)) if len(bm25_scores) > 0 else 1.0

    def compute_event_boost(q: str, event_type: str) -> float:
        """Apply a +0.15 boost if query keywords match the event_type."""
        if not event_type or event_type == "news":
            return 0.0
        if event_type.lower() in q.lower():
            return 0.15
        synonyms = {
            "earnings": ["results", "profit", "net profit", "revenue"],
            "dividend": ["payout", "yield"],
            "regulation": ["sebi", "rule", "guidelines", "compliance"],
            "macro": ["gdp", "inflation", "interest rate", "rbi", "fed"],
        }
        for syn in synonyms.get(event_type.lower(), []):
            if syn in q.lower():
                return 0.15
        return 0.0

    # ── 3. Calculate Custom Score for each candidate ──────────────────────
    current_time = time.time()
    scored_candidates = []
    
    for idx, (m, dist) in enumerate(dense_candidates):
        dense_similarity = 1.0 - dist
        
        bm25_score = bm25_scores[idx] / max_bm25
        
        source_weight = float(m.get("credibility_score", 0.7))
        
        pub_ts = float(m.get("published_ts", 0))
        age_hours = (current_time - pub_ts) / 3600.0 if pub_ts > 0 else 9999.0
        
        if age_hours <= 6:
            freshness_weight = 1.0
        elif age_hours <= 24:
            freshness_weight = 0.9
        elif age_hours <= 168: 
            freshness_weight = 0.7
        elif age_hours <= 720: 
            freshness_weight = 0.4
        else:
            freshness_weight = 0.2
            
        event_boost = compute_event_boost(query, m.get("event_type", ""))
        final_score = (
            0.45 * dense_similarity +
            0.25 * bm25_score +
            0.20 * source_weight +
            0.10 * freshness_weight +
            event_boost
        )
        scored_candidates.append((m, final_score))

    # ── 4. Cross-encoder re-ranking (if requested) ────────────────────────
    if use_reranker and len(scored_candidates) > 1:
        try:
            doc_ids = tuple(item[0].get("id", "") for item in scored_candidates)
            doc_texts = tuple(f"{item[0].get('title','')} {item[0].get('summary','')}" for item in scored_candidates)
            ce_scores = rerank_with_cache(query, doc_ids, doc_texts)
            
            max_ce = max(1e-5, max(ce_scores)) if len(ce_scores) > 0 else 1.0
            for idx, ce_val in enumerate(ce_scores):
                m, final_score = scored_candidates[idx]
                normalized_ce = ce_val / max_ce
                
                pub_ts = float(m.get("published_ts", 0))
                age_hours = (current_time - pub_ts) / 3600.0 if pub_ts > 0 else 9999.0
                if age_hours <= 6:
                    freshness_weight = 1.0
                elif age_hours <= 24:
                    freshness_weight = 0.9
                elif age_hours <= 168:
                    freshness_weight = 0.7
                elif age_hours <= 720:
                    freshness_weight = 0.4
                else:
                    freshness_weight = 0.2
                    
                event_boost = compute_event_boost(query, m.get("event_type", ""))
                new_final_score = (
                    0.45 * normalized_ce +
                    0.25 * (bm25_scores[idx] / max_bm25) +
                    0.20 * float(m.get("credibility_score", 0.7)) +
                    0.10 * freshness_weight +
                    event_boost
                )
                scored_candidates[idx] = (m, new_final_score)
                
            logger.info("Cross-encoder re-ranking complete with custom hybrid weights")
        except Exception as e:
            logger.warning(f"Cross-encoder failed, using standard hybrid formula: {e}")

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    sorted_docs = [item[0] for item in scored_candidates]

    # ── 5. Story Clustering ─────────────────────────────────────────────
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

    result = clustered_stories[:top_k]
    
    logger.info(
        f"Hybrid retrieve complete: candidate pool {len(dense_candidates)} "
        f"→ final top-{len(result)} clustered stories"
    )
    return result