"""
Metrics computed:
  RETRIEVAL
    - Precision@K      : fraction of retrieved docs that are relevant
    - Recall@K         : fraction of relevant docs that were retrieved
    - F1@K             : harmonic mean of precision and recall
    - MRR              : Mean Reciprocal Rank (where does first relevant doc appear?)
    - NDCG@K           : ranking quality (relevant docs ranked higher = better)
    - Hit Rate@K       : did ANY relevant doc appear in top-K?

  ANSWER QUALITY (LLM-as-judge via Groq)
    - Faithfulness     : is the answer grounded in retrieved context? (no hallucination)
    - Relevance        : does the answer actually address the question?
    - Completeness     : does it cover the key points from context?
    - Conciseness      : is it appropriately concise (not padded)?
"""


import math
import time
import re
from typing import Optional
from loguru import logger


# RELEVANCE SCORING

def score_doc_relevance(doc: dict, ground_truth: dict) -> float:
    """
    Score a single retrieved document's relevance to a ground truth query.
    Returns a float in [0, 1, 2] — 0=irrelevant, 1=partial, 2=highly relevant.

    Uses keyword matching against the ground truth relevant_keywords and tickers.
    No LLM call needed — fast and reproducible.
    """
    tickers_val = doc.get("tickers", "")
    tickers_str = ",".join(tickers_val) if isinstance(tickers_val, list) else str(tickers_val)
    text = (
        doc.get("title", "") + " " +
        doc.get("summary", "") + " " +
        tickers_str
    ).lower()

    keywords    = [kw.lower() for kw in ground_truth.get("relevant_keywords", [])]
    tickers     = [t.lower() for t in ground_truth.get("expected_tickers", [])]
    region_gt   = ground_truth.get("region")
    region_doc  = doc.get("region", "")

    # Count keyword hits with phrase/multi-term requirement
    keyword_hits = 0
    for kw in keywords:
        # If the keyword contains spaces, require the exact phrase (or boundary matching).
        # Otherwise, check word match.
        if " " in kw:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text):
                keyword_hits += 1
        else:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text):
                keyword_hits += 1

    ticker_hits = sum(1 for t in tickers if re.search(r'\b' + re.escape(t) + r'\b', text))

    # Region match bonus
    region_match = (region_gt is None) or (region_doc == region_gt)

    # Scoring logic
    if keyword_hits == 0 and ticker_hits == 0:
        return 0  # irrelevant
    if (keyword_hits >= len(keywords) or ticker_hits >= 1) and region_match:
        return 2  # highly relevant
    return 1      # partially relevant


def get_relevance_scores(docs: list[dict], ground_truth: dict) -> list[float]:
    return [score_doc_relevance(doc, ground_truth) for doc in docs]


# RETRIEVAL METRICS

def precision_at_k(relevance_scores: list[float], k: int, threshold: float = 1.0) -> float:
    """Fraction of top-K retrieved docs that are relevant."""
    if not relevance_scores or k == 0:
        return 0.0
    top_k = relevance_scores[:k]
    relevant = sum(1 for s in top_k if s >= threshold)
    return relevant / k


def recall_at_k(relevance_scores: list[float], total_relevant: int, k: int, threshold: float = 1.0) -> float:
    """Fraction of all relevant docs that appear in top-K."""
    if total_relevant == 0:
        return 1.0  # no relevant docs exist → perfect recall by convention
    top_k = relevance_scores[:k]
    retrieved_relevant = sum(1 for s in top_k if s >= threshold)
    return retrieved_relevant / total_relevant


def f1_at_k(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_rate_at_k(relevance_scores: list[float], k: int, threshold: float = 1.0) -> float:
    """Did at least one relevant doc appear in top-K? (binary)"""
    return float(any(s >= threshold for s in relevance_scores[:k]))


def mean_reciprocal_rank(relevance_scores: list[float], threshold: float = 1.0) -> float:
    """
    MRR: 1/rank of the first relevant document.
    Best possible = 1.0 (relevant doc is rank 1).
    Worst possible = 0.0 (no relevant doc found).
    """
    for rank, score in enumerate(relevance_scores, start=1):
        if score >= threshold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @K.
    Measures ranking quality — penalises relevant docs ranked lower.
    Score of 1.0 = perfect ranking.
    """
    if not relevance_scores:
        return 0.0

    top_k = relevance_scores[:k]

    # DCG: sum of (relevance / log2(rank+1))
    dcg = sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(top_k, start=1)
    )

    # Ideal DCG: best possible ranking (sorted descending)
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def count_actual_relevant_docs(ground_truth: dict, days: int = 30) -> int:
    """
    Calculate the actual number of relevant documents in the corpus within the time window.
    This scans the database to find how many documents are actually relevant to the query.
    """
    from datetime import datetime, timezone, timedelta
    from ingestion.embedder import get_vector_store, COLLECTIONS
    
    store = get_vector_store()
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    
    keywords = [kw.lower() for kw in ground_truth.get("relevant_keywords", [])]
    tickers = [t.lower() for t in ground_truth.get("expected_tickers", [])]
    region_gt = ground_truth.get("region")
    
    total_count = 0
    
    for coll in COLLECTIONS:
        try:
            from qdrant_client.models import Filter, FieldCondition, Range
            where = Filter(must=[FieldCondition(key="published_ts", range=Range(gte=cutoff_ts))])
            
            offset = None
            while True:
                res, next_offset = store.client.scroll(
                    collection_name=coll,
                    scroll_filter=where,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )
                for pt in res:
                    p = pt.payload
                    tickers_val = p.get("tickers", "")
                    tickers_str = ",".join(tickers_val) if isinstance(tickers_val, list) else str(tickers_val)
                    text = (
                        p.get("title", "") + " " +
                        p.get("summary", "") + " " +
                        tickers_str
                    ).lower()
                    
                    keyword_hits = 0
                    for kw in keywords:
                        pattern = r'\b' + re.escape(kw) + r'\b'
                        if re.search(pattern, text):
                            keyword_hits += 1
                    
                    ticker_hits = sum(1 for t in tickers if re.search(r'\b' + re.escape(t) + r'\b', text))
                    region_match = (region_gt is None) or (p.get("region") == region_gt)
                    
                    if (keyword_hits > 0 or ticker_hits > 0) and region_match:
                        total_count += 1
                        
                offset = next_offset
                if not offset:
                    break
        except Exception as e:
            logger.warning(f"Error counting relevant docs in {coll}: {e}")
            
    return max(1, total_count)


def compute_retrieval_metrics(
    docs: list[dict],
    ground_truth: dict,
    k_values: list[int] = [1, 3, 5, 8],
    days: int = 30,
) -> dict:
    """
    Compute all retrieval metrics for a single query.
    Returns a dict of metric_name → value.
    """
    relevance_scores = get_relevance_scores(docs, ground_truth)

    # Estimate total relevant docs in the corpus using the actual count in DB
    total_relevant = count_actual_relevant_docs(ground_truth, days)

    metrics = {
        "query_id":         ground_truth["id"],
        "query":            ground_truth["query"],
        "docs_retrieved":   len(docs),
        "relevance_scores": relevance_scores,
        "mrr":              mean_reciprocal_rank(relevance_scores),
    }

    for k in k_values:
        if k > len(docs):
            continue
        p = precision_at_k(relevance_scores, k)
        r = recall_at_k(relevance_scores, total_relevant, k)
        metrics[f"precision@{k}"] = round(p, 4)
        metrics[f"recall@{k}"]    = round(r, 4)
        metrics[f"f1@{k}"]        = round(f1_at_k(p, r), 4)
        metrics[f"hit_rate@{k}"]  = hit_rate_at_k(relevance_scores, k)
        metrics[f"ndcg@{k}"]      = round(ndcg_at_k(relevance_scores, k), 4)

    return metrics


# ANSWER QUALITY METRICS (LLM-as-Judge)

JUDGE_PROMPT = """You are an expert evaluator of AI-generated financial news answers.
You will evaluate an AI answer on 4 dimensions. Be strict and objective.

Score each dimension from 0.0 to 1.0:

FAITHFULNESS (0-1): Is every claim in the answer directly supported by the context?
  1.0 = every claim traceable to context, no hallucinations
  0.5 = mostly grounded, minor extrapolations
  0.0 = significant claims not in context / hallucinated facts

RELEVANCE (0-1): Does the answer directly address the question asked?
  - Utilize the provided IDEAL ANSWER CLUES and BAD ANSWER CLUES as guiding indicators.
  1.0 = directly and completely answers the question, aligns with ideal clues
  0.5 = partially answers, some tangents
  0.0 = answer is off-topic, ignores the question, or repeats bad clues

COMPLETENESS (0-1): Does the answer cover the key information from context?
  - Evaluate if the answer contains details/facts from the IDEAL ANSWER CLUES.
  1.0 = covers all important points from the context and ideal clues
  0.5 = covers some key points, misses others
  0.0 = misses most important information

CONCISENESS (0-1): Is the answer appropriately focused without padding?
  1.0 = tight, no fluff, every sentence adds value
  0.5 = some repetition or unnecessary content
  0.0 = very padded, repetitive, or excessively long

Respond ONLY with a JSON object in this exact format (no other text):
{
  "faithfulness": 0.0,
  "relevance": 0.0,
  "completeness": 0.0,
  "conciseness": 0.0,
  "reasoning": "one sentence explanation"
}"""


def evaluate_answer_quality(
    question: str,
    context: str,
    answer: str,
    groq_client,
    ground_truth: dict,
    model: str = "llama-3.1-8b-instant",  
) -> dict:
    """
    Use Groq LLM as a judge to score answer quality.
    Returns dict with faithfulness, relevance, completeness, conciseness scores.
    """
    import json

    ideal_clues = ", ".join(ground_truth.get("ideal_answer_clues", []))
    bad_clues = ", ".join(ground_truth.get("bad_answer_clues", []))

    user_msg = f"""QUESTION: {question}

RETRIEVED CONTEXT:
{context[:3000]}

AI ANSWER:
{answer[:2000]}

IDEAL ANSWER CLUES (should appear or be answered):
{ideal_clues}

BAD ANSWER CLUES (should NOT appear or be implied):
{bad_clues}

Evaluate the answer on the 4 dimensions."""

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        scores = json.loads(raw)

        return {
            "faithfulness":  float(scores.get("faithfulness", 0)),
            "relevance":     float(scores.get("relevance", 0)),
            "completeness":  float(scores.get("completeness", 0)),
            "conciseness":   float(scores.get("conciseness", 0)),
            "reasoning":     scores.get("reasoning", ""),
            "overall":       round(
                (scores.get("faithfulness", 0) * 0.35 +   
                 scores.get("relevance",     0) * 0.30 +
                 scores.get("completeness",  0) * 0.25 +
                 scores.get("conciseness",   0) * 0.10), 4
            ),
        }
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return {
            "faithfulness": None, "relevance": None,
            "completeness": None, "conciseness": None,
            "overall": None, "reasoning": f"Judge error: {e}",
        }


# CONTEXT QUALITY METRICS

def context_relevance_score(docs: list[dict], ground_truth: dict) -> float:
    """
    Average relevance of the retrieved context (0-1).
    Quick proxy for retrieval quality without LLM.
    """
    if not docs:
        return 0.0
    scores = get_relevance_scores(docs, ground_truth)
    # Normalise to [0,1] (max raw score is 2)
    return round(sum(scores) / (len(scores) * 2), 4)


def source_diversity_score(docs: list[dict]) -> float:
    """
    Fraction of retrieved docs that come from different sources.
    1.0 = all docs from different sources (ideal)
    0.0 = all docs from same source (poor diversity)
    """
    if not docs:
        return 0.0
    sources = [d.get("source", "") for d in docs]
    return round(len(set(sources)) / len(sources), 4)


def region_balance_score(docs: list[dict]) -> dict:
    """
    Calculate the fraction of retrieved documents that belong to each region.
    """
    if not docs:
        return {"india": 0.0, "global": 0.0}
    
    india_count = sum(1 for d in docs if d.get("region") == "india")
    total = len(docs)
    
    return {
        "india": round(india_count / total, 4),
        "global": round((total - india_count) / total, 4)
    }