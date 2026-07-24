"""
Runs the full evaluation suite over the ground truth dataset.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from loguru import logger
from groq import Groq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings
settings = get_settings()
settings.qdrant_db_path = "./qdrant_db_benchmark"

from retrieval.vector_store import retrieve, format_context
from retrieval.hybrid_search import hybrid_retrieve
from api.llm import ask_groq
from evaluation.ground_truth import EVAL_DATASET
from evaluation.metrics import (
    compute_retrieval_metrics,
    evaluate_answer_quality,
    context_relevance_score,
    source_diversity_score,
    region_balance_score,
)

settings = get_settings()
RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# SINGLE QUERY EVALUATION

def evaluate_single(
    ground_truth: dict,
    groq_client: Groq,
    quick: bool = False,
    days: int = 30,
    top_k: int = 8,
    mode: str = "dense",
) -> dict:
    """Run full evaluation for one query. Returns result dict."""

    query  = ground_truth["query"]
    region = ground_truth.get("region")
    tickers_list = ground_truth.get("expected_tickers", [])
    ticker = tickers_list[0] if tickers_list else None

    logger.info(f"Evaluating [{ground_truth['id']}]: {query[:60]} in {mode} mode")

    # ── Retrieval (with timing) ───────────────────────────────────────────
    t0 = time.perf_counter()
    if mode == "hybrid":
        docs = hybrid_retrieve(query, ticker=ticker, days=days, top_k=top_k, region=region)
    else:
        docs = retrieve(query, ticker=ticker, days=days, top_k=top_k, region=region)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── Retrieval Metrics ─────────────────────────────────────────────────
    retrieval_metrics = compute_retrieval_metrics(docs, ground_truth, days=days)
    ctx_relevance     = context_relevance_score(docs, ground_truth)
    src_diversity     = source_diversity_score(docs)
    region_balance    = region_balance_score(docs)

    result = {
        "query_id":          ground_truth["id"],
        "query":             query,
        "category":          ground_truth.get("category", "unknown"),
        "region":            region,
        "docs_retrieved":    len(docs),
        "retrieval_ms":      retrieval_ms,
        "retrieval_metrics": retrieval_metrics,
        "context_relevance": ctx_relevance,
        "source_diversity":  src_diversity,
        "region_balance":    region_balance,
    }

    if quick or not settings.groq_api_key or len(docs) == 0:
        result["answer_quality"] = None
        result["generation_ms"]  = None
        return result

    # ── Generation + Answer Quality ───────────────────────────────────────
    context = format_context(docs)

    t1 = time.perf_counter()
    answer = ask_groq(query, context, ticker, stream=False)
    generation_ms = round((time.perf_counter() - t1) * 1000, 1)

    answer_quality = evaluate_answer_quality(
        question=query,
        context=context,
        answer=answer,
        groq_client=groq_client,
        ground_truth=ground_truth,
    )

    result["answer"]         = answer
    result["generation_ms"]  = generation_ms
    result["answer_quality"] = answer_quality

    return result


# AGGREGATE METRICS

def aggregate_results(results: list[dict]) -> dict:
    """Compute mean metrics across all queries."""

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    retrieval_keys = ["precision@3", "precision@5", "recall@3", "recall@5",
                      "f1@5", "hit_rate@5", "ndcg@5", "mrr"]

    agg = {"n_queries": len(results), "retrieval": {}, "answer_quality": {}, "latency": {}}

    # Retrieval aggregates
    for key in retrieval_keys:
        vals = [r["retrieval_metrics"].get(key) for r in results]
        agg["retrieval"][key] = mean(vals)

    agg["retrieval"]["context_relevance"] = mean([r["context_relevance"] for r in results])
    agg["retrieval"]["source_diversity"]  = mean([r["source_diversity"] for r in results])

    # Answer quality aggregates
    aq_results = [r["answer_quality"] for r in results if r.get("answer_quality")]
    if aq_results:
        for dim in ["faithfulness", "relevance", "completeness", "conciseness", "overall"]:
            agg["answer_quality"][dim] = mean([r.get(dim) for r in aq_results])

    # Latency aggregates
    agg["latency"]["retrieval_ms_mean"] = mean([r.get("retrieval_ms") for r in results])
    agg["latency"]["generation_ms_mean"] = mean(
        [r.get("generation_ms") for r in results if r.get("generation_ms")]
    )

    # Per-category breakdown
    categories = set(r["category"] for r in results)
    agg["by_category"] = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        agg["by_category"][cat] = {
            "n": len(cat_results),
            "precision@5": mean([r["retrieval_metrics"].get("precision@5") for r in cat_results]),
            "hit_rate@5":  mean([r["retrieval_metrics"].get("hit_rate@5") for r in cat_results]),
        }

    # Per-region breakdown
    for reg in ["india"]:
        reg_results = [r for r in results if r.get("region") == reg]
        if reg_results:
            agg[f"region_{reg}"] = {
                "n": len(reg_results),
                "precision@5": mean([r["retrieval_metrics"].get("precision@5") for r in reg_results]),
                "hit_rate@5":  mean([r["retrieval_metrics"].get("hit_rate@5") for r in reg_results]),
            }

    return agg


# REPORT PRINTER

def print_report(agg: dict, results: list[dict]):
    """Print a clean evaluation report to stdout."""
    # Force stdout/stderr encoding if they don't support unicode emojis correctly in Windows
    import sys
    try:
        if sys.stdout.encoding != 'utf-8':
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except Exception:
        pass

    sep = "=" * 65

    print(f"\n{sep}")
    print("  [REPORT] FINSIGHT EVALUATION REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {agg['n_queries']} queries")
    print(sep)

    # Retrieval
    r = agg["retrieval"]
    print("\n[METRICS] RETRIEVAL METRICS")
    print(f"  {'Metric':<25} {'Score':>8}  {'Rating'}")
    print(f"  {'-'*50}")

    def rating(v):
        if v is None: return "N/A"
        if v >= 0.8:  return "🟢 Excellent"
        if v >= 0.6:  return "🟡 Good"
        if v >= 0.4:  return "🟠 Fair"
        return               "🔴 Needs Work"

    metrics_display = [
        ("Precision@3",        r.get("precision@3")),
        ("Precision@5",        r.get("precision@5")),
        ("Recall@3",           r.get("recall@3")),
        ("Recall@5",           r.get("recall@5")),
        ("F1@5",               r.get("f1@5")),
        ("Hit Rate@5",         r.get("hit_rate@5")),
        ("NDCG@5",             r.get("ndcg@5")),
        ("MRR",                r.get("mrr")),
        ("Context Relevance",  r.get("context_relevance")),
        ("Source Diversity",   r.get("source_diversity")),
    ]
    for name, val in metrics_display:
        val_str = f"{val:.4f}" if val is not None else "  N/A "
        print(f"  {name:<25} {val_str:>8}  {rating(val)}")

    # Answer Quality
    aq = agg.get("answer_quality", {})
    if aq:
        print("\n🤖 ANSWER QUALITY (LLM-as-Judge)")
        print(f"  {'Dimension':<25} {'Score':>8}  {'Rating'}")
        print(f"  {'-'*50}")
        for dim in ["faithfulness", "relevance", "completeness", "conciseness", "overall"]:
            val = aq.get(dim)
            val_str = f"{val:.4f}" if val is not None else "  N/A "
            label = dim.capitalize()
            if dim == "overall":
                label = "⭐ OVERALL"
            print(f"  {label:<25} {val_str:>8}  {rating(val)}")

    # Latency
    lat = agg.get("latency", {})
    print("\n⚡ LATENCY")
    print(f"  Retrieval:   {lat.get('retrieval_ms_mean', 'N/A')} ms (mean)")
    print(f"  Generation:  {lat.get('generation_ms_mean', 'N/A')} ms (mean)")

    # By Category
    if agg.get("by_category"):
        print("\n📂 BY CATEGORY (Precision@5 | Hit Rate@5)")
        for cat, vals in agg["by_category"].items():
            p = f"{vals['precision@5']:.2f}" if vals.get('precision@5') is not None else "N/A"
            h = f"{vals['hit_rate@5']:.2f}"  if vals.get('hit_rate@5')  is not None else "N/A"
            print(f"  {cat:<18}  P@5={p}  HR@5={h}  (n={vals['n']})")

    # By Region
    print("\n🌍 BY REGION (Precision@5 | Hit Rate@5)")
    for reg in ["india"]:
        vals = agg.get(f"region_{reg}", {})
        if vals:
            p = f"{vals['precision@5']:.2f}" if vals.get('precision@5') is not None else "N/A"
            h = f"{vals['hit_rate@5']:.2f}"  if vals.get('hit_rate@5')  is not None else "N/A"
            flag = "🇮🇳" if reg == "india" else "🌍"
            print(f"  {flag} {reg:<18}  P@5={p}  HR@5={h}  (n={vals['n']})")

    # Per-query summary
    print("\n📋 PER-QUERY RESULTS")
    print(f"  {'ID':<10} {'P@5':>6} {'HR@5':>6} {'MRR':>6} {'Faith':>6} {'Rel':>6}  Query")
    print(f"  {'-'*75}")
    for r in results:
        rm  = r["retrieval_metrics"]
        aq  = r.get("answer_quality") or {}
        p5  = f"{rm.get('precision@5', 0):.2f}"
        hr  = f"{rm.get('hit_rate@5',  0):.2f}"
        mrr = f"{rm.get('mrr',         0):.2f}"
        fth = f"{aq.get('faithfulness', 0):.2f}" if aq.get('faithfulness') is not None else " N/A"
        rel = f"{aq.get('relevance',    0):.2f}" if aq.get('relevance')    is not None else " N/A"
        q   = r["query"][:35]
        print(f"  {r['query_id']:<10} {p5:>6} {hr:>6} {mrr:>6} {fth:>6} {rel:>6}  {q}")

    # Resume bullet
    overall_aq = agg.get("answer_quality", {}).get("overall")
    p5_mean    = agg["retrieval"].get("precision@5") or 0.0
    hr5_mean   = agg["retrieval"].get("hit_rate@5") or 0.0
    mrr_mean   = agg["retrieval"].get("mrr") or 0.0

    print(f"\n{'─'*65}")
    print("  📝 RESUME BULLET (copy this):")
    print(f"  {'─'*63}")
    bullet = (
        f"  Built real-time RAG pipeline over {agg['n_queries']} eval queries achieving\n"
        f"  Precision@5={p5_mean:.0%}, Hit Rate={hr5_mean:.0%}, MRR={mrr_mean:.2f}"
    )
    if overall_aq:
        bullet += f", Answer Quality={overall_aq:.0%}"
    bullet += "\n  across trusted Indian financial news sources."
    print(bullet)
    print(sep + "\n")


# MAIN RUNNER

def run_evaluation(
    quick: bool = False,
    query_id: str = None,
    days: int = 30,
    top_k: int = 8,
    save: bool = True,
    mode: str = "hybrid",
):
    groq_client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None

    dataset = EVAL_DATASET
    if query_id:
        dataset = [q for q in EVAL_DATASET if q["id"] == query_id]
        if not dataset:
            logger.error(f"Query ID '{query_id}' not found")
            return

    logger.info(f"Running eval on {len(dataset)} queries | quick={quick} | days={days} | mode={mode}")

    results = []
    for gt in dataset:
        result = evaluate_single(gt, groq_client, quick=quick, days=days, top_k=top_k, mode=mode)
        results.append(result)

    agg = aggregate_results(results)
    print_report(agg, results)

    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{RESULTS_DIR}/eval_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"summary": agg, "results": results}, f, indent=2, default=str)
        logger.success(f"Results saved to {out_path}")

    return {"metrics": agg, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight Evaluation Runner")
    parser.add_argument("--quick",  action="store_true", help="Skip LLM judge (retrieval metrics only)")
    parser.add_argument("--query",  type=str, default=None, help="Run single query by ID e.g. IN_01")
    parser.add_argument("--days",   type=int, default=30,   help="Days of news to search (default 30)")
    parser.add_argument("--top-k",  type=int, default=8,    help="Docs to retrieve per query")
    parser.add_argument("--no-save",action="store_true",    help="Don't save results to file")
    parser.add_argument("--mode",   type=str, default="hybrid", choices=["dense", "hybrid"], help="Retrieval mode (dense or hybrid)")
    args = parser.parse_args()

    run_evaluation(
        quick=args.quick,
        query_id=args.query,
        days=args.days,
        top_k=args.top_k,
        save=not args.no_save,
        mode=args.mode,
    )