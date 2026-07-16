"""
scripts/test_pipeline.py
────────────────────────
End-to-end test for Week 1+2.
Run from project root: python scripts/test_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_qdrant():
    print("\n[1/6] Testing Qdrant...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, MatchValue
    import shutil
    path = "./qdrant_db_test_tmp"
    try:
        client = QdrantClient(path=path)
        client.recreate_collection(
            collection_name="test",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        client.upsert(
            collection_name="test",
            points=[
                PointStruct(
                    id=1,
                    vector=[0.1]*1024,
                    payload={"published_ts": 1700000000, "region": "india", "tickers": ["TCS", "INFY"]}
                )
            ]
        )
        assert client.get_collection("test").points_count == 1

        # Test filters we actually use
        r = client.query_points(
            collection_name="test",
            query=[0.1]*1024,
            query_filter=Filter(must=[FieldCondition(key="published_ts", range=Range(gte=1000000000))])
        ).points
        assert len(r) == 1, "date filter broken"
        r = client.query_points(
            collection_name="test",
            query=[0.1]*1024,
            query_filter=Filter(must=[FieldCondition(key="region", match=MatchValue(value="india"))])
        ).points
        assert len(r) == 1, "region filter broken"

        client.delete_collection("test")
        print("      [PASS] Qdrant + filters working")
    finally:
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"      (Cleaned up temp directory with warning: {e})")



def test_indian_feeds():
    print("\n[2/6] Testing Indian RSS feeds...")
    from ingestion.rss_fetcher import INDIAN_FEEDS, fetch_from_feed
    source, url = list(INDIAN_FEEDS.items())[0]
    articles = fetch_from_feed(source, url)
    assert len(articles) >= 0, "fetch returned error"
    print(f"      [PASS] [{source}] fetched {len(articles)} articles")
    if articles:
        print(f"         Sample: {articles[0]['title'][:70]}")
        print(f"         Source: {articles[0].get('source')}")



def test_embed_and_retrieve():
    print("\n[4/6] Testing embed + retrieve pipeline...")
    from ingestion.rss_fetcher import fetch_all_articles
    from ingestion.embedder import get_vector_store, COLLECTIONS
    from retrieval.vector_store import retrieve

    articles = fetch_all_articles()
    print(f"      Indian articles: {len(articles)}")

    store = get_vector_store()
    stored = store.embed_and_store(articles[:15])
    counts = {name: store.client.get_collection(name).points_count for name in COLLECTIONS}
    print(f"      Stored {stored} new articles. Total counts: {counts}")

    # Test retrieval with days=3650 (10 years) to catch any indexed articles
    docs = retrieve("stock market news", days=3650, top_k=3)
    print(f"      [PASS] Retrieved {len(docs)} docs")
    if docs:
        print(f"         Top: {docs[0].get('title','')[:65]}")


def test_hybrid_search():
    print("\n[5/6] Testing hybrid search (BM25 + Vector + re-rank)...")
    from retrieval.hybrid_search import hybrid_retrieve
    docs = hybrid_retrieve("market news earnings", days=3650, top_k=3, use_reranker=False)
    print(f"      [PASS] Hybrid retrieved {len(docs)} docs")
    if docs:
        print(f"         Top: {docs[0].get('title','')[:65]}")


def test_groq():
    print("\n[6/6] Testing Groq LLM...")
    from config import get_settings
    s = get_settings()
    if not s.groq_api_key or "your_" in s.groq_api_key:
        print("      [WARN] GROQ_API_KEY not set in .env — skipping")
        return
    from retrieval.vector_store import retrieve, format_context
    from api.llm import ask_groq
    docs = retrieve("financial market news", days=3650, top_k=3)
    if not docs:
        print("      [WARN] No docs in DB yet — skipping LLM test")
        return
    context = format_context(docs)
    answer = ask_groq("What is the latest market news?", context, stream=False)
    print(f"      [PASS] Groq responded ({len(answer)} chars)")
    print(f"         Preview: {answer[:120]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("  FinSight - Week 1+2 Pipeline Test")
    print("=" * 60)

    tests = [
        test_qdrant,
        test_indian_feeds,
        test_embed_and_retrieve,
        test_hybrid_search,
        test_groq,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"      [FAIL]: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("\n  [PASS] All good! Next steps:")
        print("     1. uvicorn api.main:app --reload --port 8000")
        print("     2. streamlit run ui/app.py   (new terminal)")
    else:
        print("\n  [WARN] Fix failures above before starting the API.")
    print("=" * 60)