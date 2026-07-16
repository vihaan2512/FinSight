import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import get_settings

settings = get_settings()

COLLECTIONS = ["finance_news", "finance_filings", "finance_macro"]
COLLECTION_NAME = "finance_news"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE       = 64


def get_collection_name(doc_type: str) -> str:
    """Route document type to the corresponding Qdrant collection name."""
    if doc_type in ("filing", "exchange_announcement"):
        return "finance_filings"
    elif doc_type in ("central_bank", "regulator", "macro"):
        return "finance_macro"
    else:
        return "finance_news"


def _to_ts(iso_str: str) -> int:
    """Convert ISO datetime string → Unix timestamp int (for filtering)."""
    from datetime import datetime, timezone
    from dateutil import parser as dp
    try:
        dt = dp.parse(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Segment text into overlapping chunks of character length chunk_size with overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    if text_len <= chunk_size:
        return [text]
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += chunk_size - overlap
    return chunks


def _to_uuid(id_str: str) -> str:
    """Convert any string ID into a valid UUID string format for Qdrant validation."""
    import hashlib
    import uuid
    try:
        return str(uuid.UUID(id_str))
    except ValueError:
        h = hashlib.md5(id_str.encode()).hexdigest()
        return str(uuid.UUID(hex=h))


class VectorStore:
    def __init__(self):
        if settings.qdrant_url and settings.qdrant_api_key:
            logger.info("Connecting to Qdrant Cloud cluster...")
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            logger.info(f"Connecting to local Qdrant database: {settings.qdrant_db_path}")
            self.client = QdrantClient(path=settings.qdrant_db_path)
        for name in COLLECTIONS:
            recreate = False
            if self.client.collection_exists(collection_name=name):
                try:
                    info = self.client.get_collection(collection_name=name)
                    current_size = info.config.params.vectors.size
                    if current_size != 384:
                        logger.warning(f"Collection {name} has size {current_size}, recreating with size 384")
                        self.client.delete_collection(collection_name=name)
                        recreate = True
                except Exception as e:
                    logger.warning(f"Error checking collection {name}: {e}. Recreating.")
                    try:
                        self.client.delete_collection(collection_name=name)
                    except Exception:
                        pass
                    recreate = True
            else:
                recreate = True

            if recreate:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            # Set thread count to physical cores (typically 4) to avoid core-thrashing on CPU
            torch.set_num_threads(min(4, os.cpu_count() or 4))
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        counts = {name: self.client.get_collection(name).points_count for name in COLLECTIONS}
        logger.success(f"Qdrant ready on {device.upper()} — documents: {counts}")

    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(
            text, normalize_embeddings=True
        ).tolist()

    def filter_new_articles(self, articles: list[dict]) -> list[dict]:
        """Check all collections in batch to filter out already existing articles."""
        if not articles:
            return []
        
        # Check both the original article UUID and the chunk_0 UUID
        uuids = []
        for a in articles:
            uuids.append(_to_uuid(a["id"]))
            uuids.append(_to_uuid(f"{a['id']}_chunk_0"))
            
        existing_ids = set()
        
        for name in COLLECTIONS:
            try:
                res = self.client.retrieve(
                    collection_name=name,
                    ids=uuids,
                    with_payload=False,
                    with_vectors=False
                )
                for point in res:
                    existing_ids.add(str(point.id))
            except Exception as e:
                logger.warning(f"Error checking batch duplicates in {name}: {e}")
                
        return [
            a for a in articles 
            if _to_uuid(a["id"]) not in existing_ids and _to_uuid(f"{a['id']}_chunk_0") not in existing_ids
        ]

    def embed_and_store(self, articles: list[dict]) -> int:
        new_articles = self.filter_new_articles(articles)
        if not new_articles:
            logger.info("No new articles to embed")
            return 0

        # Fetch full text concurrently using asyncio + aiohttp
        import asyncio
        import aiohttp
        import trafilatura

        async def fetch_full_text_async(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> str:
            async with semaphore:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                    async with session.get(url, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            html = await response.text()
                            extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
                            if extracted:
                                return extracted.strip()
                except Exception as e:
                    logger.warning(f"Error fetching full text for {url}: {e}")
                return ""

        async def fetch_all_texts_async(urls: list[str]) -> list[str]:
            sem = asyncio.Semaphore(15) # Concurrency limit of 15
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_full_text_async(session, url, sem) for url in urls]
                return await asyncio.gather(*tasks)

        logger.info(f"Extracting full text for {len(new_articles)} new articles concurrently via asyncio...")
        
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    full_texts = pool.submit(lambda: asyncio.run(fetch_all_texts_async([a["url"] for a in new_articles]))).result()
            else:
                full_texts = asyncio.run(fetch_all_texts_async([a["url"] for a in new_articles]))
        except Exception as e:
            logger.error(f"Async fetching failed: {e}. Falling back to empty strings.")
            full_texts = [""] * len(new_articles)

        for idx, full_txt in enumerate(full_texts):
            if full_txt:
                # Append full text, capped to 8000 characters to ensure efficient embedding length
                new_articles[idx]["text"] = f"{new_articles[idx]['title']}. {new_articles[idx]['summary']}. {full_txt[:8000]}"
            else:
                new_articles[idx]["text"] = f"{new_articles[idx]['title']}. {new_articles[idx]['summary']}"

        logger.info(f"Chunking and embedding {len(new_articles)} new articles (skipped {len(articles)-len(new_articles)} dupes)")
        
        chunk_items = []
        for a in new_articles:
            text_to_chunk = a.get("text", "")
            chunks = chunk_text(text_to_chunk, chunk_size=1200, overlap=200)
            if not chunks:
                chunks = [f"{a['title']}. {a['summary']}"]
            
            for chunk_idx, chunk_txt in enumerate(chunks):
                chunk_items.append({
                    "article": a,
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk_txt,
                    "id": f"{a['id']}_chunk_{chunk_idx}"
                })

        stored = 0

        for i in tqdm(range(0, len(chunk_items), BATCH_SIZE), desc="Embedding Chunks"):
            batch = chunk_items[i: i + BATCH_SIZE]
            texts = [
                f"Tickers: {','.join(item['article'].get('tickers',[]))}. {item['chunk_text']}"
                for item in batch
            ]
            embeddings = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

            # Group index routes and query Qdrant in batch to find semantic duplicates
            from collections import defaultdict
            from qdrant_client.models import QueryRequest

            coll_to_indices = defaultdict(list)
            for j, item in enumerate(batch):
                dest_coll = get_collection_name(item["article"].get("document_type", "news"))
                coll_to_indices[dest_coll].append(j)

            canonical_story_ids = [""] * len(batch)

            for dest_coll, indices in coll_to_indices.items():
                requests = [
                    QueryRequest(query=embeddings[idx], limit=1, with_payload=True)
                    for idx in indices
                ]
                try:
                    batch_results = self.client.query_batch_points(
                        collection_name=dest_coll,
                        requests=requests
                    )
                    for k, match_result in enumerate(batch_results):
                        idx = indices[k]
                        story_id = f"story_{batch[idx]['article']['id']}"
                        if match_result.points:
                            matched_point = match_result.points[0]
                            if matched_point.score >= 0.90:
                                story_id = matched_point.payload.get("canonical_story_id", f"story_{matched_point.id}")
                        canonical_story_ids[idx] = story_id
                except Exception as e:
                    logger.warning(f"Batch story query failed for {dest_coll}: {e}")
                    for idx in indices:
                        canonical_story_ids[idx] = f"story_{batch[idx]['article']['id']}"

            from collections import defaultdict
            points_by_coll = defaultdict(list)
            for idx, item in enumerate(batch):
                a = item["article"]
                payload = {
                    "id":                 a["id"],
                    "chunk_id":           item["id"],
                    "chunk_idx":          item["chunk_idx"],
                    "title":              a["title"],
                    "summary":            a["summary"][:500],
                    "published_ts":       _to_ts(a["published"]),
                    "published":          a["published"][:10],
                    "source":             a["source"],
                    "region":             a.get("region", "india"),
                    "url":                a["url"],
                    "tickers":            a.get("tickers", []),
                    "source_tier":        a.get("source_tier", 2),
                    "credibility_score":  a.get("credibility_score", 0.7),
                    "document_type":      a.get("document_type", "news"),
                    "author":             a.get("author", "Unknown"),
                    "canonical_story_id": canonical_story_ids[idx],
                    "text":               item["chunk_text"],
                    "full_text":          a["text"],
                }
                dest_coll = get_collection_name(payload["document_type"])
                points_by_coll[dest_coll].append(
                    PointStruct(
                        id=_to_uuid(item["id"]),
                        vector=embeddings[idx],
                        payload=payload
                    )
                )

            for dest_coll, pts in points_by_coll.items():
                self.client.upsert(
                    collection_name=dest_coll,
                    points=pts
                )
            stored += len(batch)

        total_count = sum(self.client.get_collection(name).points_count for name in COLLECTIONS)
        logger.success(f"Stored {stored} chunks. Total across all collections: {total_count}")
        return stored

    def get_collection_stats(self) -> dict:
        counts = {name: self.client.get_collection(name).points_count for name in COLLECTIONS}
        return {
            "collections":     COLLECTIONS,
            "counts":          counts,
            "total_documents": sum(counts.values()),
            "vector_size":     1024,
            "model":           EMBEDDING_MODEL,
            "db_path":         settings.qdrant_db_path,
        }


_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


if __name__ == "__main__":
    from ingestion.rss_fetcher import fetch_all_articles
    store = get_vector_store()
    articles = fetch_all_articles()
    stored = store.embed_and_store(articles)
    print(f"\nStored {stored} articles")
    print(f"Stats: {store.get_collection_stats()}")