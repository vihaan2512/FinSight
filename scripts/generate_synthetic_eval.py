import os
import sys
import json
import random
import requests
from typing import List, Dict
from loguru import logger
from groq import Groq

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings

settings = get_settings()

GENERATOR_SYSTEM_PROMPT = """You are an expert financial QA dataset generator. Your task is to generate one high-quality, realistic evaluation query based on the provided financial news article.
The query must be directly relevant, factual, and answerable ONLY using the provided article.

Rules for fields:
1. "query": A realistic, natural question that a finance professional or retail investor would ask (e.g., "Why did Tata Motors shares fall 5%?"). It must be answerable using the article text.
2. "region": Always "india".
3. "relevant_keywords": 3-5 crucial SINGLE-WORD lowercase keywords from the question (e.g. ["tata", "motors", "fall"]). Do NOT put multi-word phrases.
4. "expected_tickers": ONLY list tickers that are explicitly mentioned in the article content or the 'Tickers' metadata list provided. Do NOT invent or guess tickers. Leave empty if none apply.
5. "ideal_answer_clues": 3-5 lowercase words, numbers, or phrases that are required for a correct answer (e.g., "semiconductor shortage", "revenue drop").
6. "bad_answer_clues": 1-2 lowercase words/phrases that indicate a hallucinated, out-of-context, or wrong answer.
7. "category": "stock", "index", "sector", "macro", "commodity", or "crypto".

Return ONLY a JSON object conforming to the schema, with no markdown formatting or wrapper:
{
  "query": "question string",
  "region": "india",
  "relevant_keywords": ["kw1", "kw2"],
  "expected_tickers": ["TICKER1"],
  "ideal_answer_clues": ["clue1", "clue2"],
  "bad_answer_clues": ["bad1"],
  "category": "stock"
}"""

def generate_synthetic_queries():
    if not settings.groq_api_key:
        logger.error("GROQ_API_KEY is not set. Cannot run synthetic generation.")
        return

    logger.info("Fetching articles from running FastAPI server...")
    articles = []
    
    # Query /feed endpoint with different topics to get diverse articles
    queries = ["latest market news", "corporate announcements", "rbi policy macro", "dividend earnings", "nifty sensex"]
    seen_titles = set()
    
    for q in queries:
        try:
            res = requests.post(
                "http://127.0.0.1:8000/feed",
                json={"query": q, "days": 90, "top_k": 30},
                timeout=15.0
            )
            if res.status_code == 200:
                data = res.json()
                for doc in data.get("articles", []):
                    title = doc.get("title")
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    
                    metadatas = {
                        "title": title,
                        "summary": doc.get("summary", ""),
                        "region": doc.get("region", "india"),
                        "tickers": doc.get("tickers", "") if isinstance(doc.get("tickers"), str) else ",".join(doc.get("tickers", []))
                    }
                    document = doc.get("summary", "")
                    articles.append((metadatas, document))
        except Exception as e:
            logger.warning(f"Error fetching articles for query '{q}' via API: {e}")

    total_docs = len(articles)
    if total_docs == 0:
        logger.error("No articles returned from the API. Make sure the FastAPI backend is running and has articles ingested.")
        return

    logger.info(f"Loaded {total_docs} unique articles via API.")

    # We need 50 queries. If we have less, we duplicate/cycle; if more, we sample.
    sampled_articles = []
    while len(sampled_articles) < 50:
        for art in articles:
            sampled_articles.append(art)
            if len(sampled_articles) == 50:
                break
        if total_docs == 0:
            break

    client = Groq(api_key=settings.groq_api_key)
    eval_dataset = []
    
    logger.info("Generating 50 synthetic queries using Groq (llama-3.3-70b-versatile)...")
    for i, (meta, doc) in enumerate(sampled_articles, 1):
        title = meta.get("title", "No Title")
        summary = meta.get("summary", "")
        region = meta.get("region", "india")
        tickers = meta.get("tickers", "")
        
        user_prompt = f"Title: {title}\nSummary: {summary}\nRegion: {region}\nTickers: {tickers}\nContent: {doc}"
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=256,
                temperature=0.4,
            )
            raw_content = response.choices[0].message.content.strip()
            # Strip markdown code blocks if any
            if raw_content.startswith("```"):
                raw_content = raw_content.split("```")[1]
                if raw_content.startswith("json"):
                    raw_content = raw_content[4:]
            raw_content = raw_content.strip()
            
            entry = json.loads(raw_content)
            entry["id"] = f"SYNTH_{i:02d}"
            eval_dataset.append(entry)
            logger.info(f"[{i}/50] Generated query: '{entry.get('query')}' for '{title[:30]}...'")
        except Exception as e:
            logger.warning(f"Failed to generate query for article {i}: {e}. Creating generic fallback...")
            # Simple fallback
            fallback_ticker = [t.strip() for t in tickers.split(",") if t.strip()] if tickers else []
            eval_dataset.append({
                "id": f"SYNTH_{i:02d}",
                "query": f"What is the latest news regarding {fallback_ticker[0]}?" if fallback_ticker else f"What are the updates on {title[:30]}?",
                "region": region,
                "relevant_keywords": [w.lower() for w in title.split()[:3] if len(w) > 2],
                "expected_tickers": fallback_ticker,
                "ideal_answer_clues": [w.lower() for w in title.split()[:2] if len(w) > 2],
                "bad_answer_clues": ["hallucination"],
                "category": "stock" if fallback_ticker else "macro"
            })

    # Output to ground_truth.py
    output_path = "evaluation/ground_truth.py"
    logger.info(f"Writing dataset to {output_path}...")
    
    import datetime
    generation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_content = f'''"""
evaluation/ground_truth.py
───────────────────────────
Synthetically generated evaluation dataset of 50 queries.
"""

EVAL_DATASET = {json.dumps(eval_dataset, indent=4)}
'''
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_content)
        
    logger.success("Synthetic dataset generated successfully!")

if __name__ == "__main__":
    generate_synthetic_queries()