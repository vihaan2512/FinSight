"""
Per-ticker sentiment scoring using FinBERT + LLM explanation.

For each ticker mentioned in retrieved articles, we score sentiment
using the FinBERT model, then ask the LLM to explain the score.
"""

import json
import re
from groq import Groq
from loguru import logger
from config import get_settings

settings = get_settings()

_tokenizer = None
_model = None


def get_finbert():
    """Lazily import and load the FinBERT model to prevent startup overhead."""
    global _tokenizer, _model
    if _model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = "ProsusAI/finbert"
        logger.info(f"Loading FinBERT model: {model_name}")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
        _model = _model.to("cpu")
    return _tokenizer, _model


def predict_sentiment_batch(texts: list[str]) -> list[float]:
    """Calculate raw sentiment scores in range [-1.0, 1.0] for a list of texts using FinBERT."""
    if not texts:
        return []
    try:
        import torch
        tokenizer, model = get_finbert()
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # ProsusAI/finbert id2label: {0: "positive", 1: "negative", 2: "neutral"}
        # score = positive probability - negative probability
        scores = []
        for i in range(len(texts)):
            pos = probs[i][0].item()
            neg = probs[i][1].item()
            scores.append(pos - neg)
        return scores
    except Exception as e:
        logger.error(f"FinBERT batch prediction error: {e}")
        return [0.0] * len(texts)


def get_clean_source_name(source_id: str) -> str:
    """Map raw feed source identifiers to clean publisher names."""
    src = source_id.lower()
    if "reuters" in src:
        return "Reuters"
    if "cnbc" in src:
        return "CNBC"
    if "bloomberg" in src:
        return "Bloomberg"
    if "marketwatch" in src:
        return "MarketWatch"
    if "moneycontrol" in src:
        return "Moneycontrol"
    if "economic_times" in src or "economictimes" in src:
        return "Economic Times"
    if "business_standard" in src or "businessstandard" in src:
        return "Business Standard"
    if "livemint" in src:
        return "Livemint"
    if "bbc" in src:
        return "BBC News"
    if "ap_business" in src or "apnews" in src:
        return "Associated Press"
    if "hindu_businessline" in src or "thehindubusinessline" in src:
        return "The Hindu BusinessLine"
    if "zeebusiness" in src or "zeebiz" in src:
        return "Zee Business"
    if "financial_express" in src or "financialexpress" in src:
        return "Financial Express"
    if "sebi" in src:
        return "SEBI"
    if "rbi" in src:
        return "RBI"
    if "sec_company_filings" in src or "sec.gov" in src:
        return "SEC Filings"
    if "nse_" in src:
        return "NSE"
    if "bse_" in src:
        return "BSE"
    return source_id.replace("_", " ").title()


def score_sentiment(ticker: str, docs: list[dict]) -> dict:
    """
    Score sentiment for a ticker based on retrieved articles using FinBERT,
    then use Groq to generate a natural language explanation of the score.
    Groups articles by publisher source to show source consensus.
    """
    if not docs:
        return {
            "score": 0.0,
            "label": "Neutral 🟡",
            "reason": "No data",
            "key_factors": [],
            "source_sentiments": {},
            "consensus": "No Data"
        }

    texts = []
    for d in docs[:8]:
        title = d.get("title", "")
        summary = d.get("summary", "")
        texts.append(f"{title}. {summary}")

    scores = predict_sentiment_batch(texts)
    if scores:
        total_weight = 0.0
        weighted_sum = 0.0
        for doc, score in zip(docs[:8], scores):
            weight = doc.get("credibility_score", 0.7)
            weighted_sum += score * weight
            total_weight += weight
        avg_score = weighted_sum / total_weight if total_weight > 0 else sum(scores) / len(scores)
    else:
        avg_score = 0.0

    avg_score = max(-1.0, min(1.0, avg_score))  # clamp to [-1.0, 1.0]

    if avg_score >= 0.15:
        label = "Bullish 🟢"
    elif avg_score <= -0.15:
        label = "Bearish 🔴"
    else:
        label = "Neutral 🟡"

    source_docs = {}
    for d in docs:
        src = get_clean_source_name(d.get("source", "unknown"))
        if src not in source_docs:
            source_docs[src] = []
        source_docs[src].append(d)

    source_sentiments = {}
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    for src, s_docs in source_docs.items():
        s_texts = [f"{d.get('title','')}. {d.get('summary','')}" for d in s_docs[:5]]
        s_scores = predict_sentiment_batch(s_texts)
        s_avg = sum(s_scores) / len(s_scores) if s_scores else 0.0
        
        if s_avg >= 0.15:
            s_label = "Bullish"
            bullish_count += 1
        elif s_avg <= -0.15:
            s_label = "Bearish"
            bearish_count += 1
        else:
            s_label = "Neutral"
            neutral_count += 1
            
        source_sentiments[src] = {
            "score": round(s_avg, 3),
            "label": s_label
        }

    total_sources = len(source_sentiments)
    consensus_str = "Neutral 🟡"
    if total_sources > 0:
        counts = {"Bullish": bullish_count, "Bearish": bearish_count, "Neutral": neutral_count}
        dominant = max(counts, key=counts.get)
        dominant_count = counts[dominant]
        consensus_percent = int((dominant_count / total_sources) * 100)
        
        emoji = "🟢" if dominant == "Bullish" else "🔴" if dominant == "Bearish" else "🟡"
        consensus_str = f"{consensus_percent}% {dominant} {emoji}"

    if not settings.groq_api_key:
        return {
            "score":             round(avg_score, 3),
            "label":             label,
            "reason":            f"FinBERT calculated sentiment score of {avg_score:.2f} based on {len(texts)} articles.",
            "key_factors":       [],
            "source_sentiments": source_sentiments,
            "consensus":         consensus_str
        }

    client = Groq(api_key=settings.groq_api_key)
    digest = "\n".join([
        f"- [{d.get('source','?')} | {d.get('published','')[:10]}] {d.get('title','')}"
        for d in docs[:8]
    ])

    prompt = f"""You are a financial analyst explaining a calculated sentiment score.
The sentiment analysis system calculated a score of {avg_score:.2f} ({label}) for the ticker/company '{ticker}' based on recent news.

Here is the news digest for '{ticker}':
{digest}

Write a concise one-sentence explanation of why this sentiment score is justified, and list 2-3 key driving factors.
Your explanation MUST align with the calculated sentiment of {avg_score:.2f} ({label}).

Respond ONLY with a JSON object, nothing else:
{{
  "reason": "a one-sentence explanation of the sentiment",
  "key_factors": ["factor 1", "factor 2", "factor 3"]
}}"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",   
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)

        return {
            "score":             round(avg_score, 3),
            "label":             label,
            "reason":            result.get("reason", ""),
            "key_factors":       result.get("key_factors", []),
            "source_sentiments": source_sentiments,
            "consensus":         consensus_str
        }

    except Exception as e:
        logger.warning(f"Groq explanation failed for {ticker}: {e}")
        return {
            "score":             round(avg_score, 3),
            "label":             label,
            "reason":            f"FinBERT calculated sentiment score of {avg_score:.2f} based on recent articles.",
            "key_factors":       [],
            "source_sentiments": source_sentiments,
            "consensus":         consensus_str
        }


def score_multiple_tickers(docs: list[dict]) -> dict[str, dict]:
    """
    Score sentiment for all tickers appearing in a set of docs.
    Returns {ticker: sentiment_dict} for top 5 tickers by mention count.
    """
    ticker_counts: dict[str, int] = {}
    for doc in docs:
        tickers_data = doc.get("tickers", [])
        if isinstance(tickers_data, list):
            tickers_list = tickers_data
        elif isinstance(tickers_data, str):
            tickers_list = [t.strip() for t in tickers_data.split(",") if t.strip()]
        else:
            tickers_list = []
            
        for t in tickers_list:
            t = t.strip()
            if t:
                ticker_counts[t] = ticker_counts.get(t, 0) + 1

    top_tickers = sorted(ticker_counts, key=ticker_counts.get, reverse=True)[:5]

    results = {}
    for ticker in top_tickers:
        ticker = ticker.upper().strip()
        ticker_docs = []
        for d in docs:
            t_data = d.get("tickers", [])
            if isinstance(t_data, list):
                t_list = [x.upper().strip() for x in t_data]
            elif isinstance(t_data, str):
                t_list = [x.upper().strip() for x in t_data.split(",") if x.strip()]
            else:
                t_list = []
            if ticker in t_list:
                ticker_docs.append(d)
                
        results[ticker] = score_sentiment(ticker, ticker_docs)

    return results