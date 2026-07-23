"""
Groq LLM integration.
"""

from groq import Groq
from loguru import logger
from config import get_settings

def get_groq_client() -> Groq:
    settings = get_settings()
    return Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """You are an expert financial analyst assistant.
You answer questions about finance, markets, and stocks using ONLY the news context provided by deeply analysing the retrieved articles.
You do NOT use any outside knowledge or assumptions.

Your response format:
1. **Direct Answer** — Answer the question clearly and concisely
2. **Sentiment** — Overall market/stock sentiment: 🟢 Bullish | 🔴 Bearish | 🟡 Neutral
3. **Key Points** — 3-5 bullet points of the most important facts
4. **Risks & Catalysts** — Brief mention of risks or upcoming catalysts if relevant
5. **Sources** — List the article titles you used

Rules:
- Only use information from the provided context
- If context is insufficient, say so clearly
- Be precise with numbers, dates, and company names
- Never fabricate or hallucinate financial data
"""


def ask_groq(
    question: str,
    context: str,
    ticker: str = None,
    model: str = None,
    stream: bool = False,
):
    """
    Send a finance question + retrieved context to Groq.

    Args:
        question: User's question
        context:  Retrieved news articles (formatted string)
        ticker:   Optional ticker being asked about
        model:    Groq model ID (defaults to settings.groq_model)
        stream:   If True, returns a streaming generator

    Returns:
        str (if stream=False) or generator (if stream=True)
    """
    model = model or settings.groq_model
    ticker_hint = f" Focus on {ticker}." if ticker else ""

    user_message = f"""NEWS CONTEXT:
{context}

QUESTION: {question}{ticker_hint}

Answer based strictly on the news context above."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    logger.info(f"Calling Groq [{model}] | question: '{question[:60]}'")

    if stream:
        return _stream_response(messages, model)
    else:
        return _sync_response(messages, model)


def _sync_response(messages: list, model: str) -> str:
    """Non-streaming Groq call. Returns full answer string."""
    try:
        completion = get_groq_client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,   
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise


def _stream_response(messages: list, model: str):
    """Streaming Groq call. Yields text chunks."""
    try:
        stream = get_groq_client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        logger.error(f"Groq streaming error: {e}")
        yield f"\n\n[Error: {e}]"


def summarize_articles(articles: list[dict], model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a deep comprehensive summary of all major financial news topics.
    """
    if not articles:
        return "No articles to summarize."

    texts_list = []
    for idx, a in enumerate(articles[:25], 1):
        content = a.get('text') or a.get('summary', '')
        texts_list.append(f"{idx}. [{a.get('source', '').replace('_',' ').title()}] {a.get('title', '')}: {content[:300]}")
        
    texts = "\n\n".join(texts_list)

    system_prompt = """You are an expert financial analyst. 
Your task is to synthesize a comprehensive daily briefing summarizing ALL the important developments mentioned in the news context.
Group your summary into clean, bolded sections:
- **Macroeconomy & Policy Highlights**
- **Corporate Earnings & Board Announcements**
- **Sectoral Trends & Corporate Moves**
- **Key Risks & Vulnerabilities**
- **Key Opportunities & Catalysts**

Be detailed, structured, and thorough. Do not leave out key numbers, interest rates, policy decisions, or critical company earnings details. Write in a clear, professional analyst tone."""

    user_content = f"Synthesize a deep, structured summary of all key information and major news topics from the following articles:\n\n{texts}"

    completion = get_groq_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=800,
        temperature=0.2,
    )
    return completion.choices[0].message.content