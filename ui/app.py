"""
FinSight — Streamlit UI
Calls FastAPI backend via HTTP only. Zero direct Python imports from the backend.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import requests

from ui import api_client as api

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
[data-testid="stMetric"] {
    background: #1a1d26; border: 1px solid #2a2d3a;
    border-radius: 10px; padding: 1rem;
}
.chat-msg-user {
    background: #1e3a5f; border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem; margin: 0.5rem 0; color: #e0f0ff;
}
.article-card {
    background: #1a1d26; border: 1px solid #2a2d3a;
    border-radius: 8px; padding: 0.75rem 1rem; margin: 0.4rem 0;
}
.article-source { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
.article-title  { font-size: 14px; font-weight: 500; margin: 4px 0; color: #e0e0e0; }
hr { border-color: #2a2d3a; }
</style>
""", unsafe_allow_html=True)


# ── API health check ──────────────────────────────────────────────────────────

def check_api() -> tuple[bool, int]:
    """Returns (api_ok, doc_count)."""
    try:
        data = api.health()
        return True, data.get("db", {}).get("total_documents", 0)
    except requests.exceptions.ConnectionError:
        return False, 0
    except Exception:
        return False, 0


def check_api_detail() -> tuple[bool, dict]:
    """Returns (api_ok, health_data_dict)."""
    try:
        data = api.health()
        return True, data
    except Exception:
        return False, {}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 FinSight")
    st.markdown("*Real-Time Finance RAG*")
    st.markdown("---")

    st.markdown("**Settings**")

    use_hybrid = True

    days_window = st.slider(
        "News Age (Days)",
        min_value=1,
        max_value=30,
        value=1,
        help="Number of days back to search/retrieve news articles"
    )
    top_k       = 8

    region = "india"

    model_choice = st.selectbox(
        "Groq model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        help="70B = best · 8B = fastest · Mixtral = long context",
    )

    st.markdown("---")

    api_ok, doc_count = check_api()

    if not api_ok:
        st.error("⚠️ FastAPI not running")
        st.code("uvicorn api.main:app --reload", language="bash")
        st.caption("Start the backend first, then refresh.")
    else:
        # Fetch detailed health info
        _, health_data = check_api_detail()
        db_stats = health_data.get("db", {})
        doc_count = db_stats.get("total_documents", 0)
        last_ingest = db_stats.get("last_ingest_time") or "Pending/Never"
        
        # Display Database freshness and status
        st.markdown("**Database Status**")
        st.caption(f"🗃️ Articles Indexed: **{doc_count}**")
        st.caption(f"🕒 Last Crawled: **{last_ingest}**")
        st.caption("🔄 Runs automatically every 10m")

        # Automatically trigger cross-encoder warmup on load
        if "warmed_up" not in st.session_state:
            try:
                api.warmup()
                st.session_state["warmed_up"] = True
            except Exception:
                pass

        st.markdown("---")
        # Admin manual force trigger controls
        show_admin = st.toggle("🔧 Show Admin Controls")
        if show_admin:
            st.markdown("**Admin Controls**")
            if st.button("🔄 Force Database Ingestion", use_container_width=True):
                with st.spinner("Starting crawler on backend..."):
                    try:
                        api.ingest(days=days_window)
                        st.success("⚡ Ingestion started on backend!")
                        time.sleep(1.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to start: {e}")



# Stop everything if API is down
if not api_ok:
    st.title("📈 FinSight")
    st.error("FastAPI backend is not running. Start it with:")
    st.code("uvicorn api.main:app --reload --port 8000", language="bash")
    st.info("Then refresh this page.")
    st.stop()


# ── Navigation ────────────────────────────────────────────────────────────────

PAGES = [
    "💬 Ask FinSight", "📊 Market Pulse",
    "⭐ My Watchlist", "💼 My Portfolio", "📰 News Feed", "📈 Eval Dashboard"
]

# Read active page from URL query parameters (defaults to first page on first load)
url_page = st.query_params.get("page", None)
default_index = 0
if url_page and url_page in PAGES:
    default_index = PAGES.index(url_page)

page = st.sidebar.radio(
    "Navigate",
    PAGES,
    index=default_index,
    label_visibility="collapsed",
)

# Update query parameter on page change
if st.query_params.get("page") != page:
    st.query_params["page"] = page


# ── Shared helpers ────────────────────────────────────────────────────────────

def sentiment_card(ticker: str, s: dict):
    score = s.get("score", 0)
    color = "#22c55e" if score > 0.2 else ("#ef4444" if score < -0.2 else "#f59e0b")
    return (
        f"<div style='background:#1a1d26;border:1px solid #2a2d3a;border-radius:10px;"
        f"padding:12px;text-align:center;'>"
        f"<div style='font-size:17px;font-weight:700;'>{ticker}</div>"
        f"<div style='font-size:24px;font-weight:700;color:{color};'>"
        f"{'+' if score>0 else ''}{score:.2f}</div>"
        f"<div style='font-size:12px;color:#aaa;'>{s.get('label','')}</div>"
        f"</div>"
    )


def article_card(doc: dict):
    flag  = "🇮🇳" if doc.get("region") == "india" else "🌍"
    src   = doc.get("source","").replace("_"," ").title()
    date  = doc.get("published","")[:10]
    title = doc.get("title","")
    url   = doc.get("url","")
    return (
        f'<div class="article-card">'
        f'<div class="article-source">{flag} {src} · {date}</div>'
        f'<div class="article-title"><a href="{url}" target="_blank">{title}</a></div>'
        f'</div>'
    )


def sentiment_bar_chart(sentiments: dict, title: str):
    if not sentiments:
        return
    tickers = list(sentiments.keys())[:5]
    scores  = [sentiments[t].get("score", 0) for t in tickers]
    colors  = ["#22c55e" if s > 0.2 else ("#ef4444" if s < -0.2 else "#f59e0b") for s in scores]
    fig = go.Figure(go.Bar(
        x=tickers, y=scores, marker_color=colors,
        text=[f"{s:+.2f}" for s in scores], textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#444")
    fig.update_layout(
        title=title, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#e0e0e0"), height=260,
        margin=dict(t=40,b=20,l=20,r=20),
        yaxis=dict(range=[-1.2,1.2], gridcolor="#2a2d3a"),
        xaxis=dict(gridcolor="#2a2d3a"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ASK FINSIGHT
# ═══════════════════════════════════════════════════════════════════════════════

if page == "💬 Ask FinSight":
    st.title("💬 Ask FinSight")
    st.caption("Hybrid search (BM25 + Vector + Re-rank) · 17 trusted sources · Groq LLM")

    # Suggested questions
    st.markdown("**Quick questions:**")
    cols = st.columns(3)
    suggestions = [
        "What is Nifty doing today?",
        "Tata Motors sales performance",
        "RBI interest rate outlook",
        "TCS Q4 earnings results",
        "Reliance Industries update",
        "NSE corporate announcements",
    ]
    for i, sug in enumerate(suggestions):
        if cols[i % 3].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state["pending_q"] = sug

    st.markdown("---")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-msg-user">🙋 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} sources"):
                    for src in msg["sources"]:
                        flag = "🇮🇳" if src.get("region") == "india" else "🌍"
                        st.markdown(
                            f"{flag} **{src.get('source','').replace('_',' ').title()}** "
                            f"· {src.get('published','')}  \n"
                            f"[{src.get('title','')}]({src.get('url','')})"
                        )

    pending = st.session_state.pop("pending_q", None)
    user_input = st.chat_input("Ask anything about markets, stocks, or the economy...")
    question = pending or user_input

    if question:
        if doc_count == 0:
            st.info("ℹ️ Connecting to vector database...")

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.markdown(f'<div class="chat-msg-user">🙋 {question}</div>', unsafe_allow_html=True)

        # Show retrieval results before generation completes (~300ms)
        sources_box = st.empty()
        answer_box = st.empty()
        full_answer = ""
        sources = []
        context = ""
        t0 = time.perf_counter()

        try:
            with st.spinner("Retrieving sources..."):
                retrieved_data = api.retrieve(
                    question=question,
                    days=days_window,
                    top_k=top_k,
                    region=region,
                    use_hybrid=use_hybrid,
                )
                sources = retrieved_data.get("sources", [])
                context = retrieved_data.get("context", "")

            if sources:
                with sources_box.container():
                    # Initially show the top 3 sources to keep it clean while streaming
                    with st.expander(f"📎 Sources (retrieved in {int((time.perf_counter() - t0) * 1000)}ms)", expanded=True):
                        for src in sources[:3]:
                            flag = "🇮🇳" if src.get("region") == "india" else "🌍"
                            st.markdown(
                                f"{flag} **{src.get('source','').replace('_',' ').title()}** "
                                f"· {src.get('published','')}  \n"
                                f"[{src.get('title','')}]({src.get('url','')})"
                            )
            else:
                sources_box.warning("No relevant news found. Try increasing the days window or rephrasing.")
                st.stop()

            # Stream LLM response immediately
            def get_stream():
                for chunk in api.ask_stream(
                    question=question,
                    days=days_window,
                    top_k=top_k,
                    region=region,
                    model=model_choice,
                    use_hybrid=use_hybrid,
                    context=context,
                ):
                    yield chunk

            with answer_box.container():
                full_answer = st.write_stream(get_stream())

            # Filter sources to only those cited by the LLM in its response
            import re
            cited_sources = []
            for src in sources:
                title = src.get("title", "").strip().lower()
                clean_title = re.sub(r'[^\w\s]', '', title)
                words = clean_title.split()
                # Use a match phrase consisting of the first 4 words of the article title
                match_phrase = " ".join(words[:4]) if len(words) >= 4 else clean_title
                if match_phrase and match_phrase in re.sub(r'[^\w\s]', '', full_answer.lower()):
                    if src not in cited_sources:
                        cited_sources.append(src)

            # Fallback to the top 2 sources if no specific title match was found in the text
            if not cited_sources:
                cited_sources = sources[:2]

            # Re-render the sources box with only the cited ones in collapsed state
            with sources_box.container():
                with st.expander(f"📎 {len(cited_sources)} relevant sources (retrieved in {int((time.perf_counter() - t0) * 1000)}ms)", expanded=False):
                    for src in cited_sources:
                        flag = "🇮🇳" if src.get("region") == "india" else "🌍"
                        st.markdown(
                            f"{flag} **{src.get('source','').replace('_',' ').title()}** "
                            f"· {src.get('published','')}  \n"
                            f"[{src.get('title','')}]({src.get('url','')})"
                        )
            sources = cited_sources

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        c1, c2 = st.columns(2)
        c1.caption(f"⚡ {elapsed_ms}ms total")
        c2.caption(f"🤖 {model_choice.split('-')[0]}")

        st.session_state.chat_history.append({
            "role": "assistant", "content": full_answer, "sources": sources,
        })

    if st.session_state.get("chat_history"):
        if st.button("🗑 Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MARKET PULSE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Market Pulse":
    st.title("📊 Market Pulse")
    st.caption("Live sentiment across Indian markets")

    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        pulse_watchlist_only = st.checkbox("Filter by Watchlist", key="pulse_watchlist_checkbox")
    with col_w2:
        pulse_username = st.text_input("Username", value="default_user", key="pulse_username_input") if pulse_watchlist_only else "default_user"

    tab_india, tab_event = st.tabs(["🇮🇳 India", "📈 Event Impact"])

    def render_market_tab(fetch_fn):
        with st.spinner("Loading market data..."):
            data = fetch_fn(watchlist_only=pulse_watchlist_only, username=pulse_username)

        summary = data.get("summary","")
        if summary:
            st.markdown("### 📰 Daily Briefing & Market Summary")
            st.markdown(summary)
            st.markdown("---")



        articles = data.get("articles", [])
        if articles:
            st.markdown("### Latest articles")
            for doc in articles[:6]:
                st.markdown(article_card(doc), unsafe_allow_html=True)

    with tab_india:
        render_market_tab(api.india_summary)

    with tab_event:
        st.markdown("### Causal Event Impact Analyzer")
        st.markdown("Generate causal propagation chains for macroeconomic factors, market events, or specific sectors.")
        
        event_q = st.text_input("Enter a Market Event or Trend", value="Crude Oil Price Surge", key="event_chain_input")
        if st.button("Generate Causal Chain", key="gen_event_chain_btn", type="primary"):
            with st.spinner("Analyzing macroeconomic relationships..."):
                chain = api.get_event_impact(event_q)
                if chain:
                    st.success("Causal impact propagation chain generated:")
                    steps_html = []
                    for i, step in enumerate(chain):
                        step_disp = step
                        if "↑" in step:
                            step_disp = step.replace("↑", " <span style='color:#22c55e;'>▲</span>")
                        elif "↓" in step:
                            step_disp = step.replace("↓", " <span style='color:#ef4444;'>▼</span>")
                        card = (
                            f"<div style='background:rgba(26,29,38,0.75);border:1px solid #2a2d3a;border-radius:8px;"
                            f"padding:12px 20px;margin:5px;text-align:center;min-width:120px;box-shadow: 0 4px 6px rgba(0,0,0,0.1); flex: 1;'>"
                            f"<div style='font-size:13px;font-weight:600;color:#e0e0e0;'>{step_disp}</div>"
                            f"</div>"
                        )
                        steps_html.append(card)
                        if i < len(chain) - 1:
                            arrow = (
                                f"<div style='display:flex;align-items:center;justify-content:center;color:#4f46e5;"
                                f"font-size:20px;font-weight:bold;margin:0 5px;'>➔</div>"
                            )
                            steps_html.append(arrow)
                    container = (
                        f"<div style='display:flex;align-items:center;justify-content:center;flex-wrap:wrap;"
                        f"background:#0d0e12;border:1px solid #1a1d26;border-radius:12px;padding:15px;margin-bottom:20px;width:100%;'>"
                        f"{''.join(steps_html)}"
                        f"</div>"
                    )
                    st.markdown(container, unsafe_allow_html=True)
                else:
                    st.error("Failed to generate causal chain.")


# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⭐ My Watchlist":
    st.title("⭐ My Watchlist")
    st.caption("Track companies you care about and get a daily briefing")

    username = st.text_input("Username", value="default_user", help="Watchlist is saved per user")

    watchlist = api.get_watchlist(username)

    st.markdown("### Add Ticker to Watchlist")
    col_add1, col_add2 = st.columns([3, 1])
    with col_add1:
        new_ticker = st.text_input("Add Company Name (e.g. HDFC BANK, Reliance, Bharti Airtel)", key="add_ticker_input").upper().strip()
    with col_add2:
        st.write("") # spacing
        st.write("") # spacing
        if st.button("➕ Add", use_container_width=True):
            if new_ticker:
                cleaned_input = new_ticker.replace(",NS", ".NS").replace(",BO", ".BO")
                tickers_to_add = [t.strip() for t in cleaned_input.split(",") if t.strip()]
                
                success_added = []
                for t in tickers_to_add:
                    if api.add_to_watchlist(username, t):
                        success_added.append(t)
                
                if success_added:
                    st.success(f"Added: {', '.join(success_added)}")
                    st.rerun()
                else:
                    st.error("Failed to add ticker(s)")
            else:
                st.warning("Please enter a ticker symbol")

    st.markdown("---")
    st.markdown("### Your Tickers")
    if not watchlist:
        st.info("Your watchlist is empty. Add some tickers above!")
    else:
        # Display watchlist as columns with delete buttons
        for i, t in enumerate(watchlist):
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**{t}**")
            with cols[1]:
                if st.button("🗑️", key=f"del_{t}_{i}"):
                    if api.remove_from_watchlist(username, t):
                        st.success(f"Removed {t}")
                        st.rerun()
                    else:
                        st.error("Failed to remove ticker")

        st.markdown("---")
        st.markdown("### Watchlist Daily Briefing")
        hours = st.selectbox("Briefing timeframe", [24, 48, 168], format_func=lambda x: f"Last {x} hours" if x < 168 else "Last 7 days")
        
        if st.button("⚡ Generate Watchlist Brief", type="primary", use_container_width=True):
            with st.spinner("Generating your daily watchlist brief..."):
                brief_area = st.empty()
                full_brief = ""
                try:
                    for chunk in api.get_watchlist_brief(username, model=model_choice, stream=True, hours=hours):
                        full_brief += chunk
                        brief_area.markdown(full_brief)
                except Exception as e:
                    # Fallback to non-stream if stream fails
                    res = api.get_watchlist_brief(username, model=model_choice, stream=False, hours=hours)
                    brief_area.markdown(res.get("answer", "Error generating brief."))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3.7 — MY PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💼 My Portfolio":
    st.title("💼 My Portfolio")
    st.caption("Manage your holdings and analyze exposure & estimated sentiment impact")

    username = st.text_input("Username", value="default_user", help="Portfolio is saved per user")

    portfolio = api.get_portfolio(username)

    st.markdown("### Add / Update Stock Holding")
    col_port1, col_port2, col_port3 = st.columns([3, 2, 1])
    with col_port1:
        port_ticker = st.text_input("Add Company (e.g. HDFC BANK, TCS, RELIANCE)", key="port_ticker_input").upper().strip()
    with col_port2:
        port_weight = st.slider("Allocation Weight (%)", min_value=1, max_value=100, value=10, key="port_weight_input")
    with col_port3:
        st.write("") # spacing
        st.write("") # spacing
        if st.button("➕ Save", key="save_holding_btn", use_container_width=True):
            if port_ticker:
                # Convert weight percentage to fraction
                weight_fraction = port_weight / 100.0
                if api.add_to_portfolio(username, port_ticker, weight_fraction):
                    st.success(f"Saved {port_ticker} with {port_weight}% weight")
                    st.rerun()
                else:
                    st.error("Failed to save holding")
            else:
                st.warning("Please enter a ticker symbol")

    st.markdown("---")
    st.markdown("### Your Holdings")
    
    if not portfolio:
        st.info("Your portfolio is empty. Add some holdings above!")
    else:
        # Check if weights sum to 1.0 (approx)
        total_p_weight = sum(portfolio.values())
        if abs(total_p_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Total allocation is {total_p_weight*100:.1f}%. Weights will be normalized to 100% for analysis.")
            
        for i, (t, w) in enumerate(portfolio.items()):
            cols = st.columns([3, 2, 1])
            with cols[0]:
                st.markdown(f"**{t}**")
            with cols[1]:
                st.markdown(f"{w*100:.1f}% allocation")
            with cols[2]:
                if st.button("🗑️", key=f"del_port_{t}_{i}"):
                    if api.remove_from_portfolio(username, t):
                        st.success(f"Removed {t}")
                        st.rerun()
                    else:
                        st.error("Failed to remove holding")

        st.markdown("---")
        st.markdown("### Portfolio Exposure Analysis")
        
        if st.button("🔍 Run Portfolio Intelligence Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing portfolio exposure..."):
                analysis = api.get_portfolio_analysis(username)
                if analysis.get("status") == "success":
                    st.markdown("#### Portfolio Sentiment Performance")
                    
                    impact_val = analysis.get("portfolio_impact", "0.0%")
                    # Color based on sign
                    color = "#22c55e" if "+" in impact_val else ("#ef4444" if "-" in impact_val else "#f59e0b")
                    
                    st.markdown(
                        f"<div style='background:#101218;border:2px solid {color};border-radius:15px;"
                        f"padding:20px;text-align:center;margin-bottom:20px;'>"
                        f"<div style='font-size:16px;color:#aaa;'>Estimated Sentiment Shift</div>"
                        f"<div style='font-size:42px;font-weight:800;color:{color};'>{impact_val}</div>"
                        f"<div style='font-size:14px;color:#ccc;margin-top:8px;font-style:italic;'>{analysis.get('reason','')}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("#### Asset Breakdown")
                    for item in analysis.get("exposure", []):
                        ticker = item["ticker"]
                        weight = item["weight"]
                        score = item["sentiment_score"]
                        impact = item["impact"]
                        reason = item["reason"]
                        
                        # Determine card border/indicator color
                        card_color = "#22c55e" if score >= 0.15 else ("#ef4444" if score <= -0.15 else "#f59e0b")
                        
                        st.markdown(
                            f"<div style='background:#1a1d26;border-left:5px solid {card_color};border-radius:6px;padding:12px;margin-bottom:10px;'>"
                            f"<div style='display:flex;justify-content:between;align-items:center;'>"
                            f"<b>{ticker}</b> ({weight:.0%}) &nbsp;·&nbsp; <span style='color:{card_color};'>{impact} ({score:+.2f})</span>"
                            f"</div>"
                            f"<div style='font-size:12px;color:#aaa;margin-top:4px;'>{reason}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.error("Failed to run portfolio analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — NEWS FEED
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📰 News Feed":
    st.title("📰 News Feed")

    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        feed_watchlist_only = st.checkbox("Filter by Watchlist", key="feed_watchlist_checkbox")
    with col_w2:
        feed_username = st.text_input("Username", value="default_user", key="feed_username_input") if feed_watchlist_only else "default_user"

    search_q = st.text_input(
        "Search", placeholder="interest rates, IPO, earnings, merger...",
        label_visibility="collapsed",
    )
    if not search_q:
        search_q = "latest financial market news stocks economy"

    with st.spinner("Loading articles from API..."):
        data = api.feed(
            query=search_q, days=days_window, top_k=20, region=region,
            watchlist_only=feed_watchlist_only, username=feed_username
        )

    articles = data.get("articles", [])
    st.caption(f"{len(articles)} articles found via `POST /feed`")

    if not articles:
        st.info("No articles found. Try widening the date range.")
    else:
        # Source distribution
        src_counts: dict[str,int] = {}
        for a in articles:
            s = a.get("source","?").replace("_"," ").title()
            src_counts[s] = src_counts.get(s,0)+1
        st.markdown("---")

        for a in articles:
            with st.expander(f"🇮🇳  {a.get('title','')[:85]}"):
                c1, c2 = st.columns([2,1])
                c1.markdown(f"**Source:** {a.get('source','').replace('_',' ').title()}  ·  **Date:** {a.get('published','')}")
                c1.markdown(a.get("summary","")[:200] + "...")
                c2.markdown(f"[Read full article →]({a.get('url','')})")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EVAL DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Eval Dashboard":
    st.title("📈 Evaluation Dashboard")

    run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Running evaluation (with LLM judge) on backend... (~1 min)"):
            try:
                data = api.run_eval(quick=False, days=30)
                st.session_state["eval_data"] = data
                st.success("Done!")
            except Exception as e:
                st.error(f"Eval failed: {e}")

    # Load saved results locally if available
    if "eval_data" not in st.session_state:
        import glob, json
        files = sorted(glob.glob("evaluation/results/eval_*.json"))
        if files:
            with open(files[-1]) as f:
                raw = json.load(f)
            st.session_state["eval_data"] = {"summary": raw["summary"], "results": raw["results"]}
            st.caption(f"Loaded from {os.path.basename(files[-1])}")

    if "eval_data" not in st.session_state:
        st.info("No results yet. Click **Run Evaluation** above.")
        st.markdown("""
        **Metrics explained:**
        - **Precision@5** — of the top-5 retrieved articles, what fraction are relevant?
        - **Hit Rate@5** — did at least one relevant article appear in top-5?
        - **MRR** — Mean Reciprocal Rank (how early does the first relevant article appear?)
        - **NDCG@5** — ranking quality (relevant articles ranked higher = better score)
        - **Faithfulness** — is every claim grounded in context? (anti-hallucination)
        - **Relevance** — does the answer address the question?
        - **Completeness** — are key points covered?
        """)
        st.stop()

    data    = st.session_state["eval_data"]
    agg     = data.get("summary", {})
    results = data.get("results", [])
    r       = agg.get("retrieval", {})
    aq      = agg.get("answer_quality", {})

    def fmt(v):  return f"{v:.2%}" if v is not None else "N/A"
    def fmt4(v): return f"{v:.4f}" if v is not None else "N/A"

    st.markdown("### Retrieval metrics")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Precision@5", fmt(r.get("precision@5")))
    m2.metric("Hit Rate@5",  fmt(r.get("hit_rate@5")))
    m3.metric("Recall@5",    fmt(r.get("recall@5")))
    m4.metric("NDCG@5",      fmt4(r.get("ndcg@5")))
    m5.metric("MRR",         fmt4(r.get("mrr")))

    if aq:
        st.markdown("### Answer quality (LLM-as-judge)")
        a1, a2, a3, a4, a5 = st.columns(5)
        
        faithfulness_val = aq.get("faithfulness")
        hallucination_rate = (1.0 - faithfulness_val) if faithfulness_val is not None else None
        
        a1.metric("Faithfulness", fmt(faithfulness_val))
        a2.metric("Relevance",    fmt(aq.get("relevance")))
        a3.metric("Answer Accuracy", fmt(aq.get("completeness")))
        a4.metric("Hallucination Rate", fmt(hallucination_rate))
        a5.metric("⭐ Overall",   fmt(aq.get("overall")))