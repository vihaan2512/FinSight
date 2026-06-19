"""
Synthetically generated evaluation dataset.
"""

EVAL_DATASET = [
    {
        "query": "What are the latest stock market updates in India?",
        "region": "india",
        "relevant_keywords": [
            "stock",
            "market",
            "india"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "sensex",
            "nifty",
            "bse/nse"
        ],
        "bad_answer_clues": [
            "commodity prices"
        ],
        "category": "stock",
        "id": "SYNTH_01"
    },
    {
        "query": "What was the closing value of the Sensex on Friday?",
        "region": "india",
        "relevant_keywords": [
            "sensex",
            "nifty"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "75,527.95",
            "1.73%"
        ],
        "bad_answer_clues": [
            "us fed"
        ],
        "category": "stock",
        "id": "SYNTH_02"
    },
    {
        "query": "How will the US-Iran war affect the Indian stock market next week?",
        "region": "india",
        "relevant_keywords": [
            "iran",
            "war",
            "market"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "geopolitical tension",
            "positive note"
        ],
        "bad_answer_clues": [
            "brexit"
        ],
        "category": "macro",
        "id": "SYNTH_04"
    },
    {
        "query": "Which stocks are favored by YES Securities for June 2026?",
        "region": "india",
        "relevant_keywords": [
            "yes",
            "securities",
            "june"
        ],
        "expected_tickers": [
            "ENTERO.NS",
            "LUPIN",
            "MARUTI"
        ],
        "ideal_answer_clues": [
            "entero healthcare",
            "maruti suzuki",
            "lupin"
        ],
        "bad_answer_clues": [
            "foreign stocks"
        ],
        "category": "stock",
        "id": "SYNTH_05"
    },
    {
        "query": "What type of announcements did Malt Land Distilleries Ltd make?",
        "region": "india",
        "relevant_keywords": [
            "malt",
            "land",
            "distilleries"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "announcements"
        ],
        "bad_answer_clues": [
            "financial results"
        ],
        "category": "stock",
        "id": "SYNTH_15"
    },
    {
        "query": "How many companies are launching public offers this week in India's primary market?",
        "region": "india",
        "relevant_keywords": [
            "ipo",
            "companies",
            "launch"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "two companies",
            "quiet week"
        ],
        "bad_answer_clues": [
            "many companies"
        ],
        "category": "stock",
        "id": "SYNTH_16"
    },
    {
        "query": "How many SME IPOs are lined up for the upcoming week?",
        "region": "india",
        "relevant_keywords": [
            "sme",
            "ipos",
            "listings"
        ],
        "expected_tickers": [
            "SME"
        ],
        "ideal_answer_clues": [
            "4 sme ipos",
            "next week"
        ],
        "bad_answer_clues": [
            "mainboard issue"
        ],
        "category": "stock",
        "id": "SYNTH_22"
    },
    {
        "query": "When is Razorpay expected to file its DRHP for IPO?",
        "region": "india",
        "relevant_keywords": [
            "razorpay",
            "drhp"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "next week",
            "sources say"
        ],
        "bad_answer_clues": [
            "this year"
        ],
        "category": "stock",
        "id": "SYNTH_23"
    },
    {
        "query": "How much fundraising has Ather Energy's board approved?",
        "region": "india",
        "relevant_keywords": [
            "ather",
            "energy",
            "fundraising"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "Rs 2,500-cr",
            "fundraising intiative"
        ],
        "bad_answer_clues": [
            "share price"
        ],
        "category": "stock",
        "id": "SYNTH_26"
    },
    {
        "query": "Which companies will go ex-date next week for dividend stocks?",
        "region": "india",
        "relevant_keywords": [
            "tata",
            "dividend",
            "stocks"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "tata tech",
            "sanofi consumer",
            "ex-date"
        ],
        "bad_answer_clues": [
            "price target"
        ],
        "category": "stock",
        "id": "SYNTH_30"
    },
    {
        "query": "What is the purpose of the record date fixed by Sun Pharmaceutical Industries?",
        "region": "india",
        "relevant_keywords": [
            "sun",
            "pharmaceutical",
            "dividend"
        ],
        "expected_tickers": [
            "SUNPHARMA",
            "SUNPHARMA.NS"
        ],
        "ideal_answer_clues": [
            "final dividend",
            "record date"
        ],
        "bad_answer_clues": [
            "share buyback"
        ],
        "category": "stock",
        "id": "SYNTH_32"
    },
    {
        "query": "What is the current promoter shareholding in Sudarshan Pharma?",
        "region": "india",
        "relevant_keywords": [
            "sudarshan",
            "pharma",
            "shareholding"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "58.93%",
            "promoter shareholding"
        ],
        "bad_answer_clues": [
            "icici bank"
        ],
        "category": "stock",
        "id": "SYNTH_33"
    },
    {
        "query": "Why did Zee Entertainment stock rise 4%?",
        "region": "india",
        "relevant_keywords": [
            "zee",
            "entertainment",
            "gains"
        ],
        "expected_tickers": [
            "ZEEL.NS"
        ],
        "ideal_answer_clues": [
            "fund raise plan",
            "4% gain",
            "25% so far in june"
        ],
        "bad_answer_clues": [
            "merger news"
        ],
        "category": "stock",
        "id": "SYNTH_34"
    },
    {
        "query": "How much have FPIs withdrawn from Indian equities in the first fortnight of June?",
        "region": "india",
        "relevant_keywords": [
            "fpi",
            "equities",
            "june"
        ],
        "expected_tickers": [
            "FPI"
        ],
        "ideal_answer_clues": [
            "\u20b962,800 crore",
            "first fortnight",
            "june"
        ],
        "bad_answer_clues": [
            "may",
            "assets"
        ],
        "category": "macro",
        "id": "SYNTH_36"
    },
    {
        "query": "What type of fund is Motilal Oswal BSE Clean Environment Index Fund",
        "region": "india",
        "relevant_keywords": [
            "motilal",
            "oswal",
            "index"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "equity",
            "index fund",
            "motilal oswal"
        ],
        "bad_answer_clues": [
            "debt fund"
        ],
        "category": "stock",
        "id": "SYNTH_38"
    },
    {
        "query": "Which stocks held by over 100 mutual funds in May have seen the highest surge in the last 5 months?",
        "region": "india",
        "relevant_keywords": [
            "stocks",
            "mutual",
            "funds"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "13 stocks",
            "up to 85%",
            "5 months"
        ],
        "bad_answer_clues": [
            "individual investor"
        ],
        "category": "stock",
        "id": "SYNTH_40"
    },
    {
        "query": "What led to the Sensex soaring over 900 points and Nifty gaining 271 points?",
        "region": "india",
        "relevant_keywords": [
            "sensex",
            "nifty",
            "optimism"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "global optimism",
            "tax reforms"
        ],
        "bad_answer_clues": [
            "inflation concerns"
        ],
        "category": "index",
        "id": "SYNTH_41"
    },
    {
        "query": "What are the expected returns on top stocks rated 'Buy' this week?",
        "region": "india",
        "relevant_keywords": [
            "stocks",
            "buy",
            "returns"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "14% to 68%",
            "brokerages",
            "projected returns"
        ],
        "bad_answer_clues": [
            "negative returns"
        ],
        "category": "stock",
        "id": "SYNTH_42"
    },
    {
        "query": "What is the new limit for bank exposure to completed, cash-generating assets set by the RBI?",
        "region": "india",
        "relevant_keywords": [
            "rbi",
            "reit",
            "lending"
        ],
        "expected_tickers": [
            "REIT"
        ],
        "ideal_answer_clues": [
            "49% limit",
            "cash-generating assets"
        ],
        "bad_answer_clues": [
            "under-construction projects"
        ],
        "category": "macro",
        "id": "SYNTH_43"
    },
    {
        "query": "How did Hexagon Nutrition perform on its market debut?",
        "region": "india",
        "relevant_keywords": [
            "hexagon",
            "nutrition",
            "debut"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "gains",
            "market debut"
        ],
        "bad_answer_clues": [
            "losses"
        ],
        "category": "stock",
        "id": "SYNTH_44"
    },
    {
        "query": "What is driving the growth of the data centre generator market in India?",
        "region": "india",
        "relevant_keywords": [
            "data",
            "centre",
            "growth"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "ai",
            "edge",
            "steady growth"
        ],
        "bad_answer_clues": [
            "revenue drop"
        ],
        "category": "sector",
        "id": "SYNTH_52"
    },
    {
        "query": "Why did the Nifty Metal index decline while other sectors gained?",
        "region": "india",
        "relevant_keywords": [
            "nifty",
            "metal",
            "sector"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "nifty metal index",
            "loser",
            "all others traded with gains"
        ],
        "bad_answer_clues": [
            "global markets"
        ],
        "category": "sector",
        "id": "SYNTH_61"
    },
    {
        "query": "What is the reason for the notice of demand issued against Chandrima Mercantiles Limited?",
        "region": "india",
        "relevant_keywords": [
            "chandrima",
            "mercantiles",
            "manipulation"
        ],
        "expected_tickers": [
            "CHANDRIMA.BO"
        ],
        "ideal_answer_clues": [
            "price and volume manipulation",
            "quasar india limited",
            "rc no. 9155"
        ],
        "bad_answer_clues": [
            "revenue growth"
        ],
        "category": "stock",
        "id": "SYNTH_69"
    },
    {
        "query": "Why has India Inc's cash hoard increased to over $200 billion in FY26?",
        "region": "india",
        "relevant_keywords": [
            "cash",
            "hoard",
            "expansion"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "holding back",
            "200 billion",
            "fy26"
        ],
        "bad_answer_clues": [
            "inflation rate"
        ],
        "category": "macro",
        "id": "SYNTH_77"
    },
    {
        "query": "What were the half yearly earnings for CMR Green Tech?",
        "region": "india",
        "relevant_keywords": [
            "cmr",
            "green",
            "earnings"
        ],
        "expected_tickers": [
            "CMR"
        ],
        "ideal_answer_clues": [
            "half yearly",
            "results",
            "economic times"
        ],
        "bad_answer_clues": [
            "annual report"
        ],
        "category": "stock",
        "id": "SYNTH_79"
    },
    {
        "query": "What factors will influence the Indian stock market this week?",
        "region": "india",
        "relevant_keywords": [
            "market",
            "iran",
            "oil"
        ],
        "expected_tickers": [],
        "ideal_answer_clues": [
            "us-iran negotiations",
            "crude oil prices",
            "strait of hormuz"
        ],
        "category": "stock",
        "id": "SYNTH_92"
    },
]