import re
from ingestion.company_seed import COMMON_ENTITIES

def expand_query_with_companies(query: str) -> str:
    """
    Scan the query for company short names or full names from company_seed.py.
    If matches are found, append their full names, tickers, and ticker variations.
    """
    if not query:
        return query
        
    query_lower = query.lower()
    expanded_parts = []
    
    for short_name, full_name, ticker, _, _ in COMMON_ENTITIES:
        pattern = r"\b" + re.escape(short_name.lower()) + r"\b"
        if re.search(pattern, query_lower):
            ticker_clean = ticker.replace(".NS", "").replace(".BO", "")
            variations = {full_name, ticker, ticker_clean}
            expanded_parts.extend(variations)
            
    if expanded_parts:
        unique_parts = []
        for part in expanded_parts:
            if part not in unique_parts and part.lower() not in query_lower:
                unique_parts.append(part)
        if unique_parts:
            expanded_query = f"{query} ({', '.join(unique_parts)})"
            return expanded_query
            
    return query