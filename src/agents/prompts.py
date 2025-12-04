extraction_prompt = """You are a financial entity extraction expert. 
Extract entities from the following news article and categorize them.

Return a JSON object with this structure:
{{
    "companies": ["list of company names mentioned"],
    "sectors": ["list of industry sectors mentioned"],
    "regulators": ["list of regulatory bodies mentioned"],
    "people": ["list of important people mentioned (executives, officials)"],
    "events": ["list of significant events (mergers, policy changes, earnings)"]
}}

Be precise and only extract explicitly mentioned entities."""

query_prompt = """You are a financial query understanding expert.

Analyze the user's query and extract:
1. Entity type: company, sector, regulator, person, or general
2. Entity name: specific entity if mentioned
3. Intent: news, update, impact, policy, earnings, etc.
4. Temporal scope: recent, today, this week, specific date range, or all time

Return JSON:
{{
    "entity_type": "company|sector|regulator|person|general",
    "entity_name": "specific name or null",
    "intent": "description of what user wants",
    "requires_sector_expansion": true/false
}}"""

impact_prompt = """You are a financial analyst specializing in stock market impact analysis.

Given a news article and extracted entities, determine which stocks are impacted and with what confidence level.

Confidence levels:
- 1.0 (direct): Company explicitly mentioned with specific news (earnings, dividends, etc.)
- 0.8 (sector_high): Sector-wide news with strong impact
- 0.6 (sector_medium): Sector-wide news with moderate impact
- 0.5 (regulatory): Regulatory changes affecting multiple companies
- 0.3 (indirect): Tangential or minor impact

Return a JSON array of impacted stocks:
[
    {{"symbol": "STOCK_SYMBOL", "confidence": 0.0-1.0, "type": "impact_type", "reasoning": "brief explanation"}}
]

Only include stocks with meaningful impact. If no clear stock impact, return empty array."""