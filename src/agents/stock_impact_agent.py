from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.agents.base_agent import BaseAgent
from src.models import AgentState, Entity, StockImpact
from src.agents.prompts import impact_prompt
from src.config import config


class StockImpactAgent(BaseAgent):
    def __init__(self):
        super().__init__("StockImpactAgent")
        self.company_symbols = {
            "HDFC Bank": "HDFCBANK",
            "ICICI Bank": "ICICIBANK",
            "State Bank": "SBIN",
            "Reliance": "RELIANCE",
            "TCS": "TCS",
            "Infosys": "INFY",
            "Wipro": "WIPRO",
            "Asian Paints": "ASIANPAINT",
            "ITC": "ITC",
            "Larsen": "LT",
        }
        self.sector_stocks = {
            "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
            "Technology": ["TCS", "INFY", "WIPRO", "TECHM", "HCLTECH"],
            "Pharmaceutical": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
            "Energy": ["RELIANCE", "ONGC", "BPCL", "IOC"],
            "Financial Services": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE"],
            "Automotive": ["MARUTI", "M&M", "TATAMOTORS", "HEROMOTOCO"],
        }
        
        ### impact analysis prompt
        self.impact_prompt = ChatPromptTemplate.from_messages([
            ("system", impact_prompt),
            ("human", """Article: {title}

Content: {content}

Extracted Entities:
Companies: {companies}
Sectors: {sectors}
Regulators: {regulators}
Events: {events}""")
        ])
        
        self.parser = JsonOutputParser()
    
    def map_company_to_symbol(self, company: str) -> str:
        """Map company name to stock symbol."""
        for name, symbol in self.company_symbols.items():
            if name.lower() in company.lower() or company.lower() in name.lower():
                return symbol
        return company.upper().replace(" ", "")[:10]
    
    def get_direct_impacts(self, entities: List[Entity]) -> List[StockImpact]:
        """Get direct stock impacts from company entities."""
        impacts = []
        
        for entity in entities:
            if entity.type == "COMPANY":
                symbol = self.map_company_to_symbol(entity.text)
                impacts.append(StockImpact(
                    symbol=symbol,
                    confidence= config.CONFIDENCE_LEVELS["direct"],
                    type="direct",
                    reasoning=f"Company '{entity.text}' directly mentioned"
                ))
        
        return impacts
    
    def get_sector_impacts(self, entities: List[Entity]) -> List[StockImpact]:
        """Get sector-wide stock impacts."""
        impacts = []
        
        for entity in entities:
            if entity.type == "SECTOR":
                sector = entity.text
                if sector in self.sector_stocks:
                    for symbol in self.sector_stocks[sector]:
                        impacts.append(StockImpact(
                            symbol=symbol,
                            confidence= config.CONFIDENCE_LEVELS["sector_high"],
                            type="sector",
                            reasoning=f"Sector '{sector}' impacted"
                        ))
        
        return impacts
    
    def get_regulatory_impacts(self, entities: List[Entity]) -> List[StockImpact]:
        """Get regulatory impacts on stocks."""
        impacts = []
        
        # Check if there are regulator entities
        regulators = [e for e in entities if e.type == "REGULATOR"]
        if not regulators:
            return impacts
        
        # Check affected sectors
        sectors = [e for e in entities if e.type == "SECTOR"]
        for sector in sectors:
            if sector.text in self.sector_stocks:
                for symbol in self.sector_stocks[sector.text]:
                    impacts.append(StockImpact(
                        symbol=symbol,
                        confidence= config.CONFIDENCE_LEVELS["regulatory"],
                        type="regulatory",
                        reasoning=f"Regulatory action by {regulators[0].text} affecting {sector.text}"
                    ))
        
        return impacts
    
    def analyze_with_llm( self, title: str, content: str, entities: List[Entity] ) -> List[StockImpact]:
        """Use LLM for advanced impact analysis."""
        try:
            companies = [e.text for e in entities if e.type == "COMPANY"]
            sectors = [e.text for e in entities if e.type == "SECTOR"]
            regulators = [e.text for e in entities if e.type == "REGULATOR"]
            events = [e.text for e in entities if e.type == "EVENT"]
            
            chain = self.impact_prompt | self.llm | self.parser
            result = chain.invoke({
                "title": title,
                "content": content,
                "companies": ", ".join(companies) if companies else "None",
                "sectors": ", ".join(sectors) if sectors else "None",
                "regulators": ", ".join(regulators) if regulators else "None",
                "events": ", ".join(events) if events else "None",
            })
            
            return [StockImpact(**item) for item in result]
            
        except Exception as e:
            self.log_error(f"LLM impact analysis failed: {e}")
            return []
    
    def merge_impacts(self, impacts: List[StockImpact]) -> List[StockImpact]:
        """Merge duplicate stock impacts, keeping highest confidence."""
        seen = {}
        
        for impact in impacts:
            if impact.symbol not in seen:
                seen[impact.symbol] = impact
            else:
                # Keep the one with higher confidence
                if impact.confidence > seen[impact.symbol].confidence:
                    seen[impact.symbol] = impact
        
        return list(seen.values())
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stocks impact from article."""
        try:
            agent_state = AgentState(**state)
            article = agent_state.article
            # Skip if duplicate
            if agent_state.is_duplicate:
                self.log_info(f"Skipping impact analysis for duplicate article {article.article_id}")
                return agent_state.model_dump()
            
            self.log_info(f"Analyzing stock impacts for: {article.article_id}")
            
            impacts = []
            ## direct company impacts
            impacts.extend(self.get_direct_impacts(agent_state.entities))
            ## sector impacts
            impacts.extend(self.get_sector_impacts(agent_state.entities))
            ## regulatory impacts
            impacts.extend(self.get_regulatory_impacts(agent_state.entities))
            ## LLM for additional analysis
            llm_impacts = self.analyze_with_llm(
                article.title,
                article.content,
                agent_state.entities
            )
            impacts.extend(llm_impacts)
            
            # Merge duplicate impacts
            impacts = self.merge_impacts(impacts)
            
            agent_state.stock_impacts = impacts
            agent_state.processing_metadata["impact_count"] = len(impacts)
            
            self.log_info(f"Identified {len(impacts)} stock impacts for {article.article_id}")
            
            return agent_state.model_dump()
            
        except Exception as e:
            self.log_error(f"Stock impact analysis failed: {e}")
            state["error"] = str(e)
            return state