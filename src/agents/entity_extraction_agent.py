import re
from typing import Dict, Any, List
import spacy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.agents.base_agent import BaseAgent
from src.agents.prompts import extraction_prompt
from src.models import AgentState, Entity


class EntityExtractionAgent(BaseAgent):
    def __init__(self):
        super().__init__("EntityExtractionAgent")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.log_error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        self.sector_keywords = {
            "Banking": ["bank", "banking", "credit", "loan", "mortgage"],
            "Technology": ["tech", "software", "ai", "cloud", "digital"],
            "Pharmaceutical": ["pharma", "drug", "medicine", "healthcare"],
            "Energy": ["oil", "gas", "energy", "power", "renewable"],
            "Financial Services": ["finance", "investment", "insurance", "asset management"],
            "Automotive": ["auto", "car", "vehicle", "automotive"],
            "Telecommunications": ["telecom", "mobile", "network", "5g"],
            "Retail": ["retail", "consumer", "e-commerce", "shopping"],
        }
        
        self.regulator_keywords = [
            "RBI", "Reserve Bank", "SEBI", "SEC", "Federal Reserve", "Fed",
            "Central Bank", "Monetary Authority", "Financial Authority"
        ]
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", extraction_prompt),
            ("human", "Title: {title}\n\nContent: {content}")
        ])
        
        self.parser = JsonOutputParser()
    
    def extract_with_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy NER."""
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities.append(Entity(
                    text=ent.text,
                    type="COMPANY",
                    confidence=0.8
                ))
            elif ent.label_ == "PERSON":
                entities.append(Entity(
                    text=ent.text,
                    type="PERSON",
                    confidence=0.8
                ))
        
        return entities
    
    def extract_sectors(self, text: str) -> List[Entity]:
        """Extract sector information from text."""
        text_lower = text.lower()
        entities = []
        
        for sector, keywords in self.sector_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    entities.append(Entity(
                        text=sector,
                        type="SECTOR",
                        confidence=0.7
                    ))
                    break
        
        return entities
    
    def extract_regulators(self, text: str) -> List[Entity]:
        """Extract regulator mentions."""
        entities = []
        
        for regulator in self.regulator_keywords:
            pattern = r'\b' + re.escape(regulator) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=regulator,
                    type="REGULATOR",
                    confidence=0.9
                ))
        
        return entities
    
    def extract_with_llm(self, title: str, content: str) -> Dict[str, List[str]]:
        """Extract entities using LLM for better accuracy."""
        try:
            chain = self.extraction_prompt | self.llm | self.parser
            result = chain.invoke({"title": title, "content": content})
            return result
        except Exception as e:
            self.log_error(f"LLM extraction failed: {e}")
            return {
                "companies": [],
                "sectors": [],
                "regulators": [],
                "people": [],
                "events": []
            }
    
    def merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge duplicate entities."""
        seen = {}
        merged = []
        
        for entity in entities:
            key = (entity.text.lower(), entity.type)
            if key not in seen:
                seen[key] = entity
                merged.append(entity)
            else:
                if entity.confidence > seen[key].confidence:
                    seen[key].confidence = entity.confidence
        
        return merged
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enteties from articles"""
        try:
            agent_state = AgentState(**state)
            article = agent_state.article
            
            ### skip if duplicate
            if agent_state.is_duplicate:
                self.log_info(f"Skipping entity extraction for duplicate article {article.article_id}")
                return agent_state.model_dump()
            
            self.log_info(f"Extracting entities from: {article.article_id}")
            
            full_text = f"{article.title} {article.content}"
            entities = []
            
            ## extract with spaCy NER
            entities.extend(self.extract_with_spacy(full_text))
            # Extract sectors
            entities.extend(self.extract_sectors(full_text))
            ## Extract regulators
            entities.extend(self.extract_regulators(full_text))
            ## Extract with LLM for better accuracy
            llm_entities = self.extract_with_llm(article.title, article.content)
            
            # Add LLM extracted entities
            for company in llm_entities.get("companies", []):
                entities.append(Entity(text=company, type="COMPANY", confidence=0.9))
            
            for sector in llm_entities.get("sectors", []):
                entities.append(Entity(text=sector, type="SECTOR", confidence=0.9))
            
            for regulator in llm_entities.get("regulators", []):
                entities.append(Entity(text=regulator, type="REGULATOR", confidence=0.95))
            
            for person in llm_entities.get("people", []):
                entities.append(Entity(text=person, type="PERSON", confidence=0.85))
            
            for event in llm_entities.get("events", []):
                entities.append(Entity(text=event, type="EVENT", confidence=0.8))
            
            # Merge duplicates
            entities = self.merge_entities(entities)
            
            agent_state.entities = entities
            agent_state.processing_metadata["entity_count"] = len(entities)
            
            self.log_info(f"Extracted {len(entities)} entities from {article.article_id}")
            
            return agent_state.model_dump()
            
        except Exception as e:
            self.log_error(f"Entity extraction failed: {e}")
            state["error"] = str(e)
            return state