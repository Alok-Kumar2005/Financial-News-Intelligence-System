from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.models import AgentState, NewsArticleDB, ExtractedEntityDB, StockImpactDB
from src.vector_store import VectorStoreManager
from src.database import get_db_session


class StorageAgent(BaseAgent):
    def __init__(self, vector_store: VectorStoreManager):
        super().__init__("StorageAgent")
        self.vector_store = vector_store
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ Store article and extracted information."""
        try:
            agent_state = AgentState(**state)
            article = agent_state.article
            
            self.log_info(f"Storing article: {article.article_id}")
            
            with get_db_session() as session:
                ### storing article in database
                db_article = NewsArticleDB(
                    article_id=article.article_id,
                    title=article.title,
                    content=article.content,
                    source=article.source,
                    published_at=article.published_at,
                    url=article.url,
                    is_duplicate=agent_state.is_duplicate,
                    duplicate_of=agent_state.duplicate_of
                )
                session.add(db_article)
                
                ### sstoring entities if not duplicate
                if not agent_state.is_duplicate:
                    for entity in agent_state.entities:
                        db_entity = ExtractedEntityDB(
                            article_id=article.article_id,
                            entity_text=entity.text,
                            entity_type=entity.type,
                            confidence=entity.confidence
                        )
                        session.add(db_entity)
                    
                    ### storing stock impacts
                    for impact in agent_state.stock_impacts:
                        db_impact = StockImpactDB(
                            article_id=article.article_id,
                            stock_symbol=impact.symbol,
                            confidence=impact.confidence,
                            impact_type=impact.type,
                            reasoning=impact.reasoning
                        )
                        session.add(db_impact)
                
                session.commit()
            
            # Store in vector store if not duplicate
            if not agent_state.is_duplicate:
                entity_texts = [e.text for e in agent_state.entities]
                self.vector_store.add_article(article, entity_texts)
            
            agent_state.processing_metadata["stored"] = True
            self.log_info(f"Successfully stored article {article.article_id}")
            
            return agent_state.model_dump()
            
        except Exception as e:
            self.log_error(f"Storage failed: {e}")
            state["error"] = str(e)
            state["processing_metadata"]["stored"] = False
            return state