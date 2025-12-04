from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.vector_store import VectorStoreManager
from src.models import AgentState


class DeduplicationAgent(BaseAgent):
    def __init__(self, vector_store: VectorStoreManager):
        super().__init__("DeduplicationAgent")
        self.vector_store = vector_store
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_state = AgentState(**state)
            article = agent_state.article
            
            self.log_info(f"Checking for duplicates: {article.article_id}")
            # Find potential duplicates
            duplicates = self.vector_store.find_duplicates(article)
            
            if duplicates:
                # Mark as duplicate and reference the most similar article
                duplicate_id, similarity = duplicates[0]
                agent_state.is_duplicate = True
                agent_state.duplicate_of = duplicate_id
                agent_state.processing_metadata["duplicate_similarity"] = similarity
                
                self.log_info(
                    f"Article {article.article_id} is a duplicate of {duplicate_id} "
                    f"(similarity: {similarity:.2%})"
                )
            else:
                agent_state.is_duplicate = False
                self.log_info(f"Article {article.article_id} is unique")
            
            return agent_state.model_dump()
            
        except Exception as e:
            self.log_error(f"Deduplication failed: {e}")
            state["error"] = str(e)
            return state