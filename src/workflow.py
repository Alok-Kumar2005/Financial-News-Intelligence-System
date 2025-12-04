from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.models import AgentState, NewsArticle
from src.agents.deduplication_agent import DeduplicationAgent
from src.agents.entity_extraction_agent import EntityExtractionAgent
from src.agents.stock_impact_agent import StockImpactAgent
from src.agents.storage_agent import StorageAgent
from src.vector_store import VectorStoreManager
from loguru import logger


class NewsProcessingWorkflow:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.dedup_agent = DeduplicationAgent(vector_store)
        self.entity_agent = EntityExtractionAgent()
        self.impact_agent = StockImpactAgent()
        self.storage_agent = StorageAgent(vector_store)
        self.graph = self._build_graph()
        
        logger.info("News processing workflow initialized")
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(dict)
        
        ## ADding nodes
        workflow.add_node("deduplication", self._deduplication_node)
        workflow.add_node("entity_extraction", self._entity_extraction_node)
        workflow.add_node("impact_analysis", self._impact_analysis_node)
        workflow.add_node("storage", self._storage_node)
        
        ## edges
        workflow.set_entry_point("deduplication")
        workflow.add_edge("deduplication", "entity_extraction")
        workflow.add_edge("entity_extraction", "impact_analysis")
        workflow.add_edge("impact_analysis", "storage")
        workflow.add_edge("storage", END)
        
        return workflow.compile()
    
    def _deduplication_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.dedup_agent.process(state)
    
    def _entity_extraction_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.entity_agent.process(state)
    
    def _impact_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.impact_agent.process(state)
    
    def _storage_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.storage_agent.process(state)
    
    def process_article(self, article: NewsArticle) -> AgentState:
        """ Process a single article through the workflow."""
        logger.info(f"Starting workflow for article: {article.article_id}")
        
        initial_state = AgentState(article=article)
        final_state = self.graph.invoke(initial_state.model_dump())
        result = AgentState(**final_state)
        
        if result.error:
            logger.error(f"Workflow error for {article.article_id}: {result.error}")
        else:
            logger.info(f"Workflow completed for {article.article_id}")
        
        return result
    
    def get_graph_visualization(self) -> str:
        """workflow for visuablization"""
        return """
graph TD
    A[Start] --> B[Deduplication Agent]
    B --> C[Entity Extraction Agent]
    C --> D[Stock Impact Agent]
    D --> E[Storage Agent]
    E --> F[End]
    
    style B fill:#e1f5ff
    style C fill:#fff4e1
    style D fill:#ffe1e1
    style E fill:#e1ffe1
"""

if __name__ == "__main__":
    # Example usage
    vector_store = VectorStoreManager()
    workflow = NewsProcessingWorkflow(vector_store)
    
    # sample_article = NewsArticle(
    #     article_id="art_001",
    #     title="Sample Financial News",
    #     content="The stock market saw significant gains today...",
    #     source="Financial Times",
    #     published_at="2024-06-01T10:00:00Z",
    #     url="https://financialtimes.com/sample-article"
    # )
    
    # result_state = workflow.process_article(sample_article)
    # print(result_state)

    workflow_diagram = workflow.get_graph_visualization()
    print(workflow_diagram)