"""
LangGraph workflow oluşturma
"""
from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    classify_query,
    retrieve_from_docs,
    search_web,
    generate_response,
    set_vectorstore
)
from .vectorstore import VectorStoreManager


def create_graph(vectorstore_manager: VectorStoreManager):
    """
    LangGraph workflow'unu oluşturur.
    
    Args:
        vectorstore_manager: VectorStore yöneticisi
        
    Returns:
        Derlenmiş LangGraph workflow'u
    """
    # VectorStore'u node'lara aktar
    set_vectorstore(vectorstore_manager)
    
    # Workflow oluştur
    workflow = StateGraph(AgentState)
    
    # Node'ları ekle
    workflow.add_node("classify", classify_query)
    workflow.add_node("retrieve_docs", retrieve_from_docs)
    workflow.add_node("web_search", search_web)
    workflow.add_node("generate", generate_response)
    
    # Başlangıç noktası
    workflow.set_entry_point("classify")
    
    # Conditional routing fonksiyonu
    def route_after_classify(state: AgentState) -> str:
        """
        Sınıflandırmadan sonra hangi node'a gidileceğine karar verir.
        
        Args:
            state: Mevcut agent state
            
        Returns:
            Gidilecek node'un adı
        """
        if state["is_elasticsearch_related"]:
            return "retrieve_docs"
        return "web_search"
    
    # Conditional edge ekle
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "retrieve_docs": "retrieve_docs",
            "web_search": "web_search"
        }
    )
    
    # Normal edge'leri ekle
    workflow.add_edge("retrieve_docs", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    # Workflow'u derle
    return workflow.compile()


def visualize_graph(graph, output_path: str = "graph_diagram.png"):
    """
    Graph'ı görselleştirir.
    
    Args:
        graph: LangGraph workflow'u
        output_path: Çıktı dosya yolu
    """
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"Graph görselleştirilemedi: {e}")
     