"""
LangChain Tools
"""
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

from .config import MAX_SEARCH_RESULTS


@tool
def web_search(query: str) -> str:
    """
    Web'de DuckDuckGo ile arama yapar ve sonuçları döndürür.
    
    Args:
        query: Arama sorgusu
        
    Returns:
        Arama sonuçları
    """
    try:
        search = DuckDuckGoSearchResults(max_results=MAX_SEARCH_RESULTS)
        results = search.invoke(query)
        
        # Sonuçları formatla
        if isinstance(results, str):
            return results
        
        return results
        
    except Exception as e:
        return f"Web araması yapılamadı: {str(e)}\nLütfen internet bağlantınızı kontrol edin."

