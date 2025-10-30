"""
LangGraph Node fonksiyonları
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from .state import AgentState
from .tools import web_search
from .config import LLM_MODEL, LLM_TEMPERATURE, DEBUG


# LLM başlatma
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

# VectorStore global değişken (graph.py'den set edilecek)
vectorstore_manager = None


def set_vectorstore(vs_manager):
    """VectorStore manager'ı set eder"""
    global vectorstore_manager
    vectorstore_manager = vs_manager


def classify_query(state: AgentState) -> AgentState:
    """
    Sorunun Elasticsearch ile ilgili olup olmadığını belirler.
    
    Args:
        state: Mevcut agent state
        
    Returns:
        Güncellenmiş state
    """
    last_message = state["messages"][-1].content
    
    classification_prompt = f"""
    Aşağıdaki soru Elasticsearch ile ilgili mi? 
    Elasticsearch hakkında kurulum, kullanım, özellikler, sorgular, aggregation gibi konulardan bahsediyorsa EVET.
    Başka bir konu hakkındaysa HAYIR cevabı ver.
    
    Sadece 'EVET' veya 'HAYIR' cevabı ver, başka bir şey yazma.
    
    Soru: {last_message}
    
    Cevap:"""
    
    response = llm.invoke(classification_prompt)
    is_es_related = "EVET" in response.content.upper()
    
    state["is_elasticsearch_related"] = is_es_related
    
    # Debug için
    if DEBUG:
        print(f"🔍 Soru sınıflandırması: {'Elasticsearch' if is_es_related else 'Genel'}")
    
    return state


def retrieve_from_docs(state: AgentState) -> AgentState:
    """
    Elasticsearch dökümanlarından ilgili bilgileri alır.
    
    Args:
        state: Mevcut agent state
        
    Returns:
        Güncellenmiş state
    """
    if not state["is_elasticsearch_related"]:
        return state
    
    last_message = state["messages"][-1].content
    
    if DEBUG:
        print("📚 Dökümanlardan bilgi alınıyor...")
    
    # VectorStore'dan ilgili dökümanları al
    docs = vectorstore_manager.similarity_search(last_message)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    state["context"] = context
    return state


def search_web(state: AgentState) -> AgentState:
    """
    DuckDuckGo ile web'de arama yapar.
    
    Args:
        state: Mevcut agent state
        
    Returns:
        Güncellenmiş state
    """
    if state["is_elasticsearch_related"]:
        return state
    
    last_message = state["messages"][-1].content
    
    if DEBUG:
        print("🌐 Web'de arama yapılıyor...")
    
    search_results = web_search.invoke(last_message)
    
    state["context"] = search_results
    return state


def generate_response(state: AgentState) -> AgentState:
    """
    Context'e dayanarak cevap üretir.
    
    Args:
        state: Mevcut agent state
        
    Returns:
        Güncellenmiş state
    """
    last_message = state["messages"][-1].content
    context = state.get("context", "")
    
    if state["is_elasticsearch_related"]:
        prompt = f"""
        Sen Elasticsearch konusunda uzman bir asistansın. Aşağıdaki döküman bilgilerine dayanarak soruyu cevapla.
        
        Döküman Bilgileri:
        {context}
        
        Kullanıcı Sorusu: {last_message}
        
        Lütfen:
        - Türkçe ve detaylı bir cevap ver
        - Döküman bilgilerine sadık kal
        - Örnekler ver
        - Teknik terimleri açıkla
        """
    else:
        prompt = f"""
        Web araması sonuçlarına dayanarak aşağıdaki soruyu cevapla.
        
        Web Arama Sonuçları:
        {context}
        
        Kullanıcı Sorusu: {last_message}
        
        Lütfen:
        - Türkçe ve öz bir cevap ver
        - Güvenilir bilgiler sun
        - Gerekirse kaynakları belirt
        """
    
    if DEBUG:
        print("🤖 Cevap üretiliyor...")
    
    response = llm.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    
    return state