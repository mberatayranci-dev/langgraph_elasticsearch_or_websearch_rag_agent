"""
LangGraph RAG Sistemi - Ana Çalıştırma Dosyası
Elasticsearch dökümanları üzerinde RAG + Web Search
"""
import os
from langchain_core.messages import HumanMessage

from src.config import OPENAI_API_KEY
from src.vectorstore import VectorStoreManager
from src.graph import create_graph


def setup_environment():
    """Ortam değişkenlerini ayarlar"""
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def print_banner():
    """Başlangıç banner'ını yazdırır"""
    print("=" * 80)
    print("LangGraph RAG Sistemi - Elasticsearch Asistanı")
    print("=" * 80)
    print()


def print_instructions():
    """Kullanım talimatlarını yazdırır"""
    print(" Sistem hazır!")
    print("\n Kullanım:")
    print("   - Elasticsearch hakkında sorular sorun (dökümanlardan cevap alır)")
    print("   - Başka konular hakkında sorular sorun (web'de arama yapar)")
    print("   - Çıkmak için 'quit'veya 'çık'yazın ")
    print("=" * 80 + "\n")


def main():
    """
    Ana fonksiyon - Uygulamayı başlatır
    """
    # Ortamı hazırla
    setup_environment()
    
    # Banner'ı göster
    print_banner()
    
    # VectorStore'u kur
    print("📚 Chroma vektör veritabanı hazırlanıyor...")
    vectorstore_manager = VectorStoreManager()
    vectorstore_manager.setup(reset_db=False)
    print()
    
    # Graph'ı oluştur
    graph = create_graph(vectorstore_manager)
    
    # Talimatları göster
    print_instructions()
    
    # Ana döngü
    while True:
        try:
            # Kullanıcı girişi al
            user_input = input("Siz: ").strip()
            
            # Boş girişleri atla
            if not user_input:
                continue
            
            # Çıkış kontrolü
            if user_input.lower() in ['quit', 'exit', 'çık', 'q']:
                print("\nGörüşmek üzere!")
                break
            
            # State başlat
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "is_elasticsearch_related": False,
                "context": ""
            }
            
            # Graph'ı çalıştır
            result = graph.invoke(initial_state)
            
            # Cevabı yazdır
            assistant_message = result["messages"][-1].content
            print(f"\n Asistan: {assistant_message}\n")
            print("-" * 80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n Program sonlandırılıyor...")
            break
            
        except Exception as e:
            print(f"\n Hata oluştu: {e}\n")
            continue


if __name__ == "__main__":
    main()