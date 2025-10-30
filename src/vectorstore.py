"""
Chroma vektör veritabanı işlemleri
"""
import os
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from .config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    DOCS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_DOCS,
    EMBEDDING_MODEL
)


class VectorStoreManager:
    """Chroma vektör veritabanı yöneticisi"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = None
    
    def setup(self, reset_db: bool = False):
        """
        Vektör veritabanını kurar
        
        Args:
            reset_db: True ise mevcut DB'yi siler ve yeniden oluşturur
        """
        # Eğer reset isteniyorsa db'yi sil
        if reset_db and os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)
            print("Eski veritabanı silindi, yeni oluşturuluyor...")
        
        # Eğer Chroma DB zaten varsa, yükle
        if os.path.exists(CHROMA_DB_DIR) and not reset_db:
            print("Mevcut Chroma veritabanı yükleniyor...")
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
            return self.vectorstore
        
        # Yoksa yeni oluştur
        self._create_new_vectorstore()
        return self.vectorstore
    
    def _create_new_vectorstore(self):
        """Yeni vektör veritabanı oluşturur"""
        # Dökümanları yükle veya oluştur
        documents = self._load_or_create_documents()
        
        # Dökümanları parçala
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        
        # Chroma'ya ekle
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_DIR,
            collection_name=COLLECTION_NAME
        )
        
        print(f" {len(splits)} döküman parçası Chroma'ya eklendi.")
    
    def _load_or_create_documents(self):
        """Dökümanları yükler veya örnek döküman oluşturur"""
        try:
            # data klasörünü kontrol et, yoksa oluştur
            os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
            
            loader = TextLoader(DOCS_PATH, encoding='utf-8')
            documents = loader.load()
            print(f"Döküman yüklendi: {DOCS_PATH}")
            return documents
            
        except Exception as e:
            print(f"Döküman yüklenirken hata: {e}")
            print(" Örnek bir döküman oluşturuluyor...")
            
            # Örnek Elasticsearch dökümanı
            sample_doc = self._get_sample_document()
            
            # data klasörünü oluştur
            os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
            
            with open(DOCS_PATH, "w", encoding="utf-8") as f:
                f.write(sample_doc)
            
            loader = TextLoader(DOCS_PATH, encoding='utf-8')
            documents = loader.load()
            return documents
    
    def _get_sample_document(self) -> str:
        """Örnek Elasticsearch dökümanı döndürür"""
        return """
        Elasticsearch Nedir?
        Elasticsearch, Apache Lucene tabanlı, açık kaynaklı, dağıtık bir arama ve analiz motorudur.
        JSON tabanlı dökümanları indeksler ve gerçek zamanlı arama yapmanızı sağlar.
        
        Temel Özellikler:
        - RESTful API ile kolay entegrasyon
        - Gerçek zamanlı arama ve analiz
        - Yatay ölçeklenebilirlik (horizontal scaling)
        - Çoklu tenant desteği
        - Tam metin arama (full-text search)
        - Aggregation desteği ile güçlü analitik
        
        Kurulum:
        1. Java 8 veya üzeri kurulmalıdır
        2. Elasticsearch paketini resmi sitesinden indirin
        3. Windows: bin/elasticsearch.bat, Linux/Mac: bin/elasticsearch komutu ile başlatın
        4. Varsayılan olarak http://localhost:9200 adresinde çalışır
        
        Temel Konseptler:
        - Index: İlişkili dökümanların koleksiyonu (SQL'deki database gibi)
        - Document: JSON formatında veri (SQL'deki row gibi)
        - Field: Döküman içindeki key-value çifti (SQL'deki column gibi)
        - Mapping: Index'in şeması, field'ların tiplerini belirler
        - Shard: Index'in parçaları, dağıtık yapıyı sağlar
        - Replica: Shard'ların kopyaları, yedekleme ve okuma performansı sağlar
        
        Temel Kullanım:
        - Index oluşturma: PUT /my-index
        - Mapping tanımlama: PUT /my-index/_mapping
        - Döküman ekleme: POST /my-index/_doc veya PUT /my-index/_doc/1
        - Döküman güncelleme: POST /my-index/_update/1
        - Döküman silme: DELETE /my-index/_doc/1
        - Arama yapma: GET /my-index/_search
        - Index silme: DELETE /my-index
        
        Query DSL:
        Elasticsearch güçlü bir sorgu dili (Query DSL) sunar:
        - Match Query: Tam metin araması için
        - Term Query: Tam eşleşme araması için
        - Range Query: Aralık sorguları için (tarih, sayı)
        - Bool Query: Birden fazla sorguyu birleştirmek için (must, should, must_not)
        - Wildcard Query: Joker karakterlerle arama
        - Fuzzy Query: Benzer terimleri bulma
        
        Aggregations:
        Veri analitiği için güçlü aggregation'lar:
        - Metric Aggregations: avg, sum, min, max, stats
        - Bucket Aggregations: terms, date_histogram, range
        - Pipeline Aggregations: Aggregation sonuçları üzerinde işlem
        
        Örnek Sorgu:
```json
        GET /products/_search
        {
          "query": {
            "bool": {
              "must": [
                { "match": { "name": "laptop" } }
              ],
              "filter": [
                { "range": { "price": { "gte": 500, "lte": 1500 } } }
              ]
            }
          },
          "aggs": {
            "avg_price": {
              "avg": { "field": "price" }
            }
          }
        }
```
        
        Performans İpuçları:
        - Mapping'leri doğru tanımlayın
        - Gereksiz field'ları _source'tan çıkarın
        - Bulk API ile toplu işlemler yapın
        - Shard sayısını doğru belirleyin
        - Replica sayısını ihtiyaca göre ayarlayın
        - Filter context kullanarak sorgu cache'den faydalanın
        """
    
    def similarity_search(self, query: str, k: int = TOP_K_DOCS):
        """
        Benzerlik araması yapar
        
        Args:
            query: Arama sorgusu
            k: Döndürülecek döküman sayısı
            
        Returns:
            Benzer dökümanlar listesi
        """
        if self.vectorstore is None:
            raise ValueError("VectorStore henüz kurulmamış. setup() metodunu çağırın.")
        
        return self.vectorstore.similarity_search(query, k=k)
