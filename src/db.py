# db.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Optional


class VectorDB:
    """
    向量数据库管理类 - 使用 ChromaDB
    """
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """
        初始化向量数据库
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 1. 初始化 embedding 模型
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✅ Loaded embedding model: all-MiniLM-L6-v2")
        
        # 2. 初始化 ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 3. 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        print(f"✅ Connected to ChromaDB: {persist_directory}")
        print(f"✅ Collection: {collection_name}, existing docs: {self.collection.count()}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        生成文本的 embedding
        
        Args:
            text: 文本内容
            
        Returns:
            embedding 向量
        """
        return self.embedding_model.encode(text).tolist()
    
    def add_documents(self, documents: List[Dict]) -> int:
        """
        批量添加文档到向量数据库
        
        Args:
            documents: 文档列表，每个文档包含 text 和 metadata
            
        Returns:
            添加的文档数量
        """
        if not documents:
            print("⚠️ No documents to add")
            return 0
        
        ids = []
        texts = []
        metadatas = []
        embeddings = []
        
        for i, doc in enumerate(documents):
            # 生成唯一 ID
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            # 提取文本
            text = doc.get("text", "")
            texts.append(text)
            
            # 提取 metadata
            metadata = doc.get("metadata", {})
            metadatas.append(metadata)
            
            # 生成 embedding
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        try:
            # 批量添加到 ChromaDB
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            print(f"✅ Added {len(documents)} documents to collection '{self.collection_name}'")
            print(f"📊 Total documents in collection: {self.collection.count()}")
            return len(documents)
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return 0
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        搜索相似的文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_metadata: 过滤条件，如 {"heading": "Introduction"}
            
        Returns:
            搜索结果列表
        """
        # 生成查询的 embedding
        query_embedding = self.get_embedding(query)
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        # 格式化结果
        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "id": results['ids'][0][i],
                    "distance": results['distances'][0][i] if results.get('distances') else None
                })
        
        return formatted_results
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        用文本搜索 (更友好的接口)
        """
        return self.search(query, top_k)
    
    def get_all_documents(self) -> List[Dict]:
        """
        获取所有文档
        """
        results = self.collection.get()
        
        documents = []
        for i in range(len(results['ids'])):
            documents.append({
                "id": results['ids'][i],
                "text": results['documents'][i],
                "metadata": results['metadatas'][i]
            })
        
        return documents
    
    def delete_collection(self):
        """删除整个 collection"""
        self.client.delete_collection(self.collection_name)
        print(f"🗑️ Deleted collection: {self.collection_name}")
    
    def get_stats(self) -> Dict:
        """获取数据库统计信息"""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "persist_directory": self.persist_directory
        }


def process_and_store(pdf_path: str, db: VectorDB):
    """
    处理 PDF 并存储到向量数据库
    
    Args:
        pdf_path: PDF 文件路径
        db: VectorDB 实例
    """
    from pdf_loader import extract_lines, merge_lines
    from chunker import create_sections, flatten_sections
    
    print(f"📄 Processing: {pdf_path}")
    
    # 1. 提取 PDF
    elements = extract_lines(pdf_path)
    elements = merge_lines(elements)
    
    # 2. 创建 sections
    sections, title = create_sections(elements)
    print(f"📊 Created {len(sections)} sections")
    
    # 3. 展平为 chunks
    documents = flatten_sections(sections, title)
    print(f"📊 Created {len(documents)} chunks")
    
    # 4. 存入向量数据库
    count = db.add_documents(documents)
    
    print(f"✅ Stored {count} documents in vector DB")
    return documents


def search_pdf(db: VectorDB, query: str, top_k: int = 5):
    """
    搜索 PDF 内容
    
    Args:
        db: VectorDB 实例
        query: 查询文本
        top_k: 返回结果数量
    """
    results = db.search(query, top_k)
    
    print(f"\n🔍 Query: {query}")
    print(f"📊 Found {len(results)} results:\n")
    
    for i, result in enumerate(results):
        print(f"=== Result {i+1} (Distance: {result['distance']:.4f}) ===")
        print(f"Heading: {result['metadata'].get('heading', 'N/A')}")
        print(f"Page: {result['metadata'].get('page', 'N/A')}")
        print(f"Text: {result['text'][:200]}...")
        print()
    
    return results