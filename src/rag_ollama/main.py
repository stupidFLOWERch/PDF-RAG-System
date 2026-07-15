from .pdf_loader import extract_lines, merge_lines
from .chunker import create_sections, flatten_sections
from .db import VectorDB, process_and_store, search_pdf


def main():
    # ========== 配置 ==========
    pdf_path = "../documents/iso27001.pdf"
    db_path = "./chroma_db"
    collection_name = "iso27001"
    
    # ========== 初始化数据库 ==========
    db = VectorDB(
        collection_name=collection_name,
        persist_directory=db_path
    )
    
    # ========== 处理 PDF 并存到数据库 ==========
    documents = process_and_store(pdf_path, db)
    

def load_and_search():
    """
    只搜索，不重新处理 PDF (如果已经存过)
    """
    db = VectorDB(collection_name="iso27001", persist_directory="./chroma_db")
    
    print(f"📊 Database stats: {db.get_stats()}")
    
    while True:
        query = input("\n🔍 Enter your question (or 'quit' to exit): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        results = db.search(query, top_k=5)
        
        print(f"\n📊 Found {len(results)} results:\n")
        for i, result in enumerate(results):
            print(f"=== Result {i+1} (Score: {1 - result['distance']:.3f}) ===")
            print(f"Heading: {result['metadata'].get('heading', 'N/A')}")
            print(f"Page: {result['metadata'].get('page', 'N/A')}")
            print(f"Text: {result['text'][:300]}...")
            print()


if __name__ == "__main__":
    # 第一次运行: 处理并存储
    main()
    
    # 后续运行: 只搜索
    # load_and_search()