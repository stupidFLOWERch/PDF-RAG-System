# main.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from pdf_loader import extract_lines, merge_lines
from chunker import create_sections, flatten_sections
from db import VectorDB, process_and_store, search_pdf


def main():
    pdf_path = r"C:\Users\User\Desktop\pdf-rag-system\documents\PlantPals with author details revised.docx (2).pdf"
    
    # 1. 处理并存储
    db = VectorDB(collection_name="documents", persist_directory="./chroma_db")
    
    elements = extract_lines(pdf_path)
    elements = merge_lines(elements)
    sections, title = create_sections(elements)
    documents = flatten_sections(sections, title)
    
    # ✅ 打印所有存入的 heading
    print("\n📊 All documents stored:")
    for i, doc in enumerate(documents):
        print(f"  {i+1}. Heading: {doc['metadata']['heading'][:50]}...")
    
    db.add_documents(documents)
    
    # 2. 向量搜索测试
    print("\n" + "=" * 70)
    print("🔍 VECTOR SEARCH for 'Abstract':")
    print("=" * 70)
    vector_results = db.search("Abstract", top_k=10)
    
    for i, r in enumerate(vector_results):
        print(f"  {i+1}. Heading: {r['metadata'].get('heading')}")
        print(f"     Text: {r['text'][:100]}...")
        print()
    
    # 3. 文本搜索测试
    print("\n" + "=" * 70)
    print("🔍 TEXT SEARCH for 'Abstract':")
    print("=" * 70)
    text_results = db.search_by_text("Abstract", top_k=10)
    
    if text_results:
        for i, r in enumerate(text_results):
            print(f"  {i+1}. Heading: {r['metadata'].get('heading')}")
            print(f"     Text: {r['text'][:100]}...")
            print()
    else:
        print("  ❌ No results found! 'Abstract' not in text or heading.")


if __name__ == "__main__":
    main()