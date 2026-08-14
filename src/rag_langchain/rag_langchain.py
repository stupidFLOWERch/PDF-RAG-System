from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate 
from langchain_community.llms import Ollama 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # ✅ 添加导入

from token_splitter import TokenTextSplitter

import os
import shutil

# ==========================================
# ✅ 打印 chunk 预览
# ==========================================
def preview_chunks(chunks, max_show=5):
    """打印前几个 chunk 的预览信息"""
    print(f"\n{'='*60}")
    print(f"📄 Chunk Preview (showing {min(max_show, len(chunks))} of {len(chunks)}):")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks[:max_show]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"  Page: {chunk.metadata.get('page', 'N/A')}")
        print(f"  Source: {chunk.metadata.get('source', 'N/A')}")
        print(f"  Length: {len(chunk.page_content)} chars, ~{len(chunk.page_content.split())} words")
        print(f"  Preview: {chunk.page_content[:200].replace(chr(10), ' ')}...")
        print(f"  Metadata: {chunk.metadata}")
    print(f"{'='*60}\n")

class RAGLangChain:
    def __init__(
        self,
        pdf_path: str = None,
        persist_directory: str = "./chroma_db",
        collection_name: str = "plant-hunt-info",
        chunk_size: int = 700,
        chunk_overlap: int = 120
    ):
        if pdf_path:
            if os.path.exists(persist_directory):
                print(f"🗑️ Deleting existing database: {persist_directory}")
                shutil.rmtree(persist_directory)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"📄 Loaded {len(documents)} pages from PDF")

            # ==========================================
            # 第一步：用 RecursiveCharacterTextSplitter 按标点切分
            # ==========================================
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            )
            
            recursive_chunks = recursive_splitter.split_documents(documents)
            print(f"📊 Recursive split into {len(recursive_chunks)} chunks")

            # ==========================================
            # 第二步：用 TokenTextSplitter 精确控制 Token 数
            # ==========================================
            token_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            final_chunks = []
            for doc in recursive_chunks:
                sub_texts = token_splitter.split_text(doc.page_content)
                for sub_text in sub_texts:
                    final_chunks.append(Document(
                        page_content=sub_text,
                        metadata=doc.metadata
                    ))
            
            preview_chunks(final_chunks, max_show=20)

            # ✅ 修复：用 final_chunks
            print(f"📊 Final split into {len(final_chunks)} token-aware chunks")
                        
            # ==========================================
            # 存入向量库
            # ==========================================
            self.vectorstore = Chroma.from_documents(
                documents=final_chunks,  # ✅ 修复：用 final_chunks
                embedding=HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                ),
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            print("✅ Vector store created successfully!")
        else:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                ),
                collection_name=collection_name
            )
            print(f"📂 Loaded existing vector store: {collection_name}")
        
        self.llm = Ollama(model="llama3:latest", temperature=0.1)
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10}
        )
        
        prompt_template = """
        You are a helpful assistant. Answer the question based on the provided context.

        If the context contains relevant information, provide a clear and comprehensive answer.
        If the question cannot be answered from the context, say: 
        "Based on the provided context, I cannot find this information."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def ask(self, query: str) -> str:
        result = self.qa_chain.invoke({"query": query})
        print(f"\n📚 Retrieved {len(result['source_documents'])} chunks:")
        for i, doc in enumerate(result['source_documents'][:10]):
            print(f"  [{i+1}] {doc.page_content[:200]}...")
            print(f"      Metadata: {doc.metadata}\n")
        return result["result"]


def rag_chat_langchain():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "..", "..", "uploads", "pdf_NCVD_ACSPCI_UserManual.pdf")

    rag = RAGLangChain(
        pdf_path=pdf_path,
        persist_directory="./chroma_db",
        collection_name="plant-hunt-info",
        chunk_size=700,
        chunk_overlap=120
    )
    
    print("🤖 LangChain RAG Chatbot (type 'quit' to exit)\n")
    
    while True:
        query = input("\n❓ You: ")
        if query.lower() in ["quit", "exit", "q"]:
            break
        answer = rag.ask(query)
        print(f"\n🤖 Assistant: {answer}")


if __name__ == "__main__":
    rag_chat_langchain()