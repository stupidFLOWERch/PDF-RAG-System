from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate  # ✅ 1.0 standard
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser  # ✅ Output parser

import os
import shutil

# ==========================================
# ✅ Print chunk preview
# ==========================================
def preview_chunks(chunks, max_show=5):
    """Print preview information for the first few chunks"""
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
        # ==========================================
        # 1. Load PDF
        # ==========================================
        if pdf_path:
            if os.path.exists(persist_directory):
                print(f"🗑️ Deleting existing database: {persist_directory}")
                shutil.rmtree(persist_directory)
            
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"📄 Loaded {len(documents)} pages from PDF")

            # ==========================================
            # 2. Text splitting
            # ==========================================
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            )
            
            recursive_chunks = recursive_splitter.split_documents(documents)
            print(f"📊 Recursive split into {len(recursive_chunks)} chunks")

            # Token-level splitting (CharacterTextSplitter as fallback since TokenTextSplitter is removed in newer versions)
            from langchain_text_splitters import CharacterTextSplitter
            token_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=" "
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
            print(f"📊 Final split into {len(final_chunks)} token-aware chunks")
                        
            # ==========================================
            # 3. Store in vector database
            # ==========================================
            self.vectorstore = Chroma.from_documents(
                documents=final_chunks,
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
        
        # ==========================================
        # 4. LLM - ChatOllama for local model inference
        # ==========================================
        self.llm = ChatOllama(model="llama3:latest", temperature=0.1)
                
        # ==========================================
        # 5. Retriever - fetches relevant documents from vector store
        # ==========================================
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance - balances relevance and diversity
            search_kwargs={"k": 10}  # Return top 10 most relevant chunks
        )
        
        # ==========================================
        # 6. Prompt template
        # ==========================================
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
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
        ])
        
        # ==========================================
        # 7. LCEL Chain - the "pipeline" that connects all components
        # ==========================================
        # ✅ This is LangChain 1.0's "line" - a declarative pipeline
        # Data flows: retrieve → prompt → llm → parse
        self.qa_chain = (
            {
                "context": self.retriever,           # Step 1: Retrieve relevant docs
                "question": RunnablePassthrough()    # Step 2: Pass user question through unchanged
            }
            | self.prompt                            # Step 3: Format prompt with context + question
            | self.llm                               # Step 4: Send to LLM for generation
            | StrOutputParser()                      # Step 5: Parse LLM response to string
        )
        
        # Keep reference to retriever for displaying source documents in ask()
        self.retriever_raw = self.retriever
    
    def ask(self, query: str) -> str:
        """Query the RAG system and return the answer"""
        # Execute the chain with the user's query
        result = self.qa_chain.invoke(query)
        
        # Display retrieved source documents for transparency
        source_docs = self.retriever_raw.invoke(query)
        print(f"\n📚 Retrieved {len(source_docs)} chunks:")
        for i, doc in enumerate(source_docs[:10]):
            print(f"[{i+1}] {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}\n")
        
        return result


def rag_chat_langchain():
    """Main chat loop for the RAG chatbot"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "..", "..", "uploads", "pdf_NCVD_ACSPCI_UserManual.pdf")

    rag = RAGLangChain(
        pdf_path=pdf_path,
        persist_directory="./chroma_db",
        collection_name="plant-hunt-info",
        chunk_size=700,
        chunk_overlap=120
    )
    
    print("🤖 LangChain 1.0 RAG Chatbot (type 'quit' to exit)\n")
    
    while True:
        query = input("\n❓ You: ")
        if query.lower() in ["quit", "exit", "q"]:
            break
        answer = rag.ask(query)
        print(f"\n🤖 Assistant: {answer}")


if __name__ == "__main__":
    rag_chat_langchain()