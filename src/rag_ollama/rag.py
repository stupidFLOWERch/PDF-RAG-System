import ollama
from .db import VectorDB


class RAG:
    """
    Retrieval-Augmented Generation (RAG) pipeline.
    Retrieves relevant documents from the vector database
    and generates answers using an LLM.
    """

    def __init__(
        self,
        db: VectorDB,
        model: str = "llama3:latest"
    ):
        """
        Initialize the RAG system.

        Args:
            db: Vector database instance.
            model: Ollama model name.
        """
        self.db = db
        self.model = model

    def ask(
        self,
        query: str,
        top_k: int = 10
    ) -> str:
        """
        Answer a user query using retrieved documents.

        Args:
            query: User question.
            top_k: Number of documents to retrieve.

        Returns:
            Generated answer.
        """

        # ==========================================
        # Step 1. Retrieve relevant documents
        # ==========================================

        results = self.db.search_hybrid(
            query,
            top_k=top_k
        )

        if not results:
            return "No relevant documents found."

        # ==========================================
        # Step 2. Build context for the LLM
        # ==========================================

        context = ""

        for i, result in enumerate(results):

            context += (
                f"[{i + 1}] {result['text']}\n\n"
            )
        # print("\n========== RETRIEVED CONTEXT ==========")
        # print(context)
        # print("========================================\n")
        # ==========================================
        # Step 3. Construct the prompt
        # ==========================================

        prompt = f"""
You are a helpful information assistant.

Answer the user's question using ONLY the information provided in the context.

Context:
{context}

Question:
{query}

Answer:
"""

        # ==========================================
        # Step 4. Generate answer using Ollama
        # ==========================================

        response = ollama.chat(

            model=self.model,

            messages=[

                {
                    "role": "system",
                    "content":
                    (
                        "Answer only based on the provided context. "
                        "If the answer cannot be found in the context, "
                        "reply: 'I cannot find this information in the document.'"
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            options={
                "temperature": 0.1
            }

        )

        return (
            response["message"]["content"]
            .strip()
        )


def rag_chat():
    """
    Start an interactive RAG chatbot session.
    """

    # Initialize vector database
    db = VectorDB(
        collection_name="iso27001",
        persist_directory="./chroma_db"
    )

    # Initialize RAG
    rag = RAG(db)

    print(
        "🤖 RAG Chatbot (type 'quit' to exit)\n"
    )

    while True:

        query = input("\n❓ You: ")

        # Exit chatbot
        if query.lower() in [
            "quit",
            "exit",
            "q"
        ]:
            break

        # Generate answer
        answer = rag.ask(query)

        print(
            f"\n🤖 Assistant: {answer}"
        )


if __name__ == "__main__":
    rag_chat()