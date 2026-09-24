import re
import uuid

import chromadb
from chromadb.config import Settings
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    Vector database management using ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 1. Embedding model
        self.embedding_model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

        print(
            "✅ Loaded embedding model: "
            "BAAI/bge-base-en-v1.5"
        )

        # 2. ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        # 3. Collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(
            f"✅ Connected to ChromaDB: "
            f"{persist_directory}"
        )

        print(
            f"✅ Collection: {collection_name}, "
            f"existing docs: {self.collection.count()}"
        )

        # 4. Load reranker once
        self.reranker = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=True
        )

        print(
            "✅ Loaded reranker: "
            "BAAI/bge-reranker-v2-m3"
        )

    def get_embedding(
            self,
            text: str,
            is_query: bool = False
        ) -> list[float]:
        """
        Generate an embedding vector for the input text.

        Args:
            text: Text content.

        Returns:
            A list containing the embedding vector values.
        """
        if is_query:
            text = "Represent this sentence for searching relevant passages: " + text
        return self.embedding_model.encode(text).tolist()

    def add_documents(
        self,
        documents: list[dict]
    ) -> int:
        """
        Add multiple documents to the vector database.

        Args:
            documents:
                List of documents.
                Each document contains:
                - text
                - metadata

        Returns:
            Number of successfully added documents.
        """
        if not documents:
            print("⚠️ No documents to add")
            return 0

        ids = []
        texts = []
        metadatas = []
        embeddings = []

        for i, doc in enumerate(documents):

            # Generate a unique document ID
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)

            # Extract document text
            text = doc.get("text", "")
            texts.append(text)

            # Extract document metadata
            metadata = doc.get("metadata", {})
            metadatas.append(metadata)

            # Generate the embedding vector
            embedding = self.get_embedding(text, is_query=False)
            embeddings.append(embedding)

        try:
            # Insert documents into ChromaDB
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

            print(
                f"✅ Added {len(documents)} documents "
                f"to collection '{self.collection_name}'"
            )

            print(
                f"📊 Total documents in collection: "
                f"{self.collection.count()}"
            )

            return len(documents)

        except Exception as e: 
            print(
                f"❌ Error adding documents: {e}"
            )
            raise

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_metadata: dict | None = None
    ) -> list[dict]:
        """
        Perform pure semantic/vector search.

        The results are retrieved using the same
        BAAI/bge-base-en-v1.5 embedding model used
        during document indexing.
        """

        query_embedding = self.get_embedding(query, is_query=True)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )

        formatted_results = []

        if results["documents"]:
            for i in range(
                len(results["documents"][0])
            ):

                text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]

                distance = (
                    results["distances"][0][i]
                    if results.get("distances")
                    else 1.0
                )

                formatted_results.append({
                    "text": text,
                    "metadata": metadata,
                    "id": results["ids"][0][i],
                    "distance": distance
                })

        return formatted_results

    def search_by_text(
        self,
        query: str,
        top_k: int = 20
    ) -> list[dict]:
        """
        Perform text-based keyword search.

        Uses normalized word matching instead of
        naive substring matching.
        """

        all_docs = self.get_all_documents()

        # Common English stopwords
        stopwords = {
            "what", "is", "the", "are", "a", "an",
            "of", "to", "for", "in", "on", "at",
            "with", "without", "by", "from", "up",
            "down", "off", "over", "under", "about",
            "part", "and", "or", "between",
            "does", "do", "how", "why", "which",
            "that", "this", "these", "those",
            "it", "its", "be", "as", "into"
        }

        # Normalize punctuation
        normalized_query = query.lower()

        # Keep technical identifiers such as:
        # 27001:2022
        # 27002:2022
        # ISO/IEC
        # 6.1.3
        query_words = re.findall(
            r"\b[\w/.-]+(?::[\w.-]+)?\b",
            normalized_query
        )

        keywords = [
            word
            for word in query_words
            if word not in stopwords
            and len(word) > 2
        ]

        if not keywords:
            return []

        matched = []

        for doc in all_docs:

            text = doc["text"].lower()

            heading = doc["metadata"].get(
                "heading",
                ""
            ).lower()

            # Tokenize document
            text_words = set(
                re.findall(
                    r"\b[\w/.-]+(?::[\w.-]+)?\b",
                    text
                )
            )

            heading_words = set(
                re.findall(
                    r"\b[\w/.-]+(?::[\w.-]+)?\b",
                    heading
                )
            )

            match_score = 0

            matched_keywords = []

            for word in keywords:

                # Exact heading match
                if word in heading_words:
                    match_score += 3
                    matched_keywords.append(word)

                # Exact text match
                elif word in text_words:
                    match_score += 1
                    matched_keywords.append(word)

            if match_score > 0:

                matched.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "id": doc["id"],
                    "distance": None,
                    "match_score": match_score,
                    "matched_keywords": matched_keywords
                })

        # Highest keyword score first
        matched.sort(
            key=lambda x: x["match_score"],
            reverse=True
        )

        return matched[:top_k]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20
    ) -> list[dict]:
        """
        Hybrid retrieval using:

        1. Keyword search
        2. Semantic/vector search
        3. Content deduplication
        4. BGE reranking
        5. Final top-k results
        """

        # ==========================================
        # 1. Keyword Search
        # ==========================================

        text_results = self.search_by_text(
            query,
            top_k=candidate_k
        )

        print(
            f"🔤 Text search found "
            f"{len(text_results)} results"
        )

        # ==========================================
        # 2. Semantic Search
        # ==========================================

        semantic_results = self.search(
            query,
            top_k=candidate_k
        )

        print(
            f"🧠 Semantic search found "
            f"{len(semantic_results)} results"
        )

        # ✅ DEBUG: 看含 15,000 的 chunk 在哪
        TARGET = "15,000"

        print("\n" + "=" * 60)
        print("🔍 TARGET CHUNK TRACKING")
        print("=" * 60)

        print(f"\n[Keyword search] {len(text_results)} results")
        for i, r in enumerate(text_results):
            if TARGET in r["text"]:
                print(f"  ✅ Found at keyword rank {i+1}")
                print(f"     match_score: {r.get('match_score')}")
                print(f"     matched_keywords: {r.get('matched_keywords')}")
                break
        else:
            print("  ❌ NOT in keyword results")

        print(f"\n[Semantic search] {len(semantic_results)} results")
        for i, r in enumerate(semantic_results):
            if TARGET in r["text"]:
                print(f"  ✅ Found at semantic rank {i+1}")
                print(f"     distance: {r.get('distance'):.4f}")
                break
        else:
            print("  ❌ NOT in semantic results")
            
        # ==========================================
        # 3. Combine
        # ==========================================

        results = (
            text_results +
            semantic_results
        )

        print(
            f"📚 Combined results: "
            f"{len(results)}"
        )

        # ==========================================
        # 4. Deduplicate by normalized content
        # ==========================================

        unique_results = []
        seen_texts = set()

        for result in results:

            normalized_text = " ".join(
                result["text"]
                .lower()
                .split()
            )

            if normalized_text not in seen_texts:

                seen_texts.add(
                    normalized_text
                )

                unique_results.append(result)

        results = unique_results

        print(
            f"📚 Unique results after "
            f"content deduplication: "
            f"{len(results)}"
        )

        # ==========================================
        # 5. Rerank
        # ==========================================

        if not results:
            return []

        print(
            f"🔄 Reranking "
            f"{len(results)} documents..."
        )

        pairs = [
            [query, result["text"]]
            for result in results
        ]

        scores = self.reranker.compute_score(
            pairs,
            normalize=True
        )

        # ==========================================
        # 6. Attach reranker scores
        # ==========================================

        for result, score in zip(
            results,
            scores
        ):
            result["rerank_score"] = score

        # ==========================================
        # 7. Sort by reranker score
        # ==========================================

        results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # ==========================================
        # 8. Final top-k
        # ==========================================

        results = results[:top_k]

        # ==========================================
        # 9. Print final results
        # ==========================================

        print(
            f"\n🎯 FINAL TOP {len(results)} RESULTS"
        )

        for i, result in enumerate(results):

            print(
                f"\n--- Result {i + 1} ---"
            )

            print(
                f"Rerank Score: "
                f"{result['rerank_score']:.4f}"
            )

            print(
                f"Heading: "
                f"{result['metadata'].get('heading', 'N/A')}"
            )

            print(
                f"Page: "
                f"{result['metadata'].get('page', 'N/A')}"
            )

            print(
                f"Text:\n"
                f"{result['text']}"
            )

        return results

    def get_all_documents(
        self
    ) -> list[dict]:
        """
        Retrieve all documents stored in the collection.
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
        """
        Delete the entire collection.
        """

        self.client.delete_collection(
            self.collection_name
        )

        print(
            f"🗑️ Deleted collection: "
            f"{self.collection_name}"
        )

    def get_stats(
        self
    ) -> dict:
        """
        Return database statistics.
        """

        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "persist_directory": self.persist_directory
        }

    def clear(self):

        # Delete the existing collection
        self.client.delete_collection(
            self.collection_name
        )

        # Create a new empty collection
        self.collection = self.client.create_collection(
            self.collection_name
        )


def process_and_store(
    pdf_path: str,
    db: VectorDB
):
    """
    Process a PDF file and store its content
    in the vector database.

    Args:
        pdf_path:
            Path to the PDF file.

        db:
            VectorDB instance.
    """

    from chunker import create_sections, flatten_sections
    from pdf_loader import extract_lines, merge_lines

    print(
        f"📄 Processing: {pdf_path}"
    )

    # 1. Extract text from the PDF
    elements = extract_lines(
        pdf_path
    )

    elements = merge_lines(
        elements
    )

    # 2. Create document sections
    sections, title = create_sections(
        elements
    )

    print(
        f"📊 Created {len(sections)} sections"
    )

    # 3. Convert sections into chunks
    documents = flatten_sections(
        sections,
        title
    )

    print(
        f"📊 Created {len(documents)} chunks"
    )

    # 4. Store chunks in the vector database
    count = db.add_documents(
        documents
    )

    print(
        f"✅ Stored {count} documents "
        f"in vector DB"
    )

    return documents


def search_pdf(
    db: VectorDB,
    query: str,
    top_k: int = 5
):
    """
    Search PDF content.

    Args:
        db:
            VectorDB instance.

        query:
            User query.

        top_k:
            Number of results to return.
    """

    results = db.search_hybrid(
        query,
        top_k=top_k
    )

    print(
        f"\n🔍 Query: {query}"
    )

    print(
        f"📊 Found {len(results)} results:\n"
    )

    for i, result in enumerate(results):

        print(
            f"=== Result {i + 1} "
            f"(Distance: {result['distance']:.4f}) ==="
        )

        print(
            f"Heading: "
            f"{result['metadata'].get('heading', 'N/A')}"
        )

        print(
            f"Page: "
            f"{result['metadata'].get('page', 'N/A')}"
        )

        print(
            f"Text: "
            f"{result['text'][:200]}..."
        )

        print()

    return results