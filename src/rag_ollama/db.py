import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Optional
from FlagEmbedding import FlagReranker


class VectorDB:
    """
    Vector database management using ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize vector database.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory path for database persistence.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 1. Initialize the embedding model
        self.embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2'
        )

        print(
            f"✅ Loaded embedding model: "
            f"all-MiniLM-L6-v2"
        )

        # 2. Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        # 3. Get an existing collection or create a new one
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

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the input text.

        Args:
            text: Text content.

        Returns:
            A list containing the embedding vector values.
        """
        return self.embedding_model.encode(text).tolist()

    def add_documents(
        self,
        documents: List[Dict]
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
            embedding = self.get_embedding(text)
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
            return 0

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents with keyword weighting.
        """

        # 1. Perform semantic search and retrieve twice the requested results
        query_embedding = self.get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            where=filter_metadata
        )

        # 2. Format results and apply keyword-based weighting
        formatted_results = []

        if results['documents']:
            query_words = query.lower().split()

            for i in range(len(results['documents'][0])):
                text = results['documents'][0][i]
                metadata = results['metadatas'][0][i]

                distance = (
                    results['distances'][0][i]
                    if results.get('distances')
                    else 1.0
                )

                # Apply a boost for keyword matches
                boost = 0
                heading = metadata.get(
                    'heading',
                    ''
                ).lower()

                for word in query_words:

                    # Give a higher boost for matches in headings
                    if word in heading:
                        boost += 0.3

                    # Give a smaller boost for matches in document text
                    if word in text.lower():
                        boost += 0.1

                # Reduce the distance based on keyword matches
                adjusted_distance = distance - boost

                formatted_results.append({
                    "text": text,
                    "metadata": metadata,
                    "id": results['ids'][0][i],
                    "distance": distance,
                    "adjusted_distance": adjusted_distance
                })

            # Sort results by adjusted distance
            formatted_results.sort(
                key=lambda x: x['adjusted_distance']
            )

            formatted_results = formatted_results[:top_k]

        return formatted_results

    def search_by_text(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Perform text-based keyword search with word-level matching.
        """

        all_docs = self.get_all_documents()

        # Remove common stopwords from the query
        stopwords = {
            'what', 'is', 'the', 'are', 'a', 'an',
            'of', 'to', 'for', 'in', 'on', 'at',
            'with', 'without', 'by', 'from', 'up',
            'down', 'off', 'over', 'under', 'about',
            'part'
        }

        query_words = set(
            query.lower().split()
        )

        keywords = [
            w
            for w in query_words
            if w not in stopwords and len(w) > 2
        ]

        if not keywords:
            return []

        matched = []

        for doc in all_docs:

            text = doc['text'].lower()

            heading = doc['metadata'].get(
                'heading',
                ''
            ).lower()

            # Check whether any keyword matches the document
            match_score = 0

            for word in keywords:

                # Give a higher weight to matches in headings
                if word in heading:
                    match_score += 2

                # Give a lower weight to matches in document text
                elif word in text:
                    match_score += 1

            if match_score > 0:
                matched.append({
                    "text": doc['text'],
                    "metadata": doc['metadata'],
                    "id": doc['id'],
                    "distance": 0.0,
                    "match_score": match_score
                })

        # Sort results by match score in descending order
        matched.sort(
            key=lambda x: x['match_score'],
            reverse=True
        )

        return matched[:top_k]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:

        # ==========================================
        # 1. Initialize the Reranker
        # ==========================================

        reranker = FlagReranker(
            'BAAI/bge-reranker-v2-m3',
            use_fp16=True
        )

        # ==========================================
        # 2. Perform Text / Keyword Search
        # ==========================================

        text_results = self.search_by_text(
            query,
            top_k=top_k
        )

        print(
            f"🔤 Text search found "
            f"{len(text_results)} results"
        )

        # ==========================================
        # 3. Perform Semantic / Vector Search
        # ==========================================

        semantic_results = self.search(
            query,
            top_k=top_k
        )

        print(
            f"🧠 Semantic search found "
            f"{len(semantic_results)} results"
        )

        # ==========================================
        # 4. Combine Search Results
        # ==========================================

        results = []

        results.extend(text_results)
        results.extend(semantic_results)

        print(
            f"📚 Combined results: "
            f"{len(results)}"
        )

        # ==========================================
        # 5. Remove Duplicate Documents
        # ==========================================

        unique_results = []
        seen_ids = set()

        for result in results:

            if result["id"] not in seen_ids:

                seen_ids.add(
                    result["id"]
                )

                unique_results.append(
                    result
                )

        results = unique_results

        print(
            f"📚 Unique results after deduplication: "
            f"{len(results)}"
        )

        # ==========================================
        # 6. Rerank the Search Results
        # ==========================================

        print(
            f"🔄 Reranking "
            f"{len(results)} documents..."
        )

        pairs = [
            [query, result["text"]]
            for result in results
        ]

        scores = reranker.compute_score(
            pairs,
            normalize=True
        )

        # ==========================================
        # 7. Attach Reranker Scores to Results
        # ==========================================

        for result, score in zip(
            results,
            scores
        ):
            result["rerank_score"] = score

        # ==========================================
        # 8. Sort Results by Reranker Score
        # ==========================================

        results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # ==========================================
        # 9. Keep Only the Top K Results
        # ==========================================

        results = results[:5]

        # ==========================================
        # 10. Print the Final Results
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
    ) -> List[Dict]:
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
    ) -> Dict:
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

    from pdf_loader import extract_lines, merge_lines
    from chunker import create_sections, flatten_sections

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

    results = db.search(
        query,
        top_k
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