import chromadb
from app.config import settings
from app.embeddings import get_embedding

# RF-05: Inizializzazione del client persistente di ChromaDB.
# I dati vengono scritti e letti direttamente dalla cartella fisica su disco (es. ./data/chroma).
# Questo garantisce che i documenti sopravvivano al riavvio del servizio FastAPI.
client = chromadb.PersistentClient(path=settings.CHROMA_PATH)

# Recupera la collezione esistente (se già presente su disco) o ne crea una nuova.
collection = client.get_or_create_collection(name="doclens_documents")


def add_document_chunk(doc_id: str, text: str, source: str):
    """
    Genera l'embedding del chunk e lo salva permanentemente in ChromaDB su disco.
    """
    if not text or not text.strip():
        return

    embedding_vector = get_embedding(text)

    collection.add(
        ids=[doc_id],
        embeddings=[embedding_vector],
        documents=[text],
        metadatas=[{"source": source}]
    )


def search_similar_documents(query: str, top_k: int = settings.TOP_K) -> dict:
    """
    Interroga il database persistente per trovare i chunk più vicini semanticamente alla query.
    """
    if not query or not query.strip():
        raise ValueError("La query di ricerca non può essere vuota.")

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results