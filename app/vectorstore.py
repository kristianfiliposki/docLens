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


def search_similar_documents(query: str, top_k: int = settings.TOP_K, source_filter: str = None) -> dict:
    """
    Esegue la ricerca semantica e restituisce documenti, metadati e distanze (punteggi).
    """
    if not query or not query.strip():
        raise ValueError("La query di ricerca non può essere vuota.")

    query_embedding = get_embedding(query)
    where_clause = {"source": source_filter} if source_filter else None

    # ChromaDB restituisce dizionari contenenti lists di liste per documents, metadatas e distances
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause
    )

    return results

def list_documents() -> list[str]:
    """
    RF-10: Restituisce l'elenco dei nomi di tutti i documenti unici indicizzati in ChromaDB.
    """
    try:
        # Recupera tutti i metadati presenti nella collezione
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        sources = set()
        for meta in metadatas:
            if meta and "source" in meta:
                sources.add(meta["source"])
                
        return list(sources)
    except Exception as e:
        raise RuntimeError(f"Errore durante il recupero della lista documenti: {str(e)}")


def delete_document(source_name: str) -> int:
    """
    RF-10: Elimina un documento e tutti i suoi chunk associati da ChromaDB 
    sfruttando il filtro sui metadati. Restituisce il numero di chunk eliminati.
    """
    try:
        # Cerca tutti i chunk associati al file sorgente
        existing = collection.get(where={"source": source_name})
        chunk_ids = existing.get("ids", [])
        
        if not chunk_ids:
            return 0
            
        # Elimina fisicamente i chunk dal database vettoriale persistente
        collection.delete(where={"source": source_name})
        
        return len(chunk_ids)
    except Exception as e:
        raise RuntimeError(f"Errore durante l'eliminazione del documento '{source_name}': {str(e)}")