import ollama
from app.config import settings

def get_embedding(text: str) -> list[float]:
    """
    RF-04: Genera l'embedding di un testo (sia esso un chunk di un documento o una query)
    utilizzando rigorosamente il modello di embedding configurato (settings.EMBED_MODEL).
    """
    if not text or not text.strip():
        raise ValueError("Il testo da convertire in embedding non può essere vuoto.")

    try:
        # Invia la richiesta al demone locale di Ollama usando il modello nomic-embed-text
        response = ollama.embeddings(
            model=settings.EMBED_MODEL,
            prompt=text
        )
        
        vector = response.get("embedding")
        
        if not vector:
            raise RuntimeError("La risposta di Ollama non contiene alcun vettore di embedding.")
            
        return vector

    except Exception as e:
        raise RuntimeError(
            f"Errore durante la generazione dell'embedding con il modello '{settings.EMBED_MODEL}'. "
            f"Verifica che Ollama sia in esecuzione localmente. Dettaglio: {str(e)}"
        )