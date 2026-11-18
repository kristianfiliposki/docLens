import ollama
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from app.config import settings
from app.schemas import QueryRequest
from app.ingestion import ingest_file
from app.vectorstore import list_documents, delete_document
from app.rag import generate_rag_response

# Inizializzazione dell'app FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Assistente RAG locale basato su FastAPI, Ollama (llama3.2) e ChromaDB.",
    version="1.0.0"
)

@app.get("/health", tags=["System"])
def health_check():
    """
    RF-11: Verifica lo stato del servizio, la connessione a Ollama 
    e la presenza dei modelli richiesti (LLM e Embedding).
    """
    try:
        # Interroga Ollama per ottenere la lista dei modelli locali installati
        response = ollama.list()
        # Estrae i nomi dei modelli disponibili
        available_models = [m.get("model", m.get("name")) for m in response.get("models", [])]
        
        # Verifica la presenza dei modelli configurati in settings
        llm_present = any(settings.LLM_MODEL in m for m in available_models)
        embed_present = any(settings.EMBED_MODEL in m for m in available_models)
        
        if not (llm_present and embed_present):
            return {
                "status": "degraded",
                "ollama_connected": True,
                "models_installed": available_models,
                "warning": f"Modelli richiesti non trovati tutti. Richiesti: {settings.LLM_MODEL}, {settings.EMBED_MODEL}"
            }
            
        return {
            "status": "healthy",
            "ollama_connected": True,
            "models_installed": available_models
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama non è raggiungibile o non è attivo localmente. Dettaglio: {str(e)}"
        )


@app.post("/documents", status_code=status.HTTP_201_CREATED, tags=["Documents"])
def upload_document(file: UploadFile = File(...)):
    """
    Carica, valida (RF-01), estrae (RF-02), indicizza tramite chunking a token (RF-03)
    e salva permanentemente su ChromaDB (RF-05) un documento PDF o TXT.
    Restituisce i token totali del documento (RF-09).
    """
    try:
        result = ingest_file(file)
        return {
            "message": "Documento caricato e indicizzato con successo.",
            "data": result
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore interno durante l'indicizzazione del file: {str(e)}"
        )


@app.get("/documents", tags=["Documents"])
def get_documents():
    """
    RF-10: Elenca tutti i documenti unici attualmente indicizzati in ChromaDB.
    """
    try:
        docs = list_documents()
        return {
            "count": len(docs),
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/documents/{document_name}", tags=["Documents"])
def remove_document(document_name: str):
    """
    RF-10: Elimina un documento specifico e tutti i suoi chunk associati da ChromaDB.
    """
    try:
        deleted_count = delete_document(document_name)
        if deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Nessun documento trovato con il nome '{document_name}'."
            )
        return {
            "message": f"Documento '{document_name}' eliminato con successo.",
            "chunks_deleted": deleted_count
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/ask", tags=["RAG Query"])
def ask_question(body: QueryRequest):
    """
    RF-06, RF-07, RF-08, RF-09: Interroga il sistema RAG. 
    Esegue la ricerca dei chunk più rilevanti, genera una risposta ancorata senza allucinazioni, 
    restituisce il dettaglio completo delle fonti e il conteggio dei token consumati.
    """
    try:
        result = generate_rag_response(question=body.question, source_filter=body.source_filter)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore durante l'elaborazione della query: {str(e)}"
        )