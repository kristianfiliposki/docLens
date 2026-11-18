from fastapi import FastAPI, UploadFile, File, HTTPException
from app.config import settings
from app.ingestion import extract_text_from_file

app = FastAPI(title=settings.PROJECT_NAME, version="1.0", description="Assistente RAG locale")

@app.post("/upload", summary="RF-01: Upload e validazione documento (.pdf o .txt)")
def upload_document(file: UploadFile = File(...)):
    # Estrae e valida il testo applicando i vincoli di RF-01
    text_content = extract_text_from_file(file)
    
    # Per ora restituiamo un feedback di successo con il conteggio dei caratteri estratti
    return {
        "status": "success",
        "filename": file.filename,
        "characters_extracted": len(text_content),
        "message": "Documento validato e caricato con successo."
    }