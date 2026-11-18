import os
import io
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader
from app.config import settings

def extract_text_from_file(file: UploadFile) -> str:
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    # 1. Validazione estensione (RF-01)
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato non supportato '{ext}'. I formati consentiti sono: {settings.ALLOWED_EXTENSIONS}"
        )

    # 2. Lettura e validazione dimensione (RF-01)
    content = file.file.read()
    file_size_mb = len(content) / (1024 * 1024)

    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File troppo grande ({file_size_mb:.2f} MB). Il limite massimo consentito è {settings.MAX_FILE_SIZE_MB} MB."
        )

    text = ""
    
    # 3. Estrazione testo in base al formato
    if ext == ".txt":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback se il file di testo ha una codifica differente
            text = content.decode("latin-1")

    elif ext == ".pdf":
        try:
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Errore durante l'elaborazione del file PDF: {str(e)}"
            )

    # 4. Controllo che il documento non sia vuoto
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Il documento è vuoto o non contiene testo leggibile (potrebbe essere una scansione)."
        )

    return text