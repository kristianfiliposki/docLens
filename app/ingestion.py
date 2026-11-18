import os
import io
import tiktoken
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



def chunk_text_by_tokens(text: str, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> list[str]:
    """
    RF-03: Suddivide il testo in chunk basati sul numero esatto di token 
    (default 300 token, overlap 50 token) usando tiktoken.
    """
    if not text or not text.strip():
        return []

    # Inizializza il tokenizer di tiktoken
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # 1. Codifica tutto il testo in una lista di token (numeri interi)
    tokens = encoding.encode(text)
    token_length = len(tokens)

    chunks = []
    start = 0

    # 2. Scorre la lista di token con la finestra scorrevole
    while start < token_length:
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        
        # 3. Riconverte i token del blocco in stringa di testo
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Avanza tenendo conto dell'overlap configurato
        start += chunk_size - overlap

    return chunks