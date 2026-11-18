from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="La domanda da porre al sistema RAG")
    source_filter: Optional[str] = Field(None, description="Nome del documento specifico su cui limitare la ricerca (opzionale)")