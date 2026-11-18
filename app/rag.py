import ollama
from app.config import settings
from app.vectorstore import search_similar_documents
from app.tokens import count_tokens  

def generate_rag_response(question: str, source_filter: str = None) -> dict:
    """
    RF-06, RF-07, RF-08, RF-09: Esegue la ricerca, genera la risposta ancorata,
    struttura le fonti e calcola i token di prompt e risposta.
    """
    search_results = search_similar_documents(query=question, top_k=settings.TOP_K, source_filter=source_filter)
    
    documents = search_results.get("documents", [[]])[0]
    metadatas = search_results.get("metadatas", [[]])[0]
    distances = search_results.get("distances", [[]])[0]
    
    if not documents:
        return {
            "answer": "Non ho trovato alcun documento o informazione pertinente nel database per rispondere a questa domanda.",
            "sources": [],
            "usage": {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}
        }

    context_blocks = []
    sources_detail = []
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        context_blocks.append(doc)
        
        source_name = meta.get("source", "unknown") if meta else "unknown"
        chunk_idx = meta.get("chunk_index", i) if meta else i
        page_num = meta.get("page", 1) if meta else 1
        
        similarity_score = round(max(0.0, 1.0 - dist), 4)
        snippet = doc[:150].strip().replace("\n", " ") + "..."

        sources_detail.append({
            "document": source_name,
            "page": page_num,
            "chunk_index": chunk_idx,
            "similarity_score": similarity_score,
            "snippet": snippet
        })

    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "Sei docLens, un assistente RAG locale, rigoroso e professionale.\n"
        "Rispondi alla domanda dell'utente utilizzando ESCLUSIVAMENTE il contesto fornito.\n"
        "Se l'informazione non è presente, di' chiaramente che non ci sono informazioni sufficienti."
    )

    user_prompt = f"Contesto:\n{context_text}\n\nDomanda: {question}"

    # Calcolo dei token del prompt (sistema + contesto + domanda) (RF-09)
    full_prompt_text = f"{system_prompt}\n{user_prompt}"
    prompt_tokens = count_tokens(full_prompt_text)

    try:
        response = ollama.chat(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response["message"]["content"]

        # Calcolo dei token della risposta generata (RF-09)
        response_tokens = count_tokens(answer)

        return {
            "answer": answer,
            "sources": sources_detail,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens
            }
        }

    except Exception as e:
        raise RuntimeError(f"Errore durante la generazione della risposta: {str(e)}")