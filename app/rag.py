import ollama
from app.config import settings
from app.vectorstore import search_similar_documents

def generate_rag_response(question: str, source_filter: str = None) -> dict:
    """
    Orchestra l'intero flusso RAG:
    1. RF-06: Recupera i top_k chunk più rilevanti da ChromaDB (con eventuale filtro documento).
    2. Unisce i chunk in un unico contesto testuale.
    3. RF-07: Invia il contesto e la domanda a llama3.2 con un prompt di sistema rigoroso anti-allucinazione.
    """
    # 1. Recupero dei chunk da ChromaDB (RF-06)
    search_results = search_similar_documents(query=question, top_k=settings.TOP_K, source_filter=source_filter)
    
    documents = search_results.get("documents", [[]])[0]
    metadatas = search_results.get("metadatas", [[]])[0]
    
    # Se il database è vuoto o non trova corrispondenze
    if not documents:
        return {
            "answer": "Non ho trovato alcun documento o informazione pertinente nel database per rispondere a questa domanda.",
            "sources": []
        }

    # 2. Assemblaggio del contesto e raccolta delle fonti uniche
    context_blocks = []
    sources = set()
    
    for doc, meta in zip(documents, metadatas):
        context_blocks.append(doc)
        if meta and "source" in meta:
            sources.add(meta["source"])
            
    context_text = "\n\n---\n\n".join(context_blocks)

    # 3. Costruzione del prompt di sistema rigoroso per la risposta ancorata (RF-07)
    system_prompt = (
        "Sei docLens, un assistente RAG locale, rigoroso e professionale.\n"
        "Il tuo compito è rispondere alla domanda dell'utente utilizzando ESCLUSIVAMENTE il contesto fornito qui sotto.\n\n"
        "REGOLE RIGIDE:\n"
        "1. Basa la tua risposta unicamente sulle informazioni presenti nel contesto.\n"
        "2. Non inventare, non estrapolare e non aggiungere informazioni esterne o supposizioni.\n"
        "3. Se la risposta non è presente nel contesto fornito, devi dichiararlo chiaramente dicendo: "
        "'Non ho trovato informazioni sufficienti nei documenti caricati per rispondere a questa domanda.'\n"
        "4. Cita la fonte o il documento di riferimento quando utile."
    )

    user_prompt = f"Contesto estratto dai documenti:\n{context_text}\n\nDomanda dell'utente: {question}"

    try:
        # 4. Interrogazione del modello locale llama3.2 tramite Ollama (RF-07)
        response = ollama.chat(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response["message"]["content"]

        return {
            "answer": answer,
            "sources": list(sources)
        }

    except Exception as e:
        raise RuntimeError(f"Errore durante la generazione della risposta con l'LLM locale: {str(e)}")