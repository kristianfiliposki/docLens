# docLens 🔍🤖

> Un assistente RAG (Retrieval-Augmented Generation) locale, modulare e orientato alla privacy, costruito con **FastAPI**, **Ollama** e **ChromaDB**.

---

## 📋 Indice

1. Panoramica e Scopo
2. Architettura e Flussi
3. Stack Tecnologico
4. Requisiti Funzionali (RF-01 a RF-11)
5. Prerequisiti e Installazione
6. Configurazione (.env)
7. Guida all'Avvio
8. Esempi di Utilizzo delle API (cURL)
9. Scelte Progettuali e Sperimentazione dei Parametri
10. Limiti Noti ed Evoluzioni Future

---

## 🎯 Panoramica e Scopo

**docLens** è un servizio web locale progettato per caricare documenti (`PDF` e `TXT`), analizzarli e interrogarli in linguaggio naturale.

L'intera pipeline opera **al 100% in locale**: i modelli di embedding e il modello linguistico (LLM) sono eseguiti tramite **Ollama**. Non sono richieste chiavi API esterne, garantendo zero costi operativi e la totale riservatezza dei dati.

---

## 🏗️ Architettura e Flussi

### 1. Flusso di Ingestion (Upload)

1. Il client invia un file tramite `POST /documents`.
2. Il sistema valida l'estensione (`.pdf`, `.txt`) e la dimensione massima (RF-01).
3. Estrae il testo grezzo preservando la struttura delle pagine (RF-02).
4. Suddivide il testo in chunk basati sul numero esatto di token con sovrapposizione (*overlap*) (RF-03).
5. Calcola gli embedding vettoriali per ogni chunk tramite Ollama (RF-04).
6. Salva permanentemente chunk, vettori e metadati in ChromaDB (RF-05, RF-08).

### 2. Flusso di Domanda (Query)

1. Il client invia la domanda tramite `POST /ask` (RF-06).
2. Viene calcolato l'embedding della query utilizzando lo stesso modello di embedding (RF-04).
3. ChromaDB recupera i `top_k` chunk più simili tramite distanza geometrica (con eventuale filtro per documento) (RF-06).
4. Viene assemblato il prompt con istruzioni di sistema rigorose anti-allucinazione (RF-07).
5. Ollama genera la risposta ancorata esclusivamente al contesto, restituendo statistiche sui token e fonti dettagliate con punteggio di similarità ed estratti (RF-08, RF-09).

---

## 🛠️ Stack Tecnologico

* **Linguaggio:** Python 3.11+
* **API Framework:** FastAPI & Uvicorn
* **Validazione:** Pydantic
* **LLM & Embeddings:** Ollama (`llama3.2` per la generazione, `nomic-embed-text` per la vettorializzazione)
* **Database Vettoriale:** ChromaDB (con client persistente su disco)
* **Parsing PDF:** `pypdf`
* **Tokenizzazione:** `tiktoken` (utilizzato come misuratore standard per il chunking)
* **Test & Configurazione:** `pytest`, `httpx`, `python-dotenv`

---

## 📋 Requisiti Funzionali (RF-01 a RF-11)

* **RF-01 (Upload):** Supporto file `.pdf` e `.txt` con limite di dimensione configurabile (default 10 MB).
* **RF-02 (Estrazione):** Estrazione testuale pagina per pagina per i PDF; blocco dei file vuoti o non leggibili.
* **RF-03 (Chunking):** Suddivisione basata su token (default 300 token, overlap 50 token).
* **RF-04 (Embeddings):** Coerenza del modello vettoriale sia in ingestion che in query (`nomic-embed-text`).
* **RF-05 (Persistenza):** Archiviazione persistente su disco (`./data/chroma`) sopravvissuta ai riavvii.
* **RF-06 (Domanda):** Recupero dei `top_k` chunk più rilevanti con filtro opzionale per documento.
* **RF-07 (Risposta Ancorata):** Prompt di sistema anti-allucinazione; obbligo di dichiarare l'assenza di informazioni.
* **RF-08 (Fonti):** Dettaglio completo per ogni fonte (documento, pagina, chunk index, score, snippet).
* **RF-09 (Token Count):** Tracciamento dei token totali all'upload e conteggio prompt/risposta alle query.
* **RF-10 (Gestione Documenti):** Endpoint dedicati per elencare ed eliminare documenti con relativi chunk.
* **RF-11 (Health Check):** Verifica attiva della raggiungibilità di Ollama e della presenza dei modelli installati.

---

## ⚙️ Prerequisiti e Installazione

1. **Installare Ollama:**
   Assicurati di avere Ollama installato e avviato sul tuo sistema.

2. **Scaricare i modelli richiesti:**
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. **Preparare l'ambiente Python:**
   ```bash
   cd docLens
   python -m venv venv
   source venv/bin/activate  # Su Windows: venv\Scripts\activate
   ```

4. **Installare le dipendenze:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔧 Configurazione (.env)

Crea un file `.env` nella root del progetto partendo da `.env.example`:

```env
PROJECT_NAME=docLens
LLM_MODEL=llama3.2
EMBED_MODEL=nomic-embed-text
CHROMA_PATH=./data/chroma
TOP_K=4
CHUNK_SIZE=300
CHUNK_OVERLAP=50
MAX_FILE_SIZE_MB=10
```

---

## 🚀 Guida all'Avvio

Avvia il server di sviluppo con Uvicorn:

```bash
uvicorn app.main:app --reload
```

Una volta avviato, puoi testare tutti gli endpoint tramite la documentazione interattiva Swagger UI disponibile all'endpoint `/docs` del server locale (porta `8000`).

---

## 📡 Esempi di Utilizzo delle API (cURL)

### 1. Health Check (`GET /health`)
Verifica che il servizio e Ollama siano operativi:
```bash
curl -X GET "http://localhost:8000/health"
```

### 2. Upload Documento (`POST /documents`)
Carica e indicizza un file PDF o TXT:
```bash
curl -X POST "http://localhost:8000/documents" \
  -F "file=@manuale.pdf"
```

### 3. Elenco Documenti (`GET /documents`)
Elenca tutti i file indicizzati nell'archivio persistente:
```bash
curl -X GET "http://localhost:8000/documents"
```

### 4. Interrogazione RAG (`POST /ask`)
Pone una domanda al sistema:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quali sono i termini di garanzia descritti nel documento?",
    "source_filter": "manuale.pdf"
  }'
```

### 5. Eliminazione Documento (`DELETE /documents/{document_name}`)
Rimuove un documento e tutti i suoi chunk associati:
```bash
curl -X DELETE "http://localhost:8000/documents/manuale.pdf"
```

---

## 📊 Scelte Progettuali e Sperimentazione dei Parametri

* **Dimensione del Chunk (300 vs 800 token):** Si è scelto un default di **300 token** perché garantisce una granularità semantica elevata, impedendo che informazioni chiave vengano diluite in contesti troppo ampi. I test mostrano che chunk da 800 token riducono la precisione della ricerca geometrica su manuali tecnici densi.
* **Sovrapposizione (Overlap):** Fissata a **50 token** per fare da ponte logico ed evitare la perdita di frammenti di frasi spezzati ai confini del chunk.
* **Top-K:** Default a **4** chunk recuperati per bilanciare la completezza informativa del contesto ed evitare il superamento dei limiti di token del prompt.

---

## ⚠️ Limiti Noti

* **Assenza di OCR:** I file PDF scansionati (immagini senza livello di testo selezionabile) non possono essere letti.
* **Tokenizzazione Approssimata:** L'uso di `tiktoken` fornisce una stima solida per il chunking ma non coincide al millimetro con il tokenizzatore nativo di Llama 3.2. I token reali di esecuzione sono comunque monitorati tramite i metadati di risposta di Ollama.
* **Assenza di Autenticazione:** Progetto pensato per uso locale/personale, privo di sistemi di login o multi-tenancy nella versione 1.0.

---

## 🔮 Evoluzioni Future

* Interfaccia grafica dedicata (Streamlit o React).
* Streaming della risposta in tempo reale.
* Memoria conversazionale multi-turno.
* Supporto per file `.docx` e pipeline OCR integrata.