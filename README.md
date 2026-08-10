# YouTube RAG Assistant (Flask + HTML/CSS/JS)

Same RAG pipeline as the notebook: paste a YouTube URL, ask questions about
the video. Backend/frontend split with Flask + plain HTML/CSS/JS, and a
**lightweight, deployment-friendly stack**: no local ML model download, no
`torch`, small enough to fit free hosting tiers.

## Stack

| Piece | Tool | Why |
|---|---|---|
| Backend | Flask | Serves the API + HTML page |
| Frontend | HTML/CSS/JS | Talks to the Flask API via `fetch()` |
| Transcript | `youtube-transcript-api` | Free, no key needed |
| Text splitting | `langchain-text-splitters` | Same as the notebook |
| **Embeddings** | **Cohere API** (`langchain-cohere`) | Free tier, API-based — no `torch`/`sentence-transformers`, keeps the app small and light on RAM |
| Vector store | FAISS (`faiss-cpu`) | Same as the notebook |
| Chat model | Groq (`langchain-groq`, Llama 3.3) | Free tier |

## Why the embeddings changed from local to API-based

Earlier versions of this project used `HuggingFaceEmbeddings`, which
downloads and runs a model locally via `sentence-transformers` + `torch`.
That's well over 1GB of dependencies and can spike RAM past what most free
hosting tiers allow (Render free = 512MB, for example). Switching to
`CohereEmbeddings` moves that computation to Cohere's API — same chunking,
same FAISS store, same retrieval logic, just a small network call instead
of a local model. This is the only functional change from earlier versions
of this project; everything else in the pipeline is identical.

## 1. Setup

```bash
cd youtube_rag_flask
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get free API keys

- **Groq** (chat model): https://console.groq.com/keys — free, no card
- **Cohere** (embeddings): https://dashboard.cohere.com/api-keys — free tier, no card

## 3. Add your keys

```bash
cp .env.example .env
# edit .env and paste both real keys in
```

## 4. Run

```bash
python app.py
```

Open **http://localhost:5000**.

## Project structure

```
youtube_rag_flask/
├── app.py                  # Flask backend: routes + RAG pipeline
├── templates/
│   └── index.html          # Page structure
├── static/
│   ├── css/style.css       # Styling
│   └── js/script.js        # Frontend logic (fetch calls to the API)
├── requirements.txt
├── Procfile                 # For Render/Railway
├── Dockerfile                # For Docker-based hosts
└── .env.example
```

## How it works (unchanged from earlier versions — see inline comments in app.py)

1. `POST /api/load-video` → `extract_video_id()` → `fetch_transcript()` →
   `build_vector_store()` (split with `RecursiveCharacterTextSplitter`,
   embed with `CohereEmbeddings`, store in `FAISS`) → `build_chain()`
   (retriever + prompt + `ChatGroq`, wired with LCEL) → chain saved to
   the session.
2. `POST /api/ask` → `chain.invoke(question)` → embeds the question via
   Cohere → retrieves top-4 matching chunks from FAISS → fills the prompt
   → sends to Groq's LLM → returns the answer.
3. `POST /api/clear` → resets the session.

## API reference

| Method | Route | Body | Returns |
|---|---|---|---|
| POST | `/api/load-video` | `{"url": "..."}` | `{"success": true, "title": "...", "duration": "..."}` |
| POST | `/api/ask` | `{"question": "..."}` | `{"success": true, "answer": "..."}` |
| POST | `/api/clear` | — | `{"success": true}` |

## Deploying for free

Since the app no longer needs `torch`/`sentence-transformers`, it fits
comfortably on lightweight free tiers:

- **Render** (render.com) — free web service tier. Connect your GitHub
  repo, build command `pip install -r requirements.txt`, start command
  `gunicorn -w 2 -b 0.0.0.0:$PORT app:app` (matches the included
  `Procfile`). Add `GROQ_API_KEY`, `COHERE_API_KEY`, `FLASK_SECRET_KEY`
  under Environment.
- **Railway** / **Fly.io** — similar flow, free/trial tiers.
- **Hugging Face Spaces** — as of mid-2026, Docker Spaces require a paid
  PRO plan ($9/mo); Static Spaces (free) can't run a Flask backend. Skip
  this option unless you're on PRO.

Render is the simplest fully-free path now that the image is small.

## Notes

- Chat state and vector index are kept **server-side in memory**, keyed by
  a session cookie. Restarting the server clears all sessions.
- If you ever paste a real API key into a chat, screenshot, or commit —
  treat it as compromised and rotate it immediately at the provider's
  dashboard, even if "it still seems to work."
