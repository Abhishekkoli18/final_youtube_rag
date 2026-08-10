"""
YouTube RAG Assistant - Flask backend
--------------------------------------
Same RAG pipeline as the original notebook / Streamlit version, now exposed
as a small JSON API consumed by a plain HTML/CSS/JS frontend
(templates/index.html + static/js/script.js).

Pipeline (unchanged from the notebook):
    1. Indexing  -> fetch transcript, split into chunks
    2. Embedding -> CohereEmbeddings (API-based, free tier, no torch needed) + FAISS
    3. Retrieval -> similarity search, k=4
    4. Augmentation + Generation -> PromptTemplate + ChatGroq (free tier)
       wired together with the LCEL "chain" method

Run with:
    python app.py
Then open http://localhost:5000
"""

import os
import re
import uuid

import requests
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)
# Used to sign the session cookie that identifies each visitor.
# Set FLASK_SECRET_KEY in .env for a stable value; falls back to a random
# one (fine for local dev, but sessions reset every restart without it).
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

# In-memory store: one entry per browser session.
# { session_id: {"chain": ..., "title": ..., "duration": ..., "video_id": ...} }
# NOTE: this is intentionally simple (single-process, in-memory) so the RAG
# logic stays easy to follow. For production you'd move this to Redis/DB.
SESSIONS = {}

GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "embed-english-v3.0"


# ============================================================
# RAG pipeline - identical logic to the notebook / Streamlit app
# ============================================================
def extract_video_id(url_or_id: str):
    """Accepts a full YouTube URL (watch, youtu.be, embed, shorts) or a bare 11-char ID."""
    url_or_id = url_or_id.strip()

    patterns = [
        r"(?:youtube\.com\/watch\?v=)([0-9A-Za-z_-]{11})",
        r"(?:youtube\.com\/shorts\/)([0-9A-Za-z_-]{11})",
        r"(?:youtube\.com\/embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id

    return None


def get_video_title(video_id: str) -> str:
    """No API key needed - uses YouTube's public oEmbed endpoint."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("title", "Unknown title")
    except Exception:
        pass
    return "Unknown title"


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d} minutes"


def get_transcript_client():
    """
    Cloud hosts (Render, Railway, AWS, etc.) run on datacenter IPs that
    YouTube frequently blocks for transcript requests - this works fine
    locally but fails once deployed. If Webshare proxy credentials are
    set, route the request through a residential proxy to work around
    this. Falls back to a direct (unproxied) request otherwise, which is
    fine for local development.
    """
    proxy_username = os.environ.get("WEBSHARE_PROXY_USERNAME")
    proxy_password = os.environ.get("WEBSHARE_PROXY_PASSWORD")

    if proxy_username and proxy_password:
        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=proxy_username,
                proxy_password=proxy_password,
            )
        )
    return YouTubeTranscriptApi()


def fetch_transcript(video_id: str):
    """
    Returns (transcript_text, duration_seconds, error_message).
    Handles both the old (<1.0) and new (>=1.0) youtube-transcript-api versions.
    """
    try:
        try:
            ytt_api = get_transcript_client()
            fetched = ytt_api.fetch(video_id, languages=["en"])
            snippets = [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
        except AttributeError:
            snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

        transcript_text = " ".join(chunk["text"] for chunk in snippets)

        duration_seconds = 0
        if snippets:
            last = snippets[-1]
            duration_seconds = last["start"] + last["duration"]

        return transcript_text, duration_seconds, None

    except TranscriptsDisabled:
        return None, 0, "Captions are disabled for this video."
    except NoTranscriptFound:
        return None, 0, "No English transcript is available for this video."
    except VideoUnavailable:
        return None, 0, "This video is unavailable."
    except Exception as e:
        return None, 0, f"Could not fetch transcript ({e}). If this is happening after deploying (works locally, fails on the server), it's likely YouTube blocking the host's IP - see the WEBSHARE_PROXY_USERNAME/WEBSHARE_PROXY_PASSWORD setup in the README."


def build_vector_store(transcript_text: str):
    """Split -> embed -> store in FAISS. (Steps 1b/1c/1d from the notebook.)

    Embeddings are generated via Cohere's API instead of a local model, so
    no torch/sentence-transformers install (~1GB+) or RAM spike is needed -
    this is what keeps the deployed app small enough for free hosting tiers.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    embeddings = CohereEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


PROMPT = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"],
)


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_chain(vector_store):
    """Step 2/3/4 from the notebook, wired as the LCEL chain."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)
    parser = StrOutputParser()

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    return parallel_chain | PROMPT | llm | parser


# ============================================================
# Session helpers
# ============================================================
def get_session_id():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def get_session_data():
    sid = get_session_id()
    return SESSIONS.setdefault(sid, {"chain": None, "title": None, "duration": None, "video_id": None})


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load-video", methods=["POST"])
def load_video():
    data = request.get_json(force=True)
    url_input = (data or {}).get("url", "")

    if not os.environ.get("GROQ_API_KEY"):
        return jsonify({"success": False, "error": "Server has no GROQ_API_KEY configured. Add it to your .env file and restart."}), 400
    if not os.environ.get("COHERE_API_KEY"):
        return jsonify({"success": False, "error": "Server has no COHERE_API_KEY configured. Add it to your .env file and restart."}), 400

    video_id = extract_video_id(url_input)
    if not video_id:
        return jsonify({"success": False, "error": "That doesn't look like a valid YouTube URL or video ID."}), 400

    transcript_text, duration_seconds, error = fetch_transcript(video_id)
    if error:
        return jsonify({"success": False, "error": error}), 400

    vector_store = build_vector_store(transcript_text)
    chain = build_chain(vector_store)
    title = get_video_title(video_id)
    duration = format_duration(duration_seconds)

    session_data = get_session_data()
    session_data["chain"] = chain
    session_data["title"] = title
    session_data["duration"] = duration
    session_data["video_id"] = video_id

    return jsonify({"success": True, "title": title, "duration": duration})


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data or {}).get("question", "").strip()

    session_data = get_session_data()
    chain = session_data.get("chain")

    if not chain:
        return jsonify({"success": False, "error": "Load a video first."}), 400
    if not question:
        return jsonify({"success": False, "error": "Question is empty."}), 400

    try:
        answer = chain.invoke(question)
    except Exception as e:
        return jsonify({"success": False, "error": f"Something went wrong generating the answer: {e}"}), 500

    return jsonify({"success": True, "answer": answer})


@app.route("/api/clear", methods=["POST"])
def clear():
    sid = get_session_id()
    if sid in SESSIONS:
        SESSIONS[sid]["chain"] = None
        SESSIONS[sid]["title"] = None
        SESSIONS[sid]["duration"] = None
        SESSIONS[sid]["video_id"] = None
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)