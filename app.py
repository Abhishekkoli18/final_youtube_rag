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

import glob
import os
import re
import tempfile
import time
import uuid

try:
    import openai
except ImportError:
    openai = None
import requests
import yt_dlp
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

    filter_ip_locations narrows the Webshare IP pool to specific countries.
    This can help because YouTube's blocking is inconsistent across the
    proxy pool - some regions get hit harder than others. Set
    WEBSHARE_PROXY_COUNTRIES in .env (comma-separated, e.g. "us,de") to
    override the default.
    """
    proxy_username = os.environ.get("WEBSHARE_PROXY_USERNAME")
    proxy_password = os.environ.get("WEBSHARE_PROXY_PASSWORD")

    if proxy_username and proxy_password:
        countries_env = os.environ.get("WEBSHARE_PROXY_COUNTRIES", "us")
        countries = [c.strip() for c in countries_env.split(",") if c.strip()]

        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=proxy_username,
                proxy_password=proxy_password,
                filter_ip_locations=countries,
            )
        )
    return YouTubeTranscriptApi()


def fetch_transcript(video_id: str, max_retries: int = 3):
    """
    Returns (transcript_text, duration_seconds, error_message).
    Handles both the old (<1.0) and new (>=1.0) youtube-transcript-api versions.

    Retries with backoff on rate-limit-shaped errors (429 / "too many"),
    since these are often transient - the proxy pool rotates IPs, so a
    retry a few seconds later frequently succeeds even when the first
    attempt gets rate-limited.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            ytt_api = get_transcript_client()
            fetched = ytt_api.fetch(video_id, languages=["en"])
            snippets = [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]

            transcript_text = " ".join(chunk["text"] for chunk in snippets)

            duration_seconds = 0
            if snippets:
                last = snippets[-1]
                duration_seconds = last["start"] + last["duration"]

            return transcript_text, duration_seconds, None

        except AttributeError:
            try:
                snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript_text = " ".join(chunk["text"] for chunk in snippets)
                duration_seconds = 0
                if snippets:
                    last = snippets[-1]
                    duration_seconds = last["start"] + last["duration"]
                return transcript_text, duration_seconds, None
            except Exception as e:
                last_error = e
                break

        except TranscriptsDisabled:
            last_error = "Captions are disabled for this video."
            break
        except NoTranscriptFound:
            last_error = "No English transcript is available for this video."
            break
        except VideoUnavailable:
            return None, 0, "This video is unavailable."
        except Exception as e:
            last_error = e
            is_rate_limited = "429" in str(e) or "too many" in str(e).lower()
            if is_rate_limited and attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))  # 1s, 2s, 4s backoff
                continue
            break

    fallback_text, duration_seconds, fallback_error = fetch_transcript_with_ytdlp(video_id)
    if fallback_text:
        return fallback_text, duration_seconds, None

    fallback_note = f" Fallback via yt-dlp also failed: {fallback_error}" if fallback_error else ""
    return None, 0, (
        f"Could not fetch transcript after {max_retries} attempt(s) ({last_error})."
        f"{fallback_note} "
        "If this is happening after deploying (works locally, fails on the server), "
        "use yt-dlp / audio transcription fallback or the WEBSHARE_PROXY_USERNAME/"
        "WEBSHARE_PROXY_PASSWORD setup in the README."
    )


def _parse_subtitle_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    lower_path = file_path.lower()
    if lower_path.endswith(".vtt"):
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
                continue
            lines.append(line)
        return " ".join(lines)

    if lower_path.endswith(".srt"):
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            lines.append(line)
        return " ".join(lines)

    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def fetch_transcript_with_ytdlp(video_id: str):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en"],
                "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                ydl.download([video_url])

            subtitle_texts = []
            for pattern in ("*.vtt", "*.srt", "*.ttml", "*.json"):
                for file_path in glob.glob(os.path.join(tmpdir, pattern)):
                    parsed = _parse_subtitle_text(file_path)
                    if parsed:
                        subtitle_texts.append(parsed)

            transcript_text = " ".join(subtitle_texts).strip()
            duration_seconds = int(info.get("duration") or 0)

            if transcript_text:
                return transcript_text, duration_seconds, None

            if os.environ.get("OPENAI_API_KEY"):
                return transcribe_audio_with_openai(video_id)

            return None, duration_seconds, "No English subtitles were found with yt-dlp and OPENAI_API_KEY is not configured for audio transcription."

    except Exception as e:
        return None, 0, f"yt-dlp fallback failed: {e}"


def transcribe_audio_with_openai(video_id: str):
    if openai is None:
        return None, 0, "OpenAI package is not installed for audio transcription."

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return None, 0, "OPENAI_API_KEY is not configured for audio transcription."

    openai.api_key = openai_api_key
    model = os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-transcribe")
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                audio_path = ydl.prepare_filename(info)

            if not os.path.exists(audio_path):
                return None, 0, "Failed to save audio for transcription."

            with open(audio_path, "rb") as audio_file:
                transcription = openai.Audio.transcribe(model, audio_file)

            transcript_text = (
                transcription.get("text")
                if isinstance(transcription, dict)
                else getattr(transcription, "text", None)
            )
            duration_seconds = int(info.get("duration") or 0)
            if transcript_text:
                return transcript_text.strip(), duration_seconds, None

            return None, duration_seconds, "Audio transcription returned empty text."

    except Exception as e:
        return None, 0, f"Audio transcription failed: {e}"


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