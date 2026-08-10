// ============================================================
// YouTube RAG Assistant - frontend logic
// Talks to the Flask backend via three JSON endpoints:
//   POST /api/load-video  { url }       -> { success, title, duration, error }
//   POST /api/ask         { question }  -> { success, answer, error }
//   POST /api/clear                      -> { success }
// ============================================================

const videoUrlInput = document.getElementById("video-url");
const loadBtn = document.getElementById("load-btn");
const loadBtnText = document.getElementById("load-btn-text");
const loadSpinner = document.getElementById("load-spinner");
const statusBox = document.getElementById("status-box");

const messagesEl = document.getElementById("messages");
const emptyState = document.getElementById("empty-state");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const tipButtons = document.querySelectorAll(".tip-btn");

let videoLoaded = false;

// ---------- Helpers ----------
function setStatus(message, type) {
  statusBox.className = `status-box ${type}`;
  statusBox.innerHTML = message;
  statusBox.classList.remove("hidden");
}

function setChatEnabled(enabled) {
  videoLoaded = enabled;
  chatInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
  tipButtons.forEach((btn) => (btn.disabled = !enabled));
}

function addMessage(role, text) {
  emptyState.classList.add("hidden");

  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "user" ? "🧑" : "🤖";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  return bubble;
}

function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row bot";
  row.id = "typing-row";

  const avatar = document.createElement("div");
  avatar.className = "avatar bot";
  avatar.textContent = "🤖";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTypingIndicator() {
  const row = document.getElementById("typing-row");
  if (row) row.remove();
}

// ---------- Load video ----------
async function loadVideo() {
  const url = videoUrlInput.value.trim();
  if (!url) {
    setStatus("Please paste a YouTube URL first.", "error");
    return;
  }

  loadBtn.disabled = true;
  loadBtnText.textContent = "Loading...";
  loadSpinner.classList.remove("hidden");
  statusBox.classList.add("hidden");

  try {
    const res = await fetch("/api/load-video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();

    if (data.success) {
      setStatus(
        `<strong>Video loaded successfully!</strong><br>Title: ${data.title}<br>Duration: ${data.duration}`,
        "success"
      );
      messagesEl.innerHTML = "";
      setChatEnabled(true);
    } else {
      setStatus(data.error || "Something went wrong.", "error");
      setChatEnabled(false);
    }
  } catch (err) {
    setStatus("Could not reach the server. Is app.py running?", "error");
  } finally {
    loadBtn.disabled = false;
    loadBtnText.textContent = "Load Video";
    loadSpinner.classList.add("hidden");
  }
}

// ---------- Ask a question ----------
async function askQuestion(question) {
  addMessage("user", question);
  addTypingIndicator();
  chatInput.disabled = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    removeTypingIndicator();
    if (data.success) {
      addMessage("bot", data.answer);
    } else {
      addMessage("bot", `⚠️ ${data.error || "Something went wrong."}`);
    }
  } catch (err) {
    removeTypingIndicator();
    addMessage("bot", "⚠️ Could not reach the server.");
  } finally {
    chatInput.disabled = !videoLoaded;
    sendBtn.disabled = !videoLoaded;
    chatInput.focus();
  }
}

// ---------- Clear chat ----------
async function clearChat() {
  try {
    await fetch("/api/clear", { method: "POST" });
  } catch (err) {
    // non-fatal, still clear the UI
  }
  messagesEl.innerHTML = "";
  messagesEl.appendChild(emptyState);
  emptyState.classList.remove("hidden");
  setChatEnabled(false);
  statusBox.classList.add("hidden");
  videoUrlInput.value = "";
}

// ---------- Event listeners ----------
loadBtn.addEventListener("click", loadVideo);

videoUrlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loadVideo();
});

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  chatInput.value = "";
  askQuestion(question);
});

clearBtn.addEventListener("click", clearChat);

tipButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.disabled) return;
    askQuestion(btn.dataset.question);
  });
});
