# Master Frontend & Backend Engineering Interview Guide
## Deep-Dive Analysis of the Streamlit UI, Audio Pipelines, ChromaDB Vector Storage, and Production Infrastructure

> **Project Reference**: `AI-Video-Assistant`  
> **Core Stack**: Streamlit, Python 3.11+, PyTorch, FFmpeg, PyDub, yt-dlp, ChromaDB, Mistral AI, LangChain LCEL  
> **Target Audience**: Full-Stack AI Engineers, Python Backend Developers, and System Architects preparing for senior technical interviews.

---

## Table of Contents
1. [Module 1: Backend Architecture & Audio Pipeline Engineering](#module-1-backend-architecture--audio-pipeline-engineering)
   - [Q1: Backend Layered Architecture & Separation of Concerns](#q1-how-is-the-backend-modularized-in-ai-video-assistant-explain-core-vs-utils-vs-entrypoints)
   - [Q2: External CLI Binaries vs. Python Wrappers: The FFmpeg Dependency](#q2-why-does-ffmpeg-need-to-be-installed-as-a-system-binary-how-do-pydub-and-yt-dlp-interface-with-it)
   - [Q3: Audio Format Conversions & I/O Pipelines](#q3-trace-how-a-youtube-url-or-local-mp4-is-converted-into-16khz-mono-pcm-wav-audio)
   - [Q4: Synchronous REST API Integration: The Sarvam Speech Client](#q4-how-does-transcribe_chunk_sarvam-handle-http-multipart-form-data-uploads-and-timeouts)
   - [Q5: In-Memory Audio Buffers vs. Disk Persistence](#q5-what-are-the-trade-offs-between-writing-temporary-chunk-wav-files-to-disk-vs-using-iobytesio-in-ram)
   - [Q6: Vector Store Lifecycle & Disk Persistence in ChromaDB](#q6-how-does-chromadb-persist-embeddings-to-disk-explain-sqlite-metadata-and-hnsw-binaries)
   - [Q7: Bug Detection & Exception Handling in `core/rag_engine.py`](#q7-identify-and-fix-the-missing-argument-bug-in-load_rag_chain-in-corerag_enginepy)
   - [Q8: LangChain Expression Language (LCEL) Architecture](#q8-how-does-lcel-pipe-syntax-orchestrate-runnablepassthrough-runnablelambda-and-prompt-templates)
2. [Module 2: Frontend Engineering & Streamlit Internals](#module-2-frontend-engineering--streamlit-internals)
   - [Q9: Streamlit Reactive Execution Architecture](#q9-how-does-streamlit-execute-code-explain-the-top-to-bottom-script-rerun-lifecycle-in-apppy)
   - [Q10: Session State Management in Complex Multimodal Apps](#q10-how-does-stsession_state-maintain-chat-history-pipeline-steps-and-results-across-reruns)
   - [Q11: Real-Time Dynamic Status Indicators (`render_step_bar`)](#q11-how-does-render_step_bar-dynamically-switch-between-pending-active-and-done-states)
   - [Q12: Advanced Cyberpunk/Dark Mode CSS Injection](#q12-analyze-the-custom-css-in-apppy-root-css-variables-animations-glassmorphism-and-typography)
   - [Q13: Interactive Chat UI with Bubble Formatting](#q13-how-is-the-chat-history-rendered-in-apppy-explain-the-chat-msg-chat-bubble-and-alignment-logic)
   - [Q14: Dynamic Layout Systems: Proportional Columns and Expanders](#q14-how-do-stcolumns3-2-and-stexpander-create-a-balanced-executive-meeting-dashboard)
   - [Q15: UI State Transitions & Empty States](#q15-how-does-apppy-transition-between-the-hero-empty-state-the-running-spinner-and-the-results-view)
   - [Q16: Preventing Zombie Runs and Handling Browser Disconnections](#q16-what-happens-if-a-user-closes-their-browser-during-a-30-minute-transcription-job)
3. [Module 3: Full-Stack Integration, DevOps & Scalability](#module-3-full-stack-integration-devops--scalability)
   - [Q17: Decoupling Streamlit: Production Architecture with FastAPI & Next.js](#q17-how-would-you-refactor-this-app-into-a-decoupled-fastapi-backend-and-nextjs-frontend)
   - [Q18: Real-Time Token & Audio Streaming: WebSockets vs. SSE](#q18-compare-websockets-and-server-sent-events-sse-for-streaming-meeting-transcripts-and-llm-tokens)
   - [Q19: Distributed Task Processing with Celery & Redis](#q19-how-would-you-architect-an-asynchronous-queue-to-handle-hundreds-of-concurrent-video-transcriptions)
   - [Q20: Multi-Stage Production Dockerfile with FFmpeg & PyTorch](#q20-write-a-production-grade-dockerfile-that-includes-ffmpeg-pytorch-and-cached-whisper-weights)
   - [Q21: Relational Database Schema for Meeting Intelligence](#q21-design-a-postgresql-schema-to-store-organizations-users-meetings-transcripts-and-summaries)
   - [Q22: Cloud Object Storage (AWS S3) & Presigned URLs](#q22-how-should-large-audio-and-video-files-be-uploaded-and-stored-using-aws-s3-presigned-urls)
   - [Q23: Vector DB Production Scaling: ChromaDB vs. Pinecone vs. Qdrant](#q23-compare-chromadb-pinecone-milvus-and-qdrant-for-enterprise-meeting-retrieval)
   - [Q24: End-to-End CI/CD Pipeline with GitHub Actions](#q24-design-a-github-actions-pipeline-for-linting-unit-testing-and-container-deployment)
   - [Q25: Observability, Metrics & Telemetry (OpenTelemetry, Prometheus, Grafana)](#q25-how-do-you-monitor-gpu-utilization-transcription-latency-and-llm-token-costs-in-production)

---

## Module 1: Backend Architecture & Audio Pipeline Engineering

### Q1: How is the backend modularized in `AI-Video-Assistant`? Explain `core/` vs. `utils/` vs. entrypoints.

#### Answer
The project follows a **Domain-Driven Modular Structure**:

```
┌────────────────────────────────────────────────────────┐
│                   ENTRYPOINTS / UI                     │
│               `app.py`  &  `main.py`                   │
│       (Streamlit Frontend & Interactive CLI REPL)      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                      DOMAIN CORE                       │
│                      `core/`                           │
│  - `transcriber.py`: Speech-to-Text & Translation      │
│  - `summarizer.py`: Map-Reduce Synthesis               │
│  - `extractor.py`: Structured Action & Decision Mining │
│  - `vector_store.py`: ChromaDB Vector Indexing         │
│  - `rag_engine.py`: Retrieval & QA LCEL Chains         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                 INFRASTRUCTURE UTILITIES               │
│                      `utils/`                          │
│  - `audio_processor.py`: Media I/O, FFmpeg & Slicing   │
└────────────────────────────────────────────────────────┘
```

1. **`utils/audio_processor.py`**:
   - Manages raw I/O boundaries. It knows nothing about language models, prompts, or vectors. Its sole job is taking YouTube URLs or local video files and returning clean, sliced WAV chunks.
2. **`core/` Domain Modules**:
   - Focused business logic. Each file encapsulates one machine learning discipline: ASR (`transcriber.py`), Vector Embeddings (`vector_store.py`), Document Summarization (`summarizer.py`), and RAG (`rag_engine.py`).
3. **Dual Entrypoints (`app.py` & `main.py`)**:
   - Clean architectural decoupling: the backend engine can be executed via terminal CLI (`python main.py`) or via the web dashboard (`streamlit run app.py`) with identical underlying execution logic.

---

### Q2: Why does FFmpeg need to be installed as a system binary? How do `pydub` and `yt-dlp` interface with it?

#### Answer

#### What is FFmpeg?
FFmpeg is a high-performance C-based command-line multimedia suite that decodes, muxes, demuxes, and transcodes virtually every known audio and video codec.

#### Why Python Libraries Cannot Do This Alone:
- Pure Python is too slow to decode compressed audio streams (AAC, Opus, MP3, H.264) in real-time.
- `pydub` and `yt-dlp` are **Python wrappers / subprocess managers**, not native codec decoders.

```
Python Runtime (pydub) ──► subprocess.Popen(["ffmpeg", "-i", "input.mp4", "-ac", "1", "-ar", "16000", "out.wav"])
                                     │
                                     ▼
                          OS Kernel System Process
                                (FFmpeg.exe)
```

1. When `yt-dlp` extracts audio (`"key": "FFmpegExtractAudio"`), it invokes `ffmpeg` as an external OS process to convert the YouTube stream into `.wav`.
2. When `AudioSegment.from_file(input_path)` runs in `pydub`, it queries `ffmpeg` on the system `PATH` to decode the audio stream into raw PCM byte arrays.
3. **If FFmpeg is missing**: The Python program crashes immediately with `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`.

---

### Q3: Trace how a YouTube URL or local MP4 is converted into 16kHz Mono PCM WAV audio.

#### Answer
In `utils/audio_processor.py`:

```
Input: YouTube URL or local video.mp4
                 │
                 ▼  process_input(source)
┌────────────────────────────────────────────────────────┐
│ Case 1: YouTube URL (download_youtube_audio)           │
│ - yt-dlp extracts raw Opus/AAC audio stream            │
│ - Post-processor spawns FFmpeg: converts to .wav       │
├────────────────────────────────────────────────────────┤
│ Case 2: Local Video (convert_to_wav)                   │
│ - PyDub calls FFmpeg to decode audio container         │
│ - .set_channels(1) -> Downmixes stereo to mono         │
│ - .set_frame_rate(16000) -> Resamples to 16,000 Hz     │
│ - Exports standard linear PCM 16-bit WAV               │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
        chunk_audio(wav_path, chunk_minutes=10)
                         │
                         ▼
        List of 10-minute WAV chunk paths
```

---

### Q4: How does `transcribe_chunk_sarvam` handle HTTP multipart form-data uploads and timeouts?

#### Answer
In `core/transcriber.py`:
```python
def _send_to_sarvam(piece_path: str) -> str:
  headers = {"api-subscription-key": SARVAM_API_KEY}
  with open(piece_path, "rb") as f:
    files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
    data = {"model": SARVAM_MODEL, "with_diarization": "false"}
    response = requests.post(
        SARVAM_STT_TRANSLATE_URL,
        headers=headers,
        files=files,
        data=data,
        timeout=120,
    )
  if not response.ok:
    response.raise_for_status()
  return response.json().get("transcript", "")
```

#### Technical Mechanics:
1. **Multipart/Form-Data Encoding**:
   - `files={"file": ...}` streams the 25-second WAV binary file over HTTP with the `Content-Type: multipart/form-data` boundary header.
2. **`timeout=120`**:
   - Speech translation requires server-side GPU processing. Setting a 120-second timeout prevents the local Python thread from hanging indefinitely if Sarvam's cloud servers experience load spikes.
3. **`with open(...) as f` Context Manager**:
   - Guarantees the OS file handle is closed immediately after the request bytes are transmitted.

---

### Q5: What are the trade-offs between writing temporary chunk WAV files to disk vs. using `io.BytesIO` in RAM?

#### Answer

| Dimension | Disk Persistence (Current: `chunk_path = f"{wav_path}_chunk_{i}.wav"`) | In-Memory Buffers (`io.BytesIO()`) |
| :--- | :--- | :--- |
| **RAM Footprint** | Extremely low. Chunks are read on demand. | High. Storing multiple uncompressed WAV chunks in memory can cause out-of-memory (OOM) crashes on multi-hour audio. |
| **I/O Latency** | Slower (disk write + read cycle). | Near zero (direct memory pointer transfer). |
| **Debuggability** | Outstanding. Engineers can manually open and listen to `chunk_0.wav` to diagnose audio issues. | Difficult. Bytes are ephemeral and vanish on crash. |
| **Disk Cleanup** | Requires manual `os.remove(piece_path)` cleanup logic; unhandled exceptions leave orphaned files. | Automatically garbage collected by Python runtime. |

---

### Q6: How does ChromaDB persist embeddings to disk? Explain SQLite metadata and HNSW binaries.

#### Answer
In `core/vector_store.py`:
```python
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory="vector_db",
)
```

Inside the `vector_db/` folder on your hard drive, ChromaDB creates two primary storage mechanisms:
1. **`chroma.sqlite3` (Metadata & Document Store)**:
   - A relational SQLite database containing tables for collections, documents, chunk text, and metadata dictionaries (`{'chunk_index': i}`).
2. **`index/` Binary Files (HNSW Graph Indices)**:
   - High-dimensional float vectors (384 floating-point numbers per chunk) are serialized into specialized binary index files managed by `hnswlib`.
   - These files contain the multi-layer graph topology that allows logarithmic approximate nearest-neighbor vector traversal.

---

### Q7: Identify and fix the missing-argument bug in `load_rag_chain` in `core/rag_engine.py`.

#### Answer
In `core/rag_engine.py`, lines 58–61:
```python
def load_rag_chain():
  vector_store = load_vector_store()
  retriver = get_retriever()  # <-- BUG HERE!
```

#### The Bug:
In `core/vector_store.py`, `get_retriever` is declared as:
```python
def get_retriever(vector_store: Chroma, k: int = 4):
  return vector_store.as_retriever(...)
```
`vector_store` has no default value! Invoking `get_retriever()` with zero arguments results in:
```text
TypeError: get_retriever() missing 1 required positional argument: 'vector_store'
```

#### The Production Fix:
```python
def load_rag_chain():
  vector_store = load_vector_store()
  retriever = get_retriever(vector_store, k=4)  # Pass vector_store!
  ...
```

---

### Q8: How does LCEL pipe syntax orchestrate `RunnablePassthrough`, `RunnableLambda`, and Prompt Templates?

#### Answer
In `core/rag_engine.py`:
```python
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

#### Under the Hood:
1. **Dictionary Parallelism**:
   - When a dictionary `{ "key1": RunnableA, "key2": RunnableB }` is passed into an LCEL sequence, LangChain wraps it in a `RunnableParallel`. Both branches execute concurrently!
2. **`RunnablePassthrough()`**:
   - Takes whatever input is passed to `rag_chain.invoke(question)` and forwards it unchanged into the `"question"` key.
3. **`retriever | RunnableLambda(format_docs)`**:
   - The query string is passed into the `retriever`, which outputs a `List[Document]`.
   - `RunnableLambda(format_docs)` intercepts the document list, runs custom Python code (`"\n\n".join([doc.page_content for doc in docs])`), and outputs a clean string.
4. **Prompt $\rightarrow$ LLM $\rightarrow$ Parser**:
   - The dictionary `{"context": "...", "question": "..."}` matches the template variables in `ChatPromptTemplate`, formatting the chat messages and streaming them into `ChatMistralAI`.

---

## Module 2: Frontend Engineering & Streamlit Internals

### Q9: How does Streamlit execute code? Explain the top-to-bottom script rerun lifecycle in `app.py`.

#### Answer
Streamlit does not use an event-driven virtual DOM like React. Instead, it operates on a **Linear Re-execution Model**:

```
[User Action in Browser (Clicks 'Analyse' or 'Send →')]
                         │
                         ▼ (WebSocket Signal to Server)
[Streamlit Server interrupts previous execution]
                         │
                         ▼
[Executes app.py from Line 1 to Line 545]
  - Imports libraries
  - Re-evaluates CSS st.markdown(...)
  - Checks st.session_state flags
  - Renders UI elements that match current state
```

Every button click or text submission triggers a complete rerun from Line 1! Local Python variables are re-allocated; only variables stored inside `st.session_state` survive across reruns.

---

### Q10: How does `st.session_state` maintain chat history, pipeline steps, and results across reruns?

#### Answer
In `app.py` lines 306–316:
```python
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
  if key not in st.session_state:
    st.session_state[key] = default
```

```
Initial Page Load:
st.session_state = { "result": None, "chat_history": [], "pipeline_done": False, ... }
                         │
                         ▼ User clicks "Analyse"
Pipeline Execution:
st.session_state.result = { "title": "...", "summary": "...", "rag_chain": ... }
st.session_state.pipeline_done = True
st.rerun()
                         │
                         ▼ Page Reruns (Now lines 430-528 execute!)
Chat Interaction:
User types: "Who approved the budget?" and clicks "Send ->"
app.py executes again from Line 1!
Because st.session_state.result is populated, the results cards remain visible.
st.session_state.chat_history.append({"role": "user", "content": ...})
st.session_state.chat_history.append({"role": "assistant", "content": ...})
st.rerun()
```

---

### Q11: How does `render_step_bar` dynamically switch between `pending`, `active`, and `done` states?

#### Answer
In `app.py`:
```python
def step_status(steps: dict, key: str) -> str:
  s = steps.get(key, "pending")
  if s == "active":
    return "dot-active"
  if s == "done":
    return "dot-done"
  return "dot-pending"


def render_step_bar(label: str, key: str, icon: str):
  css = step_status(st.session_state.pipeline_steps, key)
  st.markdown(
      f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""",
      unsafe_allow_html=True,
  )
```

#### Dynamic CSS Animation:
- `.dot-pending`: Dim gray dot (`var(--border)`).
- `.dot-active`: Glowing purple dot (`var(--accent-glow)`) animated with a pulsating CSS keyframe animation (`animation: pulse 1.5s infinite;`).
- `.dot-done`: Solid vibrant green dot (`var(--success)`).

---

### Q12: Analyze the custom CSS in `app.py`: Root CSS variables, animations, glassmorphism, and typography.

#### Answer

1. **CSS Custom Properties (Variables)**:
   ```css
   :root {
       --bg: #0a0a0f;
       --surface: #111118;
       --surface-2: #1a1a25;
       --border: #2a2a3a;
       --accent: #7c3aed;       /* Electric Violet */
       --accent-glow: #9f67ff;
       --accent-2: #06b6d4;      /* Cyber Cyan */
       --text: #e8e8f0;
       --success: #10b981;
   }
   ```
   Centralized theming ensures consistency across cards, buttons, badges, and chat bubbles.
2. **Synthetic Grid Matrix**:
   Lines 52–64 project a subtle 40px $\times$ 40px background grid using dual repeating linear gradients, giving the app an authentic developer-focused aesthetic.
3. **Card Glassmorphism & Hover Translation**:
   ```css
   .card:hover { border-color: var(--accent); }
   .card::before {
       content: '';
       position: absolute;
       top: 0; left: 0;
       width: 3px; height: 100%;
       background: linear-gradient(180deg, var(--accent), var(--accent-2));
   }
   ```
   Every content card features a vertical gradient accent bar on its left border.

---

### Q13: How is the chat history rendered in `app.py`? Explain the `chat-msg`, `chat-bubble`, and alignment logic.

#### Answer
In `app.py` lines 486–503:
- The chat container is styled as an auto-scrolling terminal box (`max-height: 420px; overflow-y: auto`).
- **User Messages**:
  - Aligned to the right (`align-items: flex-end; align-self: flex-end`).
  - Styled with soft purple background tint: `rgba(124, 58, 237, 0.15)`.
- **Assistant Messages**:
  - Aligned to the left (`align-items: flex-start; align-self: flex-start`).
  - Styled with cyan background tint: `rgba(6, 182, 212, 0.1)`.
- When the user asks a question, both user query and assistant response are appended to `st.session_state.chat_history`, maintaining a conversational session log.

---

### Q14: How do `st.columns([3, 2])` and `st.expander` create a balanced executive meeting dashboard?

#### Answer
In `app.py` lines 443–455:
```python
col1, col2 = st.columns([3, 2], gap="medium")
with col1:
  # Executive Summary (occupies 60% of horizontal width)
  ...
with col2:
  # Full Transcript in Collapsible Expander (occupies 40% width)
  with st.expander("📝 Full Transcript", expanded=False):
    ...
```
- Meeting summaries are high-level and read first; hence they receive 60% width (`[3]`).
- Raw transcripts can be tens of thousands of words long. Displaying the raw transcript inline would force users to scroll through miles of text. Wrapping it inside an `st.expander(expanded=False)` keeps the dashboard clean while keeping the verbatim transcript accessible on demand.

---

### Q15: How does `app.py` transition between the hero empty state, the running spinner, and the results view?

#### Answer
Streamlit uses **Conditional Component Rendering**:
1. **Empty State (Default)**:
   - When `st.session_state.result is None`, lines 530–545 render a stylized empty state with an icon (`🎬`), instructions, and capability badges.
2. **Running State**:
   - `progress_placeholder = st.empty()` creates a dynamic UI container that can be overwritten or cleared in real time without refreshing the entire page.
3. **Results State**:
   - As soon as `st.session_state.result` is populated, the empty state is automatically bypassed, and the two-column executive cards and interactive RAG chat are rendered.

---

### Q16: What happens if a user closes their browser during a 30-minute transcription job?

#### Answer
In default Streamlit:
- The browser connection is managed over a WebSocket.
- When the user closes the tab, the WebSocket disconnects.
- However, the Python server thread running Whisper or PyTorch **continues executing until the function finishes**, consuming CPU/GPU cycles and API credits for a session that has no listener!
- **Production Solution**:
  - Offload long-running transcription tasks to a background worker queue (Celery/Redis).
  - The browser tab merely displays progress polled from a job ID. If the tab closes, the job completes in the background and is saved to PostgreSQL for the user to view whenever they log back in!

---

## Module 3: Full-Stack Integration, DevOps & Scalability

### Q17: How would you refactor this app into a decoupled FastAPI backend and Next.js frontend?

#### Answer

```
┌──────────────────────────────┐              ┌──────────────────────────────┐
│     Next.js 14 (Frontend)    │   REST / SSE │       FastAPI (Backend)      │
│  - React Server Components   │ ◄──────────► │  - Background Task Queue     │
│  - Tailwind CSS + Lucide     │              │  - Faster-Whisper ASR Worker │
│  - Wavesurfer.js Audio Player│              │  - Mistral LCEL Pipelines    │
└──────────────────────────────┘              └──────────────────────────────┘
```

#### FastAPI Endpoint Architecture:
1. `POST /api/meetings/process`: Accepts audio file or YouTube URL, creates a database record, enqueues background job, and returns `{ "meeting_id": "uuid" }`.
2. `GET /api/meetings/{id}/stream`: Server-Sent Events (SSE) endpoint streaming real-time transcription status.
3. `POST /api/meetings/{id}/chat`: RAG endpoint accepting a question and streaming tokens back to the user.

---

### Q18: Compare WebSockets and Server-Sent Events (SSE) for streaming meeting transcripts and LLM tokens.

#### Answer

| Dimension | Server-Sent Events (SSE) | WebSockets (WS) |
| :--- | :--- | :--- |
| **Protocol** | Standard HTTP (`Content-Type: text/event-stream`) | Full-Duplex TCP (`ws://` / `wss://`) |
| **Data Direction** | Server $\rightarrow$ Client (Unidirectional) | Client $\longleftrightarrow$ Server (Bidirectional) |
| **Reconnection** | Native browser auto-reconnect built-in | Requires custom client-side heartbeat & reconnect |
| **Proxy / Firewall** | Traverses standard corporate firewalls effortlessly | Often blocked or abruptly closed by proxies |
| **Verdict** | **Best for LLM token streaming & transcription progress** | Best only if client sends live microphone audio streams |

---

### Q19: How would you architect an asynchronous queue to handle hundreds of concurrent video transcriptions?

#### Answer

```
[User Upload] ──► [FastAPI Gateway] ──► Save Raw Media to S3 ──► Push Job to Redis Queue
                                                                        │
                                      ┌─────────────────────────────────┴─────────────────────────────────┐
                                      ▼                                                                   ▼
                          [GPU Worker 1 (Celery)]                                             [GPU Worker 2 (Celery)]
                          - FFmpeg 16kHz conversion                                           - FFmpeg 16kHz conversion
                          - Faster-Whisper ASR                                                - Faster-Whisper ASR
                          - Generate Chroma Embeddings                                        - Generate Chroma Embeddings
                                      │                                                                   │
                                      └─────────────────────────────────┬─────────────────────────────────┘
                                                                        │
                                                                        ▼
                                                         [Save Results to PostgreSQL]
                                                         [Notify User via Webhook / Push]
```

1. **Celery Task Worker**:
   ```python
   @celery_app.task(bind=True)
   def process_meeting_task(self, meeting_id: str, source_url: str):
     # Run audio extraction, transcription, summarization, and vector indexing
     ...
```
2. **Worker Scaling**:
   - GPU worker nodes auto-scale based on Redis queue depth ($Q_{\text{len}}$) using Kubernetes Horizontal Pod Autoscaler (HPA).

---

### Q20: Write a production-grade Dockerfile that includes FFmpeg, PyTorch, and cached Whisper weights.

#### Answer

```dockerfile
# Multi-stage production build
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies & FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY Requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r Requirements.txt

# Pre-download Whisper model weights during build to prevent cold-start download delay!
RUN python3.11 -c "import whisper; whisper.load_model('small')"

# Pre-download Sentence-Transformers embeddings
RUN python3.11 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# Run as non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

### Q21: Design a PostgreSQL schema to store organizations, users, meetings, transcripts, and summaries.

#### Answer

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users Table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Meetings Table
CREATE TABLE meetings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    source_url TEXT,
    duration_seconds INTEGER,
    language VARCHAR(50) DEFAULT 'english',
    status VARCHAR(50) DEFAULT 'completed', -- pending, processing, completed, failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Detailed Meeting Intelligence Artifacts
CREATE TABLE meeting_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meeting_id UUID UNIQUE REFERENCES meetings(id) ON DELETE CASCADE,
    full_transcript TEXT NOT NULL,
    executive_summary TEXT NOT NULL,
    action_items TEXT,
    key_decisions TEXT,
    open_questions TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Interactive Chat Logs
CREATE TABLE meeting_chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    message TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_meetings_user ON meetings(user_id);
CREATE INDEX idx_chat_meeting ON meeting_chat_messages(meeting_id);
```

---

### Q22: How should large audio and video files be uploaded and stored using AWS S3 presigned URLs?

#### Answer
Uploading a $500\text{MB}$ video directly through your application API server bottlenecks your web workers.

```
Client (Browser) ──► GET /api/upload-url ──► FastAPI generates S3 Presigned PUT URL
       │
       ▼ Direct Upload (Bypasses API Server entirely!)
Amazon S3 Bucket (raw-media-uploads/)
       │
       ▼ S3 ObjectCreated Event Notification
AWS SQS / Redis Queue ──► Triggers Celery Worker to process video!
```

1. **Step 1**: Client requests a presigned URL:
   ```python
   s3_client.generate_presigned_url(
       "put_object",
       Params={"Bucket": "meetings-bucket", "Key": f"audio/{uuid}.wav"},
       ExpiresIn=300,
   )
```
2. **Step 2**: The client uploads the file directly to S3 via standard HTTP PUT.
3. **Step 3**: The API server is notified to begin background transcription.

---

### Q23: Compare ChromaDB, Pinecone, Milvus, and Qdrant for enterprise meeting retrieval.

#### Answer

| Vector DB | Deployment | Language | Best Fit | Scalability |
| :--- | :--- | :--- | :--- | :--- |
| **ChromaDB** *(Used Here)* | Embedded / Local | Python/C++ | Prototyping, small-to-medium single-server apps | Millions of vectors |
| **Qdrant** | Self-hosted / Cloud | Rust | High-performance filtering, payload storage | Tens of millions |
| **Pinecone** | Fully Managed Cloud | Proprietary | Zero DevOps management, serverless scaling | Billions of vectors |
| **Milvus** | Distributed Cluster | Go/C++ | Massive enterprise scale, multi-modal search | Billions of vectors |

---

### Q24: Design a GitHub Actions pipeline for linting, unit testing, and container deployment.

#### Answer

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install System Dependencies
        run: sudo apt-get update && sudo apt-get install -y ffmpeg

      - name: Install Python Dependencies
        run: |
          pip install --upgrade pip
          pip install ruff pytest -r Requirements.txt

      - name: Lint with Ruff
        run: ruff check .

      - name: Run Unit Tests
        env:
          MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
        run: pytest tests/ -v

  build-docker:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ secrets.DOCKERHUB_USERNAME }}/ai-video-assistant:latest
```

---

### Q25: How do you monitor GPU utilization, transcription latency, and LLM token costs in production?

#### Answer
1. **GPU Monitoring**:
   - Deploy `nvidia-dcgm-exporter` to export GPU VRAM usage, temperature, and CUDA core utilization into **Prometheus**.
2. **ASR & RAG Latency**:
   - Instrument functions with **OpenTelemetry (OTel)** spans to measure exact time spent in:
     $T_{\text{audio\_conversion}} + T_{\text{whisper\_inference}} + T_{\text{chroma\_query}} + T_{\text{llm\_generation}}$.
3. **LLM Cost & Token Accounting**:
   - Use **LangSmith** or **Helicone** proxy to log input tokens, output tokens, latency, and cost per meeting in real time.

---
*Created as part of the AI Video Assistant technical documentation.*
