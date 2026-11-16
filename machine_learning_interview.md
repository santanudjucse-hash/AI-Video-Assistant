# Master Machine Learning & AI Engineering Interview Guide
## Deep-Dive Analysis of the AI Video Assistant, Audio Processing, ASR, RAG, and Vector Search

> **Project Reference**: `AI-Video-Assistant`  
> **Core Stack**: Python 3.11+, PyTorch, OpenAI Whisper, Sarvam AI, Sentence-Transformers (`all-MiniLM-L6-v2`), ChromaDB, Mistral AI (`mistral-small-latest`), LangChain LCEL, PyDub, yt-dlp, Streamlit  
> **Target Audience**: Beginners to Senior AI Engineers preparing for Machine Learning, GenAI, and Multimodal Audio/RAG interviews.

---

## Table of Contents
1. [Module 1: Audio Signal Processing & Speech-to-Text (ASR)](#module-1-audio-signal-processing--speech-to-text-asr)
   - [Q1: Fundamentals of Digital Audio: Sampling Rate, Bit Depth, Mono vs. Stereo](#q1-what-is-digital-audio-explain-sampling-rate-nyquist-theorem-channels-and-why-16khz-mono-wav-is-required)
   - [Q2: Whisper Architecture & Deep Learning for ASR](#q2-how-does-openai-whisper-work-from-the-ground-up-explain-log-mel-spectrograms-encoder-decoder-transformers-and-multitask-tokens)
   - [Q3: Multilingual ASR & Speech Translation: Whisper vs. Sarvam AI](#q3-how-does-transcription-differ-from-speech-translation-explain-sarvam-ai-saarasv25-for-hinglish)
   - [Q4: Audio Chunking & Slicing Strategies](#q4-why-does-the-system-slice-audio-into-10-minute-and-25-second-pieces-explain-the-memory-and-api-constraints)
2. [Module 2: Embeddings, Vector Databases & Retrieval (RAG)](#module-2-embeddings-vector-databases--retrieval-rag)
   - [Q5: Text Embeddings & Sentence-Transformers (`all-MiniLM-L6-v2`)](#q5-what-is-a-text-embedding-how-does-all-minilm-l6-v2-convert-transcripts-into-dense-vectors)
   - [Q6: Vector Similarity Metrics: Cosine Similarity vs. Euclidean Distance vs. Dot Product](#q6-compare-cosine-similarity-euclidean-distance-and-inner-product-how-are-retrieval-scores-computed)
   - [Q7: Vector Database Architecture: Inside ChromaDB & HNSW](#q7-how-does-chromadb-store-and-index-vectors-explain-hierarchical-navigable-small-world-hnsw-graphs)
   - [Q8: Text Chunking & Splitting Strategies](#q8-why-use-recursivecharactertextsplitter-with-chunk_size500-and-chunk_overlap50-in-rag)
   - [Q9: Retrieval-Augmented Generation (RAG) Architecture](#q9-walk-through-the-rag-pipeline-in-core-rag_enginepy-how-does-lcel-wire-retrieval-to-generation)
3. [Module 3: LLM Reasoning, Map-Reduce Summarization & Information Extraction](#module-3-llm-reasoning-map-reduce-summarization--information-extraction)
   - [Q10: Mistral Architecture & Sliding-Window Attention](#q10-what-is-mistral-ai-explain-sliding-window-attention-swa-grouped-query-attention-gqa-and-temperature03)
   - [Q11: Map-Reduce Summarization Pattern](#q11-how-does-the-map-reduce-summarization-pattern-work-in-core-summarizerpy-compare-with-stuff-and-refine)
   - [Q12: Information Extraction: Action Items, Decisions, and Open Questions](#q12-how-does-core-extractorpy-extract-structured-action-items-owners-and-deadlines-from-unstructured-meetings)
   - [Q13: Hallucination Mitigation & Grounded Prompting](#q13-how-does-the-system-prompt-prevent-hallucinations-during-meeting-qa)
4. [Module 4: Codebase Line-by-Line Technical Breakdown](#module-4-codebase-line-by-line-technical-breakdown)
   - [Q14: Deep Dive into `utils/audio_processor.py`](#q14-analyze-utilsaudio_processorpy-line-by-line-explain-yt-dlp-and-pydub-transformations)
   - [Q15: Deep Dive into `core/transcriber.py`](#q15-analyze-coretranscriberpy-line-by-line-explain-model-caching-and-language-routing)
   - [Q16: Deep Dive into `core/vector_store.py`](#q16-analyze-corevector_storepy-line-by-line-explain-collection-persistence)
   - [Q17: Deep Dive into `core/rag_engine.py`](#q17-analyze-corerag_enginepy-line-by-line-identify-potential-bugs-and-design-patterns)
   - [Q18: Deep Dive into `core/summarizer.py`](#q18-analyze-coresummarizerpy-line-by-line-explain-prompt-chaining)
   - [Q19: Deep Dive into `main.py`](#q19-analyze-mainpy-explain-the-cli-execution-flow-and-interactive-chat-loop)
5. [Module 5: Advanced System Scaling, Evaluation & Productionization](#module-5-advanced-system-scaling-evaluation--productionization)
   - [Q20: Quantitative Evaluation: ROUGE, BLEU, and RAGAS](#q20-how-do-you-evaluate-transcription-wer-and-meeting-summaries-rouge-bleu-ragas)
   - [Q21: Hardware Acceleration: CUDA, TensorRT-LLM, and Whisper.cpp](#q21-how-would-you-optimize-whisper-latency-for-production-compare-fp32-fp16-int8-and-whispercpp)
   - [Q22: Streaming ASR & Real-Time Meeting Transcription](#q22-how-would-you-convert-this-batch-transcription-system-into-a-real-time-live-streaming-assistant)
   - [Q23: Handling Long Multi-Hour Meetings: Context Window Limits](#q23-what-happens-when-a-meeting-is-4-hours-long-how-does-the-system-prevent-context-overflow)
   - [Q24: Security, PII Redaction & Data Privacy in Meeting AI](#q24-how-do-you-handle-sensitive-pii-passwords-and-confidential-discussion-in-meeting-transcripts)
   - [Q25: Architecture Evolution: Production Microservices Architecture](#q25-design-an-enterprise-production-architecture-with-fastapi-celery-redis-and-webrtc)

---

## Module 1: Audio Signal Processing & Speech-to-Text (ASR)

### Q1: What is digital audio? Explain Sampling Rate, Nyquist Theorem, Channels, and why 16kHz mono WAV is required.

#### Answer
Sound in the physical world is a continuous analog wave of air pressure vibrations. To process sound in a computer, we must convert this continuous wave into discrete digital numbers via **Analog-to-Digital Conversion (ADC)**.

```
Analog Wave (Continuous Pressure) ──► Sampling (Discrete Time) ──► Quantization (Bit Depth) ──► Digital Audio Tensor
```

#### Core Concepts:
1. **Sampling Rate ($f_s$)**:
   - The number of audio snapshots recorded per second, measured in Hertz (Hz).
   - Standard rates: $44,100\text{ Hz}$ (CD quality), $48,000\text{ Hz}$ (Video production), $16,000\text{ Hz}$ (Speech recognition).
2. **The Nyquist-Shannon Sampling Theorem**:
   $$f_s \ge 2 \cdot f_{\text{max}}$$
   - To accurately capture frequencies up to $f_{\text{max}}$ without distortion (aliasing), you must sample at at least twice that frequency.
   - Human speech intelligible frequencies reside below $8,000\text{ Hz}$. Therefore, a sampling rate of **$16,000\text{ Hz}$ ($16\text{ kHz}$)** captures all relevant acoustic features of human speech while throwing away unnecessary high-frequency noise.
3. **Channels (Mono vs. Stereo)**:
   - **Stereo (2 channels)**: Left and right audio recorded separately for spatial hearing.
   - **Mono (1 channel)**: Single audio stream. Speech recognition models process human linguistic content, not spatial placement. Stereo adds 2x redundant memory without adding acoustic information.
4. **Why `utils/audio_processor.py` line 33 converts to 16kHz Mono**:
   ```python
   audio = audio.set_channels(1).set_frame_rate(16000)
   ```
   OpenAI Whisper was explicitly pretrained on $16\text{ kHz}$ single-channel audio. Passing $44.1\text{ kHz}$ stereo directly into Whisper would cause mismatched feature dimensions and degrade transcription accuracy.

---

### Q2: How does OpenAI Whisper work from the ground up? Explain Log-Mel Spectrograms, Encoder-Decoder Transformers, and Multitask Tokens.

#### Answer
OpenAI Whisper is an **Encoder-Decoder Transformer** trained on 680,000 hours of multilingual, multitask supervised audio data.

```
Raw Audio (16kHz Mono WAV)
          │
          ▼
[Short-Time Fourier Transform (STFT)] ──► 80-Channel Log-Mel Spectrogram
          │
          ▼
[2x 1D Convolutional Downsampling Layers] (Stride 2)
          │
          ▼
[Transformer Audio Encoder] (Self-Attention over acoustic frames)
          │
          ▼  Cross-Attention
[Transformer Text Decoder] (Autoregressive Token Generation)
          │
          ▼
Generated Tokens: [<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>, "Hello", "world"]
```

#### Step-by-Step Mechanism:
1. **Audio Feature Extraction (Log-Mel Spectrogram)**:
   - Whisper breaks raw audio into 30-second windows.
   - It computes the **Short-Time Fourier Transform (STFT)** using a 25ms window and 10ms hop size.
   - It projects the linear frequency spectrum onto **80 Mel frequency bins**, reflecting how human ears perceive pitch (non-linear sensitivity).
   - Log compression is applied: $\log(1 + \text{Mel})$, creating an $80 \times 3000$ feature tensor for a 30s audio segment.
2. **The Encoder**:
   - Two 1D convolutional layers downsample the input by a factor of 2.
   - Transformer encoder blocks with multi-head self-attention extract deep acoustic representations.
3. **The Decoder & Special Multitask Prompt Tokens**:
   - The decoder is an autoregressive language model that attends to the encoder representations via cross-attention.
   - Whisper handles transcription, language identification, and translation using a sequence of special conditioning tokens:
     - `<|startoftranscript|>`: Signals start of decoding.
     - `<|en|>` / `<|hi|>`: Target language identification.
     - `<|transcribe|>` vs. `<|translate|>`: Dictates whether to transcribe in the native language or translate directly into English!
     - `<|notimestamps|>`: Omits millisecond timestamps.

---

### Q3: How does transcription differ from speech translation? Explain Sarvam AI (`saaras:v2.5`) for Hinglish.

#### Answer

| Dimension | Transcription (`task="transcribe"`) | Speech Translation (`task="translate"`) |
| :--- | :--- | :--- |
| **Input $\rightarrow$ Output** | Hindi audio $\rightarrow$ Hindi text (Devanagari: "नमस्ते दुनिया") | Hindi audio $\rightarrow$ English text ("Hello world") |
| **Language Consistency** | Preserves original spoken language | Translates cross-lingually into target language |
| **Handling Code-Mixing (Hinglish)** | Struggles when speakers mix English and Hindi words within one sentence | Accurately identifies intent and outputs coherent English text |

#### Why Sarvam AI is used for Hinglish in `core/transcriber.py`:
In India and global corporate environments, meetings frequently use **Hinglish** (code-switching between Hindi and English: *"Kal meeting me feature finalize karenge"*).
- Standard Whisper models often fail on Hinglish, outputting phonetic gibberish or switching scripts back and forth between Latin and Devanagari.
- **Sarvam AI (`saaras:v2.5`)** is a specialized foundation speech model built specifically for Indian languages. It performs direct **Speech-to-Text-Translation (STTT)**, taking spoken Hindi/Hinglish audio and outputting standardized, grammatically correct English text.

---

### Q4: Why does the system slice audio into 10-minute and 25-second pieces? Explain the memory and API constraints.

#### Answer

In this codebase, audio is sliced at two distinct stages:

```
[Full Audio File: e.g., 60 mins]
              │
              ▼  utils/audio_processor.py: chunk_audio(chunk_minutes=10)
   [10-minute Macro Chunks] ──► Prevents PyTorch RAM Overflow & Enables Progress Tracking
              │
              ▼ (If Language == "hinglish")
   core/transcriber.py: transcribe_chunk_sarvam(chunk_path)
              │
              ▼  Slices into 25-second pieces (SARVAM_PIECE_SECONDS = 25)
   [25s Micro Pieces] ──► Complies with Sarvam API Synchronous 30s Payload Limit
```

1. **Macro Chunking (10 minutes in `utils/audio_processor.py`)**:
   - Processing a 2-hour audio file in one continuous memory buffer can exceed GPU VRAM or host RAM during PyTorch tensor allocation.
   - Slicing into 10-minute chunks allows sequential garbage collection, step-by-step UI progress updates (`st.progress`), and fault tolerance (if chunk 4 fails, chunks 1–3 are already completed).
2. **Micro Slicing (25 seconds in `core/transcriber.py`)**:
   - Sarvam AI's synchronous HTTP REST API strictly enforces a **30-second maximum duration per audio payload**. Any audio $> 30\text{s}$ returns an HTTP 400 Bad Request error.
   - The code slices audio into 25-second segments with a 5-second safety margin to ensure reliable HTTP transmission.

---

## Module 2: Embeddings, Vector Databases & Retrieval (RAG)

### Q5: What is a text embedding? How does `all-MiniLM-L6-v2` convert transcripts into dense vectors?

#### Answer

#### What is an Embedding?
An embedding is a mathematical mapping of discrete human text into a continuous high-dimensional vector space ($\mathbb{R}^D$):
$$\text{Text} \longrightarrow \vec{v} = [0.042, -0.118, 0.941, \dots, -0.023] \in \mathbb{R}^{384}$$
Words and sentences that share semantic meaning are mapped close together in this vector space, while unrelated concepts are positioned far apart.

```
       ▲ Dimension 2
       │         (High semantic similarity)
       │         [Meeting Action Item]   [Task Assigned to Engineer]
       │                   •                        •
       │
       │
       │                   •
       │          [Dinner Recipe for Pasta] (Low similarity)
       └────────────────────────────────────────► Dimension 1
```

#### Inside `all-MiniLM-L6-v2`:
In `core/vector_store.py` line 9:
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```
- **Architecture**: A distilled BERT-based Transformer model featuring 6 layers, 384 hidden dimensions, and 22.7M parameters.
- **Bi-Encoder Training**: Trained using **Contrastive Learning** (SimCSE / MultipleNegativesRankingLoss). Positive sentence pairs (questions and relevant answers) are pulled together, while random negative sentences are pushed apart.
- **Efficiency**: It generates a 384-dimensional dense vector in just $\approx 10\text{ms}$ on CPU, making it lightweight for local deployment without requiring cloud embedding API costs.

---

### Q6: Compare Cosine Similarity, Euclidean Distance, and Inner Product. How are retrieval scores computed?

#### Answer

Given two embedding vectors $\vec{u}$ and $\vec{v}$:

1. **Cosine Similarity**:
   $$\text{Cosine}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|} = \frac{\sum_{i=1}^D u_i v_i}{\sqrt{\sum_{i=1}^D u_i^2} \sqrt{\sum_{i=1}^D v_i^2}}$$
   - Measures the **angle** between the two vectors, completely ignoring their magnitude (length).
   - Range: $[-1, 1]$ (or $[0, 1]$ for normalized embeddings). $1.0$ means identical direction.
   - **Why it's preferred in RAG**: Longer text chunks naturally have larger vector magnitudes. Cosine similarity prevents long chunks from dominating retrieval purely due to length.
2. **Euclidean Distance ($L_2$ Distance)**:
   $$D_{L2}(\vec{u}, \vec{v}) = \sqrt{\sum_{i=1}^D (u_i - v_i)^2}$$
   - Measures the geometric distance between two vector arrow tips. Smaller distance means higher similarity.
3. **Inner Product (Dot Product)**:
   $$\vec{u} \cdot \vec{v} = \sum_{i=1}^D u_i v_i$$
   - When vectors are **$L_2$ normalized** ($\|\vec{u}\| = \|\vec{v}\| = 1$), Cosine Similarity and Dot Product are mathematically equivalent, allowing ultra-fast hardware computation using matrix multiplication (`GEMM`).

---

### Q7: How does ChromaDB store and index vectors? Explain Hierarchical Navigable Small World (HNSW) graphs.

#### Answer

In `core/vector_store.py`:
```python
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
)
```

#### The Naive Search Problem (Flat Search):
If you have $N$ chunks in your meeting transcript, comparing a query against every chunk requires $O(N \cdot D)$ computations. When scaling to thousands of meetings, brute-force search is too slow.

#### The Solution: HNSW (Hierarchical Navigable Small World):
ChromaDB uses HNSW, an **Approximate Nearest Neighbor (ANN)** graph algorithm inspired by the "Six Degrees of Separation" concept.

```
Layer 2 (Express Highway):       [Node A] ──────────────────────────► [Node Z]
                                    │                                    │
                                    ▼                                    ▼
Layer 1 (Regional Roads):        [Node A] ──────► [Node M] ──────────► [Node Z]
                                    │               │                    │
                                    ▼               ▼                    ▼
Layer 0 (Local Streets):         [Node A] ─► [B] ─► [Node M] ─► [P] ──► [Node Z]
                                                    ▲
                                              Target Vector Found!
```

1. **Multi-Layer Skip-List Graph**:
   - The top layer has few nodes and long-distance links for coarse, rapid exploration across vector space.
   - Lower layers contain dense, localized links for fine-grained nearest neighbor identification.
2. **Search Complexity**: Drops from linear $O(N)$ to logarithmic **$O(\log N)$**.
3. **Storage in ChromaDB**:
   - Embeddings and HNSW graph indices are serialized in the `vector_db/` folder using SQLite for document metadata and binary files for vector indices.

---

### Q8: Why use `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=50` in RAG?

#### Answer
In `core/vector_store.py` lines 20–23:
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(transcript)
```

#### Why "Recursive" Splitting?
A naive splitter cuts text strictly every 500 characters, frequently severing words or sentences in half (e.g., cutting in the middle of a person's name or action item).
`RecursiveCharacterTextSplitter` tries splitting on logical boundaries in a hierarchical cascade:
1. Double newlines (`\n\n` - Paragraphs)
2. Single newlines (`\n` - Sentences / Lines)
3. Spaces (` ` - Words)
4. Characters (as an absolute fallback)

#### Why `chunk_size = 500`?
- 500 characters is approximately **80–120 words** (a single coherent thought or topic turn in a meeting).
- If chunks are too small (e.g., 50 chars), they lack context. If too large (e.g., 5000 chars), the embedding gets diluted, hurting retrieval precision.

#### Why `chunk_overlap = 50`?
- Overlap creates a 50-character sliding buffer between adjacent chunks.
- If a key decision spans across chunk boundaries, overlap ensures that both chunks retain enough context so the boundary sentence is not lost during retrieval.

---

### Q9: Walk through the RAG pipeline in `core/rag_engine.py`. How does LCEL wire retrieval to generation?

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

#### Execution Graph:
```
User Query: "What was the budget approved?"
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
["question": RunnablePassthrough()]    ["context": retriever]
  Passes string verbatim:                Embeds query via all-MiniLM-L6-v2
  "What was the budget approved?"        Queries Chroma HNSW for top 4 chunks
                                                 │
                                                 ▼
                                       [RunnableLambda(format_docs)]
                                       Joins 4 doc contents with "\n\n"
       │                                         │
       └────────────────────┬────────────────────┘
                            │
                            ▼
          [ChatPromptTemplate.from_messages]
          Fills {context} and {question}
                            │
                            ▼
          [ChatMistralAI: mistral-small-latest]
          Generates answer strictly on context
                            │
                            ▼
          [StrOutputParser]
          Extracts string text from AIMessage
```

---

## Module 3: LLM Reasoning, Map-Reduce Summarization & Information Extraction

### Q10: What is Mistral AI? Explain Sliding-Window Attention (SWA), Grouped-Query Attention (GQA), and `temperature=0.3`.

#### Answer

#### What is Mistral?
Mistral is a state-of-the-art open-weight and API-driven language model family engineered by Mistral AI, recognized for high reasoning density per parameter.

#### Key Architectural Innovations:
1. **Sliding-Window Attention (SWA)**:
   - Traditional attention scales quadratically: $O(N^2)$ with sequence length $N$.
   - Mistral uses a fixed window size $W$ (e.g., 4096 tokens). Token $i$ only attends to tokens in $[i-W, i]$.
   - However, through stacked transformer layers, information propagates across windows, achieving effective theoretical attention across 32,000+ tokens with linear memory complexity $O(N \cdot W)$.
2. **Grouped-Query Attention (GQA)**:
   - Instead of allocating a separate Key and Value head for every Query head, GQA shares single Key/Value heads across groups of Query heads. This reduces KV-cache memory bandwidth during inference by up to 8x.
3. **`temperature=0.3` in this project**:
   - `temperature=0.0`: Completely deterministic, greedy argmax decoding.
   - `temperature=0.3`: Low-temperature setting. It preserves factual fidelity to the transcript while providing slight linguistic flexibility for grammatical fluency when synthesizing long summaries.

---

### Q11: How does the Map-Reduce Summarization pattern work in `core/summarizer.py`? Compare with Stuff and Refine.

#### Answer

| Strategy | Mechanism | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Stuffing** | Pours the entire transcript into a single prompt. | Simple, 1 LLM call. | Crashes if transcript exceeds model context window; causes LLM distraction. |
| **Refine** | Iteratively updates summary: Chunk 1 summary $\rightarrow$ update with Chunk 2 $\rightarrow$ update with Chunk 3. | High continuity across sections. | **Strictly sequential**: cannot be parallelized; early errors compound over time. |
| **Map-Reduce** *(Used Here)* | **Map**: Summarize every chunk in parallel. **Reduce**: Combine all mini-summaries into one master document. | **Scales infinitely**; Map step can run concurrently across worker threads. | May lose fine cross-chunk context if ideas span multiple segments. |

#### Implementation in `core/summarizer.py`:
```python
# 1. Chunk transcript into 3000-character segments
chunks = split_transcript(transcript)

# 2. Map Phase: Summarize each chunk
chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

# 3. Reduce Phase: Synthesize partial summaries into final bullet points
combined = "\n\n".join(chunk_summaries)
final_summary = combined_chain.invoke(combined)
```

---

### Q12: How does `core/extractor.py` extract structured Action Items, Owners, and Deadlines from unstructured meetings?

#### Answer
In `core/extractor.py`:
```python
def extract_action_items(transcript: str) -> str:
  chain = build_chain(
      "You are an expert meeting analyst. From the meeting transcript, extract"
      " all action items. For each provide:\n"
      "- Task description\n"
      "- Owner (who is responsible)\n"
      "- Deadline (if mentioned, else write 'Not specified')\n\n"
      "Format as a numbered list. If none found say 'No action items found.'"
  )
  return chain.invoke(transcript)
```

#### Prompt Engineering Mechanics:
1. **Explicit Schema Definition**: Demands three explicit attributes: *Task description*, *Owner*, *Deadline*.
2. **Negative Constraint Handling**: Instructs *"else write 'Not specified'"* and *"If none found say 'No action items found.'"*. This prevents the LLM from inventing hallucinated owners or fake deadlines when none were mentioned in the audio!
3. **Reusable Chain Construction**:
   ```python
   def build_chain(system_prompt: str):
     return (
         RunnablePassthrough()
         | RunnableLambda(lambda x: {"text": x})
         | ChatPromptTemplate.from_messages(
             [("system", system_prompt), ("human", "{text}")]
         )
         | llm
         | StrOutputParser()
     )
```
   This encapsulates the LCEL pipeline cleanly, allowing `extract_key_decisions` and `extract_questions` to reuse the identical chain factory with different system prompts.

---

### Q13: How does the system prompt prevent hallucinations during meeting QA?

#### Answer
In `core/rag_engine.py` lines 30–39:
```markdown
You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}
```

#### Why this prompt structure works:
1. **Scope Restriction ("based ONLY on...")**: Binds the LLM's attention to the `{context}` slot, penalizing it for retrieving facts from its pretraining memory weights.
2. **Explicit Fallback Exit Condition**: Without an explicit fallback command, LLMs feel pressured to answer and will extrapolate plausible-sounding falsehoods. Providing an explicit escape phrase (*"I could not find this information..."*) drops hallucination rates on out-of-domain questions to near zero.
3. **Attribution Constraint**: Instructing the model to attribute quotes encourages direct factual grounding.

---

## Module 4: Codebase Line-by-Line Technical Breakdown

### Q14: Analyze `utils/audio_processor.py` line-by-line. Explain `yt-dlp` and `pydub` transformations.

#### Answer

```python
 8: def download_youtube_audio(url :str) ->str:
 9:     output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
10:     ydl_opts = {
11:         "format": "bestaudio/best",
12:         "outtmpl": output_path,
13:         "postprocessors": [
14:             {
15:                 "key": "FFmpegExtractAudio",
16:                 "preferredcodec": "wav",
17:                 "preferredquality": "192",
18:             }
19:         ],
20:         "quiet": True,
21:     }
22:     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
23:         info = ydl.extract_info(url, download=True)
24:         filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
25:     return filename
```
- **Lines 10-21**: Configures `yt-dlp` options. It selects `bestaudio/best`, directs output to `DOWNLOAD_DIR`, and attaches an FFmpeg post-processor to decode webm/opus/m4a streams into standard linear PCM WAV audio.
- **Line 24**: Replaces `.webm` / `.m4a` extensions with `.wav` to match FFmpeg's output file path.

```python
29: def convert_to_wav(input_path: str) -> str:
30:     output_path = os.path.splitext(input_path)[0] + "_converted.wav"
31:     audio = AudioSegment.from_file(input_path)
32:     audio = audio.set_channels(1).set_frame_rate(16000)
33:     audio.export(output_path, format="wav")
34:     return output_path
```
- **Line 31**: PyDub auto-detects containers (MP4, MKV, AVI, MP3, FLAC) using underlying FFmpeg bindings.
- **Line 32**: Resamples the audio to **1-channel Mono** at **16,000 Hz**, fulfilling Whisper's input specifications.

```python
39: def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
40:     audio = AudioSegment.from_wav(wav_path)
41:     chunk_ms = chunk_minutes * 60 * 1000 
42:     chunks = []
45:     for i, start in enumerate(range(0,len(audio),chunk_ms)):
46:         chunk = audio[start : start + chunk_ms]
47:         chunk_path = f"{wav_path}_chunk_{i}.wav"
48:         chunk.export(chunk_path , format = "wav")
49:         chunks.append(chunk_path)
52:     return chunks
```
- **Lines 40-52**: PyDub indexes audio in milliseconds. It slices 10-minute ($600,000\text{ ms}$) blocks and writes them out as indexed `.wav` files.

---

### Q15: Analyze `core/transcriber.py` line-by-line. Explain model caching and language routing.

#### Answer

```python
18: _model = None
19: 
21: def load_model():
22:     global _model  
23:     if _model is None: 
24:         print(f"Loading Whisper model: {WHISPER_MODEL} ...")
25:         _model = whisper.load_model(WHISPER_MODEL) 
26:         print("Whisper model loaded.")
27:     return _model 
```
- **Singleton Caching Pattern**: Loading a Whisper model weights file ($\approx 500\text{MB}$ for `small`) from disk into RAM/VRAM takes several seconds. Storing it in module-level `_model` ensures the weights are loaded **exactly once** during the application lifecycle.

```python
95: def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
96:     if language.lower() == "hinglish":
97:         return transcribe_chunk_sarvam(chunk_path)
98:     return transcribe_chunk_whisper(chunk_path)
```
- **Dynamic Model Dispatcher**: Routes audio chunks based on user language selection between local Whisper inference or Sarvam AI cloud translation.

---

### Q16: Analyze `core/vector_store.py` line-by-line. Explain collection persistence.

#### Answer

```python
 7: CHROMA_DIR = "vector_db"
 8: COLLECTION_NAME = "meeting_transcript"
 9: EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
...
31:     embeddings = get_embeddings()
32:     vector_store = Chroma.from_documents(
33:         documents= docs,
34:         embedding=embeddings,
35:         collection_name=COLLECTION_NAME,
36:         persist_directory=CHROMA_DIR
37:     )
```
- **`persist_directory=CHROMA_DIR`**: Instructs Chroma to write its internal parquet data files, SQLite database (`chroma.sqlite3`), and HNSW graph binaries to the `vector_db/` folder on disk.
- When `load_vector_store()` is called later during chat interactions, it reads directly from `vector_db/` without re-embedding the entire meeting transcript.

---

### Q17: Analyze `core/rag_engine.py` line-by-line. Identify potential bugs and design patterns.

#### Answer

#### Line 58–61 Bug Analysis:
```python
def load_rag_chain():
  vector_store = load_vector_store()
  retriver = get_retriever()  # <-- BUG HERE!
```
- Look at the definition in `core/vector_store.py`:
  ```python
  def get_retriever(vector_store: Chroma, k: int = 4):
    ...
```
- In `core/vector_store.py`, `vector_store` is a **required positional parameter**!
- In `core/rag_engine.py` line 60, calling `get_retriever()` with zero arguments will raise a `TypeError: get_retriever() missing 1 required positional argument: 'vector_store'`.
- **The Fix**:
  ```python
  def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
```

#### Architectural Strengths:
- `RunnableLambda(format_docs)` neatly bridges the LangChain `Document` list returned by the vector store into a raw string suitable for prompt interpolation.

---

### Q18: Analyze `core/summarizer.py` line-by-line. Explain prompt chaining.

#### Answer
In `core/summarizer.py`:
```python
50:     combined_chain = (
51:         RunnablePassthrough()
            | RunnableLambda(lambda x:{"text":x}) 
            | combined_prompt 
            | llm 
            | StrOutputParser()
52:     )
```
- **Line 51**: `RunnableLambda(lambda x: {"text": x})` wraps the concatenated string of chunk summaries into a dictionary matching the prompt's input variable `{text}`.
- This represents functional composition:
  $$\text{Output} = \text{StrOutputParser}(\text{LLM}(\text{Prompt}(\text{Input})))$$

---

### Q19: Analyze `main.py`. Explain the CLI execution flow and interactive chat loop.

#### Answer
In `main.py`:
- Lines 11–38 execute `run_pipeline(source, language)` sequentially:
  $$\text{Audio Ingestion} \rightarrow \text{Transcription} \rightarrow \text{Title} \rightarrow \text{Summary} \rightarrow \text{Extraction} \rightarrow \text{Vector Store Indexing}$$
- Lines 56–65 implement an **interactive CLI Read-Eval-Print Loop (REPL)**:
  ```python
  while True:
    question = input("You: ").strip()
    if question.lower() in ["exit", "quit", "q"]:
      break
    answer = ask_question(rag_chain, question)
    print(f"\nAssistant: {answer}\n")
```
  The compiled `rag_chain` remains held in memory across turns, allowing the user to ask multiple consecutive questions about the meeting.

---

## Module 5: Advanced System Scaling, Evaluation & Productionization

### Q20: How do you evaluate transcription (WER) and meeting summaries (ROUGE, BLEU, RAGAS)?

#### Answer

#### 1. ASR Evaluation: Word Error Rate (WER)
$$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Total Reference Words}}$$
- Evaluated against ground-truth human transcriptions using Levenshtein distance on words. A WER $< 5\%$ indicates human-parity accuracy.

#### 2. Summary Evaluation: ROUGE & BLEU
- **ROUGE-N**: Measures n-gram recall between the generated summary and gold-standard human summaries.
- **ROUGE-L**: Measures Longest Common Subsequence (LCS) to capture sentence structure.
- **BLEU**: Measures n-gram precision (primarily used in translation).

#### 3. RAG Evaluation with RAGAS:
- **Context Precision**: Did Chroma retrieve the exact meeting chunks containing the answer?
- **Faithfulness**: Did Mistral answer strictly using the retrieved chunks without hallucinating?
- **Answer Relevance**: Did the answer directly resolve the user's question?

---

### Q21: How would you optimize Whisper latency for production? Compare FP32, FP16, INT8, and Faster-Whisper.

#### Answer

| Implementation | Precision | Inference Engine | Relative Speed | VRAM Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Stock OpenAI Whisper** | FP32 / FP16 | PyTorch | $1.0\times$ (Baseline) | $\sim 2.0\text{ GB}$ (`small`) |
| **Faster-Whisper** | FP16 / INT8 | CTranslate2 | **$4.0\times$ Faster** | $\sim 600\text{ MB}$ |
| **Whisper.cpp** | INT4 / INT8 | Pure C/C++ (CPU optimized) | Fast on CPU / Apple Metal | Ultra-low memory |
| **TensorRT-LLM Whisper** | FP8 / INT8 | NVIDIA TensorRT | **$6.0\times$ Faster** | GPU-accelerated enterprise |

#### Upgrading to `faster-whisper`:
```python
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav", beam_size=5)
transcript = " ".join([segment.text for segment in segments])
```
Using `faster-whisper` cuts transcription time by **75%** by leveraging optimized matrix kernels in CTranslate2!

---

### Q22: How would you convert this batch transcription system into a real-time live streaming assistant?

#### Answer
Currently, the system is **batch-oriented** (you must wait for the entire audio file to finish before processing).

```
[User Mic / WebRTC Stream]
           │
           ▼ (Audio Chunks: 200ms PCM frames)
[WebSocket Gateway (FastAPI)]
           │
           ▼ (Audio Stream Buffer)
[Voice Activity Detection (VAD - Silero)] ──► Strips background silence
           │
           ▼ (Speech detected: ~3s buffer)
[Streaming ASR Engine (Whisper Streaming / Deepgram / AssemblyAI)]
           │
           ▼ (Partial Transcripts via WebSocket)
[Live Token Stream to Frontend React / Next.js]
```

1. **WebRTC / WebSocket Ingestion**: Stream binary audio chunks directly from the browser microphone.
2. **Silero VAD (Voice Activity Detection)**: Discards silence chunks so Whisper only processes speech segments.
3. **Local Agreement Decoding**: Re-transcribes sliding windows and locks in tokens that remain consistent across consecutive windows.

---

### Q23: What happens when a meeting is 4 hours long? How does the system prevent context overflow?

#### Answer
A 4-hour meeting produces $\approx 35,000\text{ words}$ ($\approx 45,000\text{ tokens}$).
1. **ASR Scaling**:
   - `utils/audio_processor.py` slices audio into 10-minute blocks. The system processes each block sequentially and frees audio memory buffers, easily transcribing 4 hours without memory issues.
2. **Summarization Scaling**:
   - The **Map-Reduce** pattern in `core/summarizer.py` scales linearly:
     - 45,000 tokens $\rightarrow$ 15 chunks $\rightarrow$ 15 parallel map summaries ($\approx 15 \times 100\text{ tokens} = 1,500\text{ tokens}$).
     - The reduce prompt receives only 1,500 tokens, easily fitting inside Mistral's context window!
3. **RAG Scaling**:
   - The vector store chunks the transcript into 500-character segments. Whether there are 100 chunks or 10,000 chunks, the retriever only pulls the top $k=4$ most relevant chunks ($\approx 2,000\text{ characters}$), keeping the LLM prompt cost small and constant!

---

### Q24: How do you handle sensitive PII, passwords, and confidential discussion in meeting transcripts?

#### Answer
Corporate meetings often mention employee salaries, credit cards, or passwords.
1. **Presidio / Regex PII Masking**:
   - Run Microsoft Presidio or local Named Entity Recognition (NER) models over the transcript before storing it in ChromaDB or sending it to third-party LLM APIs:
     ```python
     from presidio_analyzer import AnalyzerEngine
     from presidio_anonymizer import AnonymizerEngine

     analyzer = AnalyzerEngine()
     results = analyzer.analyze(text=transcript, language="en")
     anonymized_transcript = AnonymizerEngine().anonymize(
         text=transcript, analyzer_results=results
     )
```
2. **Zero Data Retention (ZDR)**:
   - Ensure commercial API agreements (OpenAI, Mistral, Sarvam) enforce Zero Data Retention so meeting audio and text are never used for training.
3. **Ephemeral Vector Store**:
   - Purge the `vector_db/` directory after the user session terminates, or encrypt sqlite files at rest with AES-256.

---

### Q25: Design an enterprise production architecture with FastAPI, Celery, Redis, and WebSockets.

#### Answer

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER (Next.js 14)                       │
│      - WebRTC / Audio Upload          - Real-Time Chat & Progress      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / WebSocket (WSS)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY (FastAPI)                           │
│      - JWT Auth                       - Upload S3 Presigned URLs       │
│      - Enqueues Jobs to Redis         - SSE / WebSocket Live Events    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
            [Redis Task Queue]          [PostgreSQL Database]
                        │               (Metadata, Users, Chats)
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED WORKER POOL (Celery)                    │
│                                                                        │
│   Worker Cluster A (GPU Nodes):                                        │
│   - Worker 1: FFmpeg Audio Normalization (16kHz Mono)                 │
│   - Worker 2: Faster-Whisper ASR Inference (CUDA FP16)                │
│   - Worker 3: Sarvam Indic Speech Translation API                      │
│                                                                        │
│   Worker Cluster B (CPU / LLM Nodes):                                  │
│   - Worker 4: ChromaDB HNSW Indexing                                   │
│   - Worker 5: Mistral Map-Reduce Summarization & RAG Q&A               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         PERSISTENCE & STORAGE                          │
│   - AWS S3 (Raw Audio & Transcripts)                                   │
│   - ChromaDB / Milvus / Qdrant (Cluster Vector DB)                     │
└────────────────────────────────────────────────────────────────────────┘
```

---
*Created as part of the AI Video Assistant technical documentation.*
