# Agentic Video Understanding System

**[→ Visual walkthrough of the system](https://akshit9162.github.io/Agentic-Video-Understanding-System/)**

Upload a long video or paste a YouTube URL and you get back:
- a playable summary video with the original audio
- a transcript
- a question-answering endpoint that answers with timestamps

Keyframes are chosen by a two-level reinforcement-learning policy. Everything runs as async jobs behind FastAPI, Celery and Redis, and progress streams live over Server-Sent Events.

## Pipeline

```
upload / YouTube URL
   └─▶ Celery job ──▶ CLIP ViT-B/32 frame embeddings
                  ──▶ RL policy refines 25 keyframes
                  ──▶ PySceneDetect maps keyframes to scenes ──▶ OpenCV clips
                  ──▶ FFmpeg merges the original audio ──▶ summary.mp4
                  ──▶ Faster-Whisper transcript (SHA-256 cache)
                  ──▶ LlamaIndex index over timestamped chunks ──▶ /rag/query
```

## Keyframe selection

Choosing K frames out of N directly is a combinatorial search, so the agent edits a current selection one frame at a time:

- **Horizontal policy:** picks *which* of the K = 25 selected frames to move.
- **Vertical policy:** picks *how* to move it: −1, +1, −5 or +5 frames.
- **Reward:** `tanh(0.4·importance + 0.3·diversity + 0.3·coverage)` on CLIP features.
- **Inference:** deterministic rollout. The summary is capped at 20% of the original duration.

**What measurement showed.** In deterministic rollout, the trained policy collapses onto a small number of keyframe slots. Its CLIP-distance importance signal scores slightly below random against TVSum human annotators, because it rewards outlier frames over representative ones.

That result led to a from-scratch reproduction of the published method this policy is based on: [prlvs-reproduction](https://github.com/akshit9162/prlvs-reproduction).

## Grounded Q&A over the transcript

- Whisper segments are merged into chunks of up to 200 words. Each chunk keeps its start and end timestamps.
- Chunks are embedded with `BAAI/bge-small-en-v1.5` and persisted as one LlamaIndex index per video.
- `POST /rag/query` returns the top matching chunks with timestamp ranges, plus a synthesised answer when an LLM is configured.
- The LLM is Claude when `ANTHROPIC_API_KEY` is set, with OpenAI as a fallback.
- `POST /agent/transcript-insights` runs a LangChain chain (prompt → LLM → parser) that returns a 3–5 bullet summary. Without an API key it falls back to an extractive summary.

## API

| Endpoint | Purpose |
|---|---|
| `POST /summarize/upload` | Queue a summary job for an uploaded file |
| `POST /summarize/youtube` | Queue a job for a YouTube URL |
| `GET /tasks/{id}` | Job state and result |
| `GET /tasks/{id}/stream` | Server-Sent Events progress stream |
| `POST /rag/query` | Timestamp-grounded question answering |
| `POST /agent/transcript-insights` | LLM bullet summary of the transcript |

A Streamlit front end (`app/streamlit_app.py`) uses this API. The Celery worker uses the single-process `solo` pool, because native video libraries crash under fork-based multiprocessing on macOS.

## Run

```bash
docker compose up --build             # API, worker and Redis

# or without Docker
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
redis-server &
celery -A celery_app worker --pool=solo &
uvicorn app.api:app --reload
streamlit run app/streamlit_app.py
```

FFmpeg must be on `PATH`, or set `FFMPEG_PATH`. CI (GitHub Actions) runs the smoke tests in `tests/` on CPU-only PyTorch.

## Layout

| Path | Contents |
|---|---|
| `src/env.py`, `src/model.py`, `src/train.py` | RL environment, policies, training |
| `src/inference.py`, `src/scene_detection.py`, `src/video_utils.py` | Summary generation |
| `src/speech_summary.py`, `src/rag.py`, `src/agent_pipeline.py` | Transcript, RAG, LLM insights |
| `app/api.py`, `tasks.py`, `celery_app.py` | FastAPI service and Celery jobs |
| `docs/` | GitHub Pages walkthrough |
