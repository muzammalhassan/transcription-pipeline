# Transcription Pipeline

A production-oriented speech-to-text pipeline that accepts any audio format,
handles long recordings via overlapping chunking, and returns timestamped
segments ready for downstream consumption.

---

## Architecture

```
  Audio file (any format)
       │
       ▼
┌──────────────────┐
│  AudioProcessor  │  ① Decode any format → 16 kHz mono PCM WAV
│  (pydub+ffmpeg)  │     Fast-path if already normalised
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  AudioChunker    │  ② Split long files into 30 s windows w/ 2 s overlap
└────────┬─────────┘     Short files pass through as a single "chunk"
         │
         ▼  (one chunk at a time)
┌──────────────────┐
│ TranscriptionEng │  ③ Whisper inference → segments + word timestamps
│  (openai-whisper)│     time_offset shifts chunk timestamps → global timeline
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Dedup & Merge    │  ④ Drop overlap-zone duplicates, re-index, merge
└────────┬─────────┘
         │
         ▼
  TranscriptionResult
  ├── full_text
  ├── segments[]: { id, text, start, end, words[], confidence }
  ├── to_json()
  ├── to_srt()   ← subtitle-ready
  └── to_vtt()   ← HTML5 <track>-ready
```

---

## Key Engineering Decisions

### 1. Audio normalisation: pydub + ffmpeg

**Why not soundfile / librosa?**
Both are excellent but tied to libsndfile, which doesn't support MP3, AAC, or
OPUS without patching. `pydub` delegates all codec work to `ffmpeg`, giving
us ~100 input formats with zero C extension code. It adds a subprocess call
per file, which is fine because audio I/O is not the bottleneck.

**Target: 16 kHz mono PCM**
All major speech models were trained on 16 kHz mono. Resampling *before*
the model is safer (explicit, auditable, cacheable) than relying on the model
to silently resample internally.

### 2. Long audio: overlapping fixed-size windows

| Option | Pros | Cons |
|---|---|---|
| **Fixed 30 s windows + overlap** ← used | Simple, no extra deps | Rare mid-word cuts |
| VAD-aware boundaries | Cleaner cuts | Adds silero-vad dep + latency |
| Sentence-boundary split | Best quality | Chicken-and-egg: needs a first-pass |

The 2 s overlap ensures that a word spanning a chunk boundary is heard
in full by at least one chunk. The deduplication step then discards the
redundant copy using a simple `start_time ≥ (last_end − overlap/2)` filter.

**Why 30 s?** Whisper's context window is exactly 480 000 samples at 16 kHz =
30 s. Longer chunks silently truncate; shorter chunks miss cross-sentence
context. 30 s is the sweet spot.

### 3. Transcription engine: OpenAI Whisper

Whisper is open-source (MIT), multilingual, and produces both segment-level
and word-level timestamps out of the box. It supports automatic language
detection, making it a practical default.

**Model size guidance:**
- `tiny` / `base` — local development, low-RAM machines
- `small` — a solid CI / staging default
- `medium` / `large` — production accuracy requirement

**Production alternative: `faster-whisper`**
`faster-whisper` uses CTranslate2 to achieve 2–4× speedup with the same
accuracy, and supports 4-bit quantisation for further memory reduction.
The `TranscriptionEngine` interface is identical; swap the import.

### 4. Typed models, not raw dicts

`TranscriptionSegment` and `TranscriptionResult` are plain dataclasses.
Downstream consumers get typed objects (autocomplete, static analysis) and
can serialise to JSON / SRT / VTT in one call. Swapping to Pydantic later
requires only changing the base class.

### 5. Dependency injection for testability

`TranscriptionPipeline` accepts an optional `engine=` parameter. Tests and
demos inject `MockTranscriptionEngine`, which generates realistic
segment/word data without downloading a model. The real engine is loaded
lazily on first use.

---

## Quick Start

### Install

```bash
# System dependency (required for non-WAV formats)
brew install ffmpeg          # macOS
apt-get install ffmpeg       # Ubuntu/Debian

# Python packages
pip install pydub openai-whisper
```

### Run the demo (no Whisper needed)

```bash
python demo.py          # 10 s synthetic audio
python demo.py --long   # 75 s audio — demonstrates chunking
```

### Use real Whisper

```python
from pipeline import TranscriptionPipeline

pipeline = TranscriptionPipeline(
    model_size="base",   # tiny | base | small | medium | large
    device="cpu",        # or "cuda"
    language="en",       # None = auto-detect
)

result = pipeline.transcribe("interview.mp3")

print(result.full_text)
print(result.to_json())     # structured JSON with timestamps
print(result.to_srt())      # subtitle file
```

### Output schema

```json
{
  "audio_file": "interview.mp3",
  "language": "en",
  "duration_seconds": 75.0,
  "model": "base",
  "word_count": 183,
  "full_text": "The quick brown fox…",
  "segments": [
    {
      "id": 0,
      "text": "The quick brown fox jumps over the lazy dog.",
      "start": 0.0,
      "end": 3.24,
      "duration": 3.24,
      "confidence": -0.12,
      "words": [
        { "text": "The",   "start": 0.00, "end": 0.24, "confidence": 0.99 },
        { "text": "quick", "start": 0.26, "end": 0.56, "confidence": 0.97 }
      ]
    }
  ]
}
```

---

## Run Tests

```bash
python tests/test_models.py     # 10 tests — model serialisation & exports
python tests/test_chunker.py    # 10 tests — chunking logic & edge cases
python tests/test_pipeline.py   # 14 tests — full integration (mock engine)
```

All 34 tests pass with only `pydub` installed. No GPU or internet required.

---

## Supported Audio Formats

| Format | Extension | Notes |
|---|---|---|
| WAV (PCM) | `.wav` | Native; no ffmpeg needed |
| MP3 | `.mp3` | Requires ffmpeg |
| AAC / M4A | `.aac`, `.m4a` | Requires ffmpeg |
| FLAC | `.flac` | Requires ffmpeg |
| OGG Vorbis | `.ogg` | Requires ffmpeg |
| OPUS | `.opus` | Requires ffmpeg |
| WebM | `.webm` | Requires ffmpeg |
| AIFF | `.aiff` | Requires ffmpeg |
| WMA | `.wma` | Requires ffmpeg |

---

## Extension Points

| Feature | Where to add |
|---|---|
| Speaker diarisation | Post-processing step after `pipeline.transcribe()` (e.g. `pyannote.audio`) |
| VAD-aware chunking | Replace `AudioChunker.split()` with a silero-VAD-guided boundary finder |
| Parallel chunk transcription | `concurrent.futures.ThreadPoolExecutor` over the `chunks` loop in `pipeline.py` |
| Streaming / real-time | Replace `AudioChunker` with a ring-buffer fed from a microphone stream |
| Confidence-based retry | After merge, flag segments with `avg_logprob < threshold` and re-transcribe with a larger model |
| Output to DB / queue | Wrap `pipeline.transcribe()` in a worker that pushes `result.to_dict()` to Kafka / Postgres |

---

## Project Structure

```
transcription_pipeline/
├── pipeline/
│   ├── models.py           # TranscriptionResult, TranscriptionSegment, Word
│   ├── audio_processor.py  # Format conversion → 16 kHz mono WAV
│   ├── chunker.py          # Overlapping window chunking for long audio
│   ├── transcriber.py      # Whisper wrapper with timestamp extraction
│   └── pipeline.py         # Orchestrates all steps
├── mock/
│   └── mock_engine.py      # Drop-in engine for tests / demos
├── tests/
│   ├── test_models.py
│   ├── test_chunker.py
│   └── test_pipeline.py
├── demo.py
└── requirements.txt
```
