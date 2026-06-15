#!/usr/bin/env python3
"""
demo.py — end-to-end demonstration of the transcription pipeline.

Runs entirely with synthetic audio + mock engine (no GPU / Whisper download needed).
Swap MockTranscriptionEngine for TranscriptionEngine to use real Whisper.

Usage:
    python demo.py                    # short demo (10 s)
    python demo.py --long             # long audio demo (75 s, triggers chunking)
    python demo.py --audio /path/to/file.wav   # your own file
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import wave
import math

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.pipeline import TranscriptionPipeline
from mock.mock_engine import MockTranscriptionEngine


# ── Synthetic audio helpers ──────────────────────────────────────────────────

def _generate_tone_wav(path: str, duration_s: float, freq_hz: float = 440.0,
                       sample_rate: int = 16000) -> str:
    """Write a WAV file containing a simple sine tone (realistic size/format)."""
    n_samples = int(duration_s * sample_rate)
    amplitude = 16000
    samples = [
        int(amplitude * math.sin(2 * math.pi * freq_hz * t / sample_rate))
        for t in range(n_samples)
    ]
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return path


def _bar(value: float, max_value: float, width: int = 20) -> str:
    filled = int(round(value / max_value * width)) if max_value else 0
    return "█" * filled + "░" * (width - filled)


# ── Pretty printing helpers ──────────────────────────────────────────────────

def _print_banner(text: str) -> None:
    print()
    print("─" * 60)
    print(f"  {text}")
    print("─" * 60)


def _print_result(result) -> None:
    _print_banner("TRANSCRIPTION RESULT")
    print(f"  File      : {result.audio_file}")
    print(f"  Duration  : {result.duration:.1f}s")
    print(f"  Language  : {result.language}")
    print(f"  Model     : {result.model}")
    print(f"  Segments  : {len(result.segments)}")
    print(f"  Words     : {result.word_count}")
    print()
    print(f"  Full text :")
    print(f"  {result.full_text}")
    print()

    _print_banner("TIMESTAMPED SEGMENTS")
    for seg in result.segments:
        conf_bar = _bar(seg.confidence + 0.5 if seg.confidence else 0.5, 1.0)
        print(
            f"  [{seg.start:6.2f}s → {seg.end:6.2f}s]  "
            f"(conf {conf_bar})  {seg.text}"
        )
        if seg.words:
            word_str = "  ".join(
                f"\033[90m{w.text}@{w.start:.2f}s\033[0m" for w in seg.words[:5]
            )
            suffix = "  …" if len(seg.words) > 5 else ""
            print(f"    words: {word_str}{suffix}")
    print()

    _print_banner("EXPORT FORMATS")
    print("\n── SRT (first 3 entries) ──────────────────")
    srt_lines = result.to_srt().split("\n")
    print("\n".join(srt_lines[:12]))

    print("\n── VTT (first 3 entries) ──────────────────")
    vtt_lines = result.to_vtt().split("\n")
    print("\n".join(vtt_lines[:10]))

    print("\n── JSON (first segment) ───────────────────")
    d = result.to_dict()
    d["segments"] = d["segments"][:1]  # trim for readability
    print(json.dumps(d, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Transcription pipeline demo")
    parser.add_argument("--long", action="store_true",
                        help="Use a 75 s file to demonstrate chunking")
    parser.add_argument("--audio", metavar="PATH",
                        help="Path to a real audio file (WAV/MP3/etc.)")
    parser.add_argument("--chunk-duration", type=float, default=30.0,
                        help="Chunk window in seconds (default: 30)")
    args = parser.parse_args()

    print("\n\033[1m🎙  Transcription Pipeline Demo\033[0m")
    print("  Engine : MockTranscriptionEngine (swap for real Whisper)")
    print("  Chunking : ON (30 s windows, 2 s overlap)")

    with tempfile.TemporaryDirectory(prefix="tp_demo_") as tmpdir:
        if args.audio:
            audio_path = args.audio
            print(f"\n  Input  : {audio_path} (user-supplied)")
        else:
            duration = 75.0 if args.long else 10.0
            audio_path = os.path.join(tmpdir, f"demo_{int(duration)}s.wav")
            _generate_tone_wav(audio_path, duration)
            print(f"\n  Input  : synthetic {duration}s WAV tone → {audio_path}")

        pipeline = TranscriptionPipeline(
            chunk_duration=args.chunk_duration,
            chunk_overlap=2.0,
            engine=MockTranscriptionEngine(seed=42),
        )

        print("\n  Running pipeline …\n")
        result = pipeline.transcribe(audio_path)

    _print_result(result)

    print("\n\033[1m  ✓ Pipeline complete.\033[0m")
    print("  To use real Whisper: replace MockTranscriptionEngine with")
    print("  TranscriptionEngine(model_size='base') in pipeline.py.\n")


if __name__ == "__main__":
    main()
