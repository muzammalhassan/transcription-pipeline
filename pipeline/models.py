"""
Data models for the transcription pipeline.

Design notes:
  - Plain dataclasses keep the models framework-agnostic
    (easy to swap in Pydantic, attrs, etc. later).
  - TranscriptionSegment mirrors Whisper's native output so
    the real and mock engines speak the same language.
  - Word-level timestamps are optional; not all engines support them.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Word:
    """A single word with start/end timestamps and an optional confidence score."""
    text: str
    start: float        # seconds
    end: float          # seconds
    confidence: Optional[float] = None  # 0.0 – 1.0; None when engine doesn't provide it


@dataclass
class TranscriptionSegment:
    """
    A contiguous speech segment (roughly one sentence / phrase).
    Segments are the primary unit returned to callers.
    """
    id: int
    text: str
    start: float        # seconds from audio start
    end: float          # seconds from audio start
    words: List[Word] = field(default_factory=list)
    confidence: Optional[float] = None   # avg log-prob from Whisper; higher = better

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": self.duration,
            "words": [
                {
                    "text": w.text,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(w.confidence, 3) if w.confidence is not None else None,
                }
                for w in self.words
            ],
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
        }


@dataclass
class TranscriptionResult:
    """
    Top-level container returned by the pipeline.
    Holds all segments plus metadata about the source file and model used.
    """
    segments: List[TranscriptionSegment]
    language: str
    duration: float         # total audio duration in seconds
    model: str              # e.g. "whisper-base", "mock"
    audio_file: str         # original filename (for traceability)

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def full_text(self) -> str:
        """Concatenated transcript of all segments."""
        return " ".join(s.text.strip() for s in self.segments)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "audio_file": self.audio_file,
            "language": self.language,
            "duration_seconds": round(self.duration, 2),
            "model": self.model,
            "word_count": self.word_count,
            "full_text": self.full_text,
            "segments": [s.to_dict() for s in self.segments],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_srt(self) -> str:
        """Export in SubRip (.srt) subtitle format — useful for video captioning."""
        lines = []
        for seg in self.segments:
            lines.append(str(seg.id + 1))
            lines.append(f"{_srt_ts(seg.start)} --> {_srt_ts(seg.end)}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """Export in WebVTT format — ready for HTML5 <track> elements."""
        lines = ["WEBVTT", ""]
        for seg in self.segments:
            lines.append(f"{_vtt_ts(seg.start)} --> {_vtt_ts(seg.end)}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _srt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm (SRT standard)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm (VTT standard)."""
    return _srt_ts(seconds).replace(",", ".")
