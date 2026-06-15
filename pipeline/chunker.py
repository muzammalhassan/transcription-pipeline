"""
AudioChunker — split long audio into overlapping windows.

Why chunking?
─────────────
Whisper's context window is 30 seconds of audio (480 000 samples at 16 kHz).
Passing a 2-hour recording naïvely would either:
  a) fail outright, or
  b) silently truncate everything after 30 s.

Strategy
────────
1. Fixed-size windows  (default 30 s) with a configurable *overlap* (default 2 s).
2. The overlap prevents words at chunk boundaries from being cut off mid-utterance.
3. After each chunk is transcribed we discard segments that fall inside the
   *overlap zone* of the previous chunk — keeping one clean, non-redundant copy.

Alternative strategies considered
───────────────────────────────────
• VAD-aware splitting (e.g. silero-vad): cleaner boundaries but adds a dependency
  and latency. Worth adding in a v2 when accuracy at boundaries matters more.
• Sentence-boundary splitting: requires a first-pass transcript — chicken-and-egg.
• Batch-of-chunks parallelism: straightforward to add; the chunker outputs
  independent files so they can be transcribed concurrently (see pipeline.py).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Whisper's hard context limit; keep default chunk_duration ≤ this.
WHISPER_MAX_SECONDS = 30.0


@dataclass
class AudioChunk:
    """Represents one slice of a potentially-longer audio file."""
    index: int
    start_time: float    # seconds in the original timeline
    end_time: float      # seconds in the original timeline
    file_path: str       # path to the exported chunk WAV

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return (
            f"AudioChunk(idx={self.index}, "
            f"{self.start_time:.1f}s–{self.end_time:.1f}s, "
            f"path={Path(self.file_path).name})"
        )


class AudioChunker:
    """
    Splits a normalised WAV file into a list of :class:`AudioChunk` objects.

    Parameters
    ----------
    chunk_duration
        Length of each window in seconds.  Must be ≤ 30 s for Whisper.
    overlap
        How many seconds each successive window rewinds before starting.
        Prevents words at the boundary from being silently dropped.
    output_dir
        Directory where chunk WAV files are written.
    """

    def __init__(
        self,
        chunk_duration: float = WHISPER_MAX_SECONDS,
        overlap: float = 2.0,
        output_dir: str = "/tmp",
    ) -> None:
        if chunk_duration > WHISPER_MAX_SECONDS:
            raise ValueError(
                f"chunk_duration ({chunk_duration}s) exceeds Whisper's "
                f"30 s context limit."
            )
        if overlap >= chunk_duration:
            raise ValueError("overlap must be smaller than chunk_duration")

        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.output_dir = output_dir

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def needs_chunking(self, audio_path: str) -> bool:
        """True when the file is longer than one chunk window."""
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0 > self.chunk_duration

    def split(self, audio_path: str) -> List[AudioChunk]:
        """
        Slice *audio_path* into overlapping windows.

        If the file is shorter than ``chunk_duration`` a single chunk pointing
        at the original file is returned (no disk I/O).
        """
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path)
        total_ms = len(audio)
        total_s = total_ms / 1000.0

        if total_s <= self.chunk_duration:
            logger.debug("File is short enough — no chunking needed")
            return [AudioChunk(0, 0.0, total_s, audio_path)]

        chunk_ms = int(self.chunk_duration * 1000)
        overlap_ms = int(self.overlap * 1000)
        step_ms = chunk_ms - overlap_ms  # advance per window

        chunks: List[AudioChunk] = []
        start_ms = 0

        while start_ms < total_ms:
            end_ms = min(start_ms + chunk_ms, total_ms)
            idx = len(chunks)

            chunk_path = os.path.join(self.output_dir, f"chunk_{idx:04d}.wav")
            audio[start_ms:end_ms].export(chunk_path, format="wav")

            chunk = AudioChunk(
                index=idx,
                start_time=start_ms / 1000.0,
                end_time=end_ms / 1000.0,
                file_path=chunk_path,
            )
            chunks.append(chunk)
            logger.debug("  %s", chunk)

            if end_ms >= total_ms:
                break
            start_ms += step_ms

        logger.info("Split %.1fs audio into %d chunks", total_s, len(chunks))
        return chunks

    def cleanup(self, chunks: List[AudioChunk], original_path: str) -> None:
        """Delete temporary chunk files (but never the original)."""
        for chunk in chunks:
            if chunk.file_path != original_path and os.path.exists(chunk.file_path):
                os.remove(chunk.file_path)
                logger.debug("Deleted temp chunk: %s", chunk.file_path)
