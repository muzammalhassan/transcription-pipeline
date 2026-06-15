"""
TranscriptionPipeline — the public face of the library.

Data flow
─────────

  ┌─────────────┐   normalize   ┌───────────────┐   split    ┌──────────────┐
  │  Audio file │ ──────────── ▶│  16kHz mono   │ ─────────▶ │ AudioChunk[] │
  │  (any fmt)  │               │  WAV (tmp)    │            └──────┬───────┘
  └─────────────┘               └───────────────┘                   │
                                                        transcribe each chunk
                                                                   │
                                                           ┌────────▼────────┐
                                                           │  merge + dedup  │
                                                           └────────┬────────┘
                                                                    │
                                                        ┌───────────▼──────────┐
                                                        │  TranscriptionResult │
                                                        └──────────────────────┘

Deduplication strategy
──────────────────────
Each chunk overlaps the previous by `chunk_overlap` seconds.  After
transcribing a chunk we drop any new segments whose *start time* falls
within the overlap window of the last retained segment.  This is intentionally
simple — a more sophisticated approach (e.g. edit-distance alignment of the
overlapping text) would be more robust but adds significant complexity.
"""

from __future__ import annotations

import logging
import tempfile
import os
from pathlib import Path
from typing import List, Optional

from .audio_processor import AudioProcessor
from .chunker import AudioChunker
from .transcriber import TranscriptionEngine
from .models import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    """
    End-to-end speech → text pipeline with timestamped segments.

    Parameters
    ----------
    model_size
        Whisper model size (``"tiny"`` … ``"large"``).  Defaults to ``"base"``.
    device
        ``"cpu"`` or ``"cuda"``.
    language
        Force a language (BCP-47) or ``None`` for auto-detection.
    chunk_duration
        Window size in seconds (≤ 30 for Whisper).
    chunk_overlap
        Seconds of context carried over between consecutive windows.
    engine
        Optionally inject a custom engine (useful for testing / mocking).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: Optional[str] = None,
        chunk_duration: float = 30.0,
        chunk_overlap: float = 2.0,
        engine: Optional[TranscriptionEngine] = None,
    ) -> None:
        self.processor = AudioProcessor()
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.engine = engine or TranscriptionEngine(
            model_size=model_size,
            device=device,
            language=language,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def transcribe(self, audio_file: str) -> TranscriptionResult:
        """
        Full pipeline: load → normalise → chunk → transcribe → merge.

        Parameters
        ----------
        audio_file
            Path to any supported audio file (WAV, MP3, M4A, FLAC, …).

        Returns
        -------
        :class:`~pipeline.models.TranscriptionResult`
        """
        logger.info("=== TranscriptionPipeline.transcribe(%s) ===", audio_file)

        # All temporary files live inside a single tmpdir that is cleaned up
        # automatically when the `with` block exits — no leaking temp files.
        with tempfile.TemporaryDirectory(prefix="tp_") as tmpdir:
            # ── Step 1: Normalise ──────────────────────────────────────
            norm_path, audio_info = self.processor.normalize(
                audio_file,
                output_path=os.path.join(tmpdir, "normalised.wav"),
            )
            duration = audio_info["duration_seconds"]
            logger.info("Duration: %.1f s", duration)

            # ── Step 2: Chunk ──────────────────────────────────────────
            chunker = AudioChunker(
                chunk_duration=self.chunk_duration,
                overlap=self.chunk_overlap,
                output_dir=tmpdir,
            )
            chunks = chunker.split(norm_path)

            # ── Step 3: Transcribe ─────────────────────────────────────
            all_segments: List[TranscriptionSegment] = []
            detected_language = "unknown"

            for chunk in chunks:
                segs, lang = self.engine.transcribe_file(
                    chunk.file_path,
                    time_offset=chunk.start_time,
                )
                detected_language = lang  # last chunk wins (they should agree)

                # De-duplicate overlap zone
                if chunk.index > 0 and all_segments:
                    segs = self._dedup_overlap(all_segments, segs)

                all_segments.extend(segs)

            # ── Step 4: Re-index ───────────────────────────────────────
            for i, seg in enumerate(all_segments):
                seg.id = i

            logger.info(
                "=== Done: %d segment(s) | lang=%s | %.1f s ===",
                len(all_segments),
                detected_language,
                duration,
            )
            return TranscriptionResult(
                segments=all_segments,
                language=detected_language,
                duration=duration,
                model=self.engine.model_size,
                audio_file=Path(audio_file).name,
            )

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _dedup_overlap(
        self,
        existing: List[TranscriptionSegment],
        new_segs: List[TranscriptionSegment],
    ) -> List[TranscriptionSegment]:
        """
        Drop segments from *new_segs* that are within the overlap window
        already covered by *existing*.

        We use a conservative boundary: any new segment whose start time is
        strictly before ``(last_end - overlap / 2)`` is considered a duplicate.
        The ``/ 2`` gives a margin that avoids discarding genuinely new content
        near the boundary.
        """
        if not existing or not new_segs:
            return new_segs

        last_end = existing[-1].end
        boundary = last_end - (self.chunk_overlap / 2.0)

        kept = [seg for seg in new_segs if seg.start >= boundary]
        dropped = len(new_segs) - len(kept)
        if dropped:
            logger.debug("Dedup: dropped %d overlapping segment(s)", dropped)
        return kept
