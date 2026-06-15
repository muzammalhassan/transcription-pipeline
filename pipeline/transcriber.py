"""
TranscriptionEngine — thin wrapper around OpenAI Whisper.

Model size tradeoffs
────────────────────
  tiny    ~39 M params   ~32× realtime on CPU   English WER ~14 %
  base    ~74 M params   ~16× realtime on CPU   English WER ~9 %  ← default
  small   ~244 M params  ~6×  realtime on CPU   English WER ~7 %
  medium  ~769 M params  ~2×  realtime on CPU   English WER ~5 %
  large   ~1.5 B params  ~1×  realtime on CPU   English WER ~3 %

Production note
───────────────
For throughput-sensitive workloads consider `faster-whisper`
(github.com/SYSTRAN/faster-whisper) which uses CTranslate2 and achieves
2–4× speedup with identical accuracy, plus 4-bit quantisation support.
The engine interface below is identical; just swap the backend import.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .models import TranscriptionSegment, Word

logger = logging.getLogger(__name__)

VALID_MODELS = frozenset({
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large", "large-v2", "large-v3",
})


class TranscriptionEngine:
    """
    Wraps ``openai-whisper`` and returns typed :class:`TranscriptionSegment` objects.

    Parameters
    ----------
    model_size
        One of :data:`VALID_MODELS`.  ``"base"`` is a good default for a local CPU.
    device
        ``"cpu"`` or ``"cuda"``.  Auto-detected when set to ``"auto"``.
    language
        BCP-47 language code (e.g. ``"en"``, ``"fr"``).
        ``None`` enables automatic language detection.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: Optional[str] = None,
    ) -> None:
        if model_size not in VALID_MODELS:
            raise ValueError(f"Unknown model '{model_size}'. Valid: {sorted(VALID_MODELS)}")
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model = None  # loaded lazily on first use

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def transcribe_file(
        self,
        audio_path: str,
        time_offset: float = 0.0,
    ) -> Tuple[List[TranscriptionSegment], str]:
        """
        Transcribe one (chunk) audio file.

        Parameters
        ----------
        audio_path
            Path to a 16 kHz mono WAV file.
        time_offset
            Added to every timestamp so segment times reference the
            *original* audio timeline (not the chunk's local timeline).

        Returns
        -------
        (segments, language_code)
        """
        self._ensure_model_loaded()

        decode_opts: dict = {"word_timestamps": True, "verbose": False}
        if self.language:
            decode_opts["language"] = self.language

        logger.info("Transcribing %s (offset=+%.1fs)", audio_path, time_offset)
        result = self._model.transcribe(audio_path, **decode_opts)

        segments = self._parse_result(result, time_offset)
        detected_lang: str = result.get("language", "unknown")

        logger.info(
            "  → %d segment(s) | lang=%s", len(segments), detected_lang
        )
        return segments, detected_lang

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            ) from exc

        logger.info("Loading Whisper '%s' on %s …", self.model_size, self.device)
        self._model = whisper.load_model(self.model_size, device=self.device)
        logger.info("Model ready.")

    @staticmethod
    def _parse_result(
        result: dict,
        time_offset: float,
    ) -> List[TranscriptionSegment]:
        """Convert the raw Whisper dict into typed model objects."""
        segments: List[TranscriptionSegment] = []

        for i, raw_seg in enumerate(result.get("segments", [])):
            words: List[Word] = []
            for w in raw_seg.get("words", []):
                words.append(Word(
                    text=w["word"],
                    start=w["start"] + time_offset,
                    end=w["end"] + time_offset,
                    confidence=w.get("probability"),
                ))

            segments.append(TranscriptionSegment(
                id=i,
                text=raw_seg["text"].strip(),
                start=raw_seg["start"] + time_offset,
                end=raw_seg["end"] + time_offset,
                words=words,
                # Whisper reports avg_logprob; higher (less negative) = more confident
                confidence=raw_seg.get("avg_logprob"),
            ))

        return segments
