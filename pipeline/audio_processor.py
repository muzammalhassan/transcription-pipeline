"""
AudioProcessor — format conversion and normalisation.

Engineering decisions
─────────────────────
1.  **pydub + ffmpeg** for decoding:
      pydub is a thin Python wrapper; ffmpeg does the actual decoding.
      This gives us support for ~100 codecs (MP3, AAC, OGG, FLAC, OPUS …)
      without writing any C extension code.

2.  **Target: 16 kHz mono PCM WAV**
      All major speech models (Whisper, wav2vec, DeepSpeech) were trained on
      16 kHz mono audio. Resampling *before* passing to the model is far
      cheaper than letting the model handle it ad-hoc.

3.  **Fail fast on unknown formats**
      Better to raise an explicit error than silently produce garbage.

4.  **Lazy normalisation**
      If the input is already 16 kHz mono WAV we skip the re-export to avoid
      an unnecessary encode–decode cycle.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Every codec ffmpeg can handle — we whitelist common ones to surface
# user mistakes early rather than letting ffmpeg fail cryptically.
SUPPORTED_EXTENSIONS: set[str] = {
    ".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac",
    ".aac", ".wma", ".opus", ".webm", ".aiff", ".au",
}

# Whisper / wav2vec2 standard — do not change without re-testing the engine
TARGET_SAMPLE_RATE: int = 16_000
TARGET_CHANNELS: int = 1  # mono


class AudioProcessingError(Exception):
    """Raised when audio cannot be loaded or normalised."""


class AudioProcessor:
    def __init__(
        self,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        target_channels: int = TARGET_CHANNELS,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self._check_deps()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def probe(self, file_path: str) -> Dict:
        """Return metadata without modifying the file."""
        audio = self._load(file_path)
        return {
            "path": file_path,
            "format": Path(file_path).suffix.lower(),
            "duration_seconds": round(len(audio) / 1000.0, 3),
            "sample_rate_hz": audio.frame_rate,
            "channels": audio.channels,
            "bit_depth": audio.sample_width * 8,
            "file_size_bytes": os.path.getsize(file_path),
        }

    def normalize(
        self,
        file_path: str,
        output_path: str | None = None,
    ) -> Tuple[str, Dict]:
        """
        Convert *any* supported audio file → 16 kHz mono WAV.

        Returns
        -------
        (output_path, probe_info)
            output_path  — path to the normalised WAV (may equal input if
                           the file was already in the target format).
            probe_info   — metadata dict from :meth:`probe`.
        """
        path = Path(file_path)
        if not path.exists():
            raise AudioProcessingError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise AudioProcessingError(
                f"Unsupported extension '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        info = self.probe(file_path)
        logger.info(
            "Loaded audio | %.1fs | %dHz | %dch | %s",
            info["duration_seconds"],
            info["sample_rate_hz"],
            info["channels"],
            info["format"],
        )

        # Fast-path: already in the correct format
        if (
            ext == ".wav"
            and info["sample_rate_hz"] == self.target_sample_rate
            and info["channels"] == self.target_channels
        ):
            logger.debug("Audio already normalised — skipping re-export")
            return file_path, info

        audio = self._load(file_path)

        if audio.channels != self.target_channels:
            logger.debug("Downmix %dch → mono", audio.channels)
            audio = audio.set_channels(self.target_channels)

        if audio.frame_rate != self.target_sample_rate:
            logger.debug("Resample %dHz → %dHz", audio.frame_rate, self.target_sample_rate)
            audio = audio.set_frame_rate(self.target_sample_rate)

        if output_path is None:
            output_path = str(path.parent / f"{path.stem}_normalised.wav")

        audio.export(output_path, format="wav")
        logger.info("Normalised WAV written → %s", output_path)
        return output_path, info

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load(file_path: str):
        """Load audio via pydub, raising a friendly error on failure."""
        try:
            from pydub import AudioSegment  # local import keeps startup fast
            return AudioSegment.from_file(file_path)
        except Exception as exc:
            raise AudioProcessingError(f"Could not decode '{file_path}': {exc}") from exc

    @staticmethod
    def _check_deps() -> None:
        try:
            import pydub  # noqa: F401
        except ImportError as exc:
            raise AudioProcessingError(
                "pydub is required: pip install pydub"
            ) from exc

        from pydub.utils import which
        if not which("ffmpeg"):
            logger.warning(
                "ffmpeg not found on PATH. "
                "Only native WAV files may work; install ffmpeg for full format support."
            )
