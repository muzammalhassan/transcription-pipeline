"""
MockTranscriptionEngine — a drop-in replacement for TranscriptionEngine
that generates realistic-looking output without loading any model.

Use cases
─────────
• Unit tests that should not require GPU / large downloads.
• CI/CD pipelines where speed matters more than accuracy.
• Demos and documentation.

The mock generates segments at a configurable words-per-second rate,
with word-level timestamps, mimicking Whisper's output schema exactly.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

from pipeline.models import TranscriptionSegment, Word

# Realistic-ish sentences that look like real transcription output
_LOREM_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "In a speech recognition pipeline, latency and accuracy are often in tension.",
    "We handle long audio files by splitting them into overlapping thirty-second windows.",
    "Each chunk is transcribed independently and the results are stitched together.",
    "Word-level timestamps allow downstream systems to highlight text in sync with audio.",
    "The normalisation step converts any input format to sixteen kilohertz mono PCM.",
    "Confidence scores help downstream consumers decide whether to flag uncertain segments.",
    "Automatic language detection works well for monolingual recordings.",
    "For multilingual content, forcing the language via the API improves accuracy.",
    "The pipeline can be extended with speaker diarisation as a post-processing step.",
    "Retrying with a larger model is a simple fallback for low-confidence segments.",
    "Batch processing lets us parallelise chunk transcription across CPU cores.",
    "The SRT export makes it trivial to add captions to a video file.",
    "Edge cases like background noise and music can degrade model performance significantly.",
    "A voice activity detector can be used to skip silent regions before transcription.",
]

_WORDS_PER_SECOND = 2.5      # approximate human speech rate
_SEGMENT_GAP = 0.15          # silence between segments (seconds)
_MIN_WORDS_PER_SEG = 4
_MAX_WORDS_PER_SEG = 18


class MockTranscriptionEngine:
    """
    Generates plausible-looking transcription output for a given audio duration.

    Parameters
    ----------
    seed
        Random seed for reproducibility.  ``None`` means non-deterministic.
    words_per_second
        Controls how fast the "speaker" talks.
    language
        Language code to report in results (default ``"en"``).
    """

    model_size = "mock"

    def __init__(
        self,
        seed: Optional[int] = 42,
        words_per_second: float = _WORDS_PER_SECOND,
        language: str = "en",
    ) -> None:
        self._rng = random.Random(seed)
        self.words_per_second = words_per_second
        self.language = language

    # ------------------------------------------------------------------ #
    # Matches TranscriptionEngine's public interface exactly              #
    # ------------------------------------------------------------------ #

    def transcribe_file(
        self,
        audio_path: str,
        time_offset: float = 0.0,
    ) -> Tuple[List[TranscriptionSegment], str]:
        """
        Generate mock segments that fill the given audio file's duration.

        We read the file to get its true duration so timestamps are realistic,
        then synthesise text + word timestamps to cover that span.
        """
        duration = self._get_duration(audio_path)
        segments = self._generate_segments(duration, time_offset)
        return segments, self.language

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _get_duration(self, audio_path: str) -> float:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0
        except Exception:
            return 10.0  # safe fallback when pydub isn't available

    def _generate_segments(
        self,
        duration: float,
        time_offset: float,
    ) -> List[TranscriptionSegment]:
        segments: List[TranscriptionSegment] = []
        cursor = time_offset
        end_time = time_offset + duration
        seg_id = 0

        while cursor < end_time - 0.5:
            # Pick a sentence and slice it to a random word count
            sentence = self._rng.choice(_LOREM_SENTENCES)
            all_words = sentence.split()
            n_words = self._rng.randint(
                min(_MIN_WORDS_PER_SEG, len(all_words)),
                min(_MAX_WORDS_PER_SEG, len(all_words)),
            )
            chosen_words = all_words[:n_words]
            text = " ".join(chosen_words)

            seg_duration = n_words / self.words_per_second
            seg_end = min(cursor + seg_duration, end_time)

            # Generate per-word timestamps
            word_objs: List[Word] = []
            word_cursor = cursor
            for w in chosen_words:
                w_duration = len(w) / (self.words_per_second * 5)  # ~5 chars/word
                w_end = min(word_cursor + w_duration, seg_end)
                word_objs.append(Word(
                    text=w,
                    start=round(word_cursor, 3),
                    end=round(w_end, 3),
                    confidence=round(self._rng.uniform(0.85, 0.99), 3),
                ))
                word_cursor = w_end

            segments.append(TranscriptionSegment(
                id=seg_id,
                text=text,
                start=round(cursor, 3),
                end=round(seg_end, 3),
                words=word_objs,
                confidence=round(self._rng.uniform(-0.3, -0.05), 4),  # Whisper-style log-prob
            ))

            cursor = seg_end + _SEGMENT_GAP
            seg_id += 1

        return segments
