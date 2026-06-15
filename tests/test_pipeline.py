"""Integration tests for the full pipeline (mock engine — no Whisper needed)."""

import os
import struct
import sys
import tempfile
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.pipeline import TranscriptionPipeline
from pipeline.models import TranscriptionResult
from mock.mock_engine import MockTranscriptionEngine


def _write_wav(path: str, duration_s: float, sample_rate: int = 16000) -> str:
    n = int(duration_s * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return path


def _make_pipeline(chunk_duration: float = 30.0) -> TranscriptionPipeline:
    return TranscriptionPipeline(
        chunk_duration=chunk_duration,
        chunk_overlap=2.0,
        engine=MockTranscriptionEngine(seed=0),
    )


# ── Basic smoke tests ────────────────────────────────────────────────────────

def test_short_file_returns_result():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "s.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        assert isinstance(result, TranscriptionResult)


def test_result_has_segments():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "s.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        assert len(result.segments) > 0


def test_result_full_text_non_empty():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "s.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        assert len(result.full_text) > 0


def test_audio_file_name_preserved():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "interview.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        assert result.audio_file == "interview.wav"


def test_model_label():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "s.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        assert result.model == "mock"


# ── Timestamp correctness ────────────────────────────────────────────────────

def test_segments_are_time_ordered():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "s.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        starts = [seg.start for seg in result.segments]
        assert starts == sorted(starts), "Segments not in time order"


def test_segment_end_after_start():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "s.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        for seg in result.segments:
            assert seg.end > seg.start, f"Segment {seg.id}: end <= start"


def test_word_timestamps_within_segment():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "s.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        for seg in result.segments:
            for word in seg.words:
                assert word.start >= seg.start - 0.01
                assert word.end <= seg.end + 0.01


# ── Long audio (chunking) ────────────────────────────────────────────────────

def test_long_audio_completes():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "long.wav"), 65.0)
        result = _make_pipeline(chunk_duration=30.0).transcribe(audio)
        assert isinstance(result, TranscriptionResult)


def test_long_audio_has_more_segments():
    with tempfile.TemporaryDirectory() as d:
        short = _write_wav(os.path.join(d, "short.wav"), 10.0)
        long  = _write_wav(os.path.join(d, "long.wav"),  65.0)
        pl = _make_pipeline(chunk_duration=30.0)
        r_short = pl.transcribe(short)
        r_long  = pl.transcribe(long)
        assert len(r_long.segments) > len(r_short.segments)


def test_no_duplicate_segment_ids():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "long.wav"), 65.0)
        result = _make_pipeline(chunk_duration=30.0).transcribe(audio)
        ids = [seg.id for seg in result.segments]
        assert ids == list(range(len(ids))), "Segment IDs are not contiguous"


# ── Serialisation ────────────────────────────────────────────────────────────

def test_to_json_is_valid():
    import json
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "s.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        parsed = json.loads(result.to_json())
        assert "segments" in parsed


def test_to_srt_contains_arrow():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "s.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        assert "-->" in result.to_srt()


def test_to_vtt_starts_with_webvtt():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "s.wav"), 10.0)
        result = _make_pipeline().transcribe(audio)
        assert result.to_vtt().startswith("WEBVTT")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
