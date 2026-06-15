"""Tests for pipeline.models"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.models import Word, TranscriptionSegment, TranscriptionResult, _srt_ts, _vtt_ts


def _make_result(n_segments: int = 3) -> TranscriptionResult:
    segs = []
    for i in range(n_segments):
        start = i * 5.0
        end = start + 4.5
        words = [
            Word(text="hello", start=start, end=start + 1.0, confidence=0.99),
            Word(text="world", start=start + 1.2, end=start + 2.0, confidence=0.95),
        ]
        segs.append(TranscriptionSegment(id=i, text="hello world", start=start, end=end, words=words, confidence=-0.1))
    return TranscriptionResult(segments=segs, language="en", duration=n_segments * 5.0, model="mock", audio_file="test.wav")


def test_segment_duration():
    seg = TranscriptionSegment(id=0, text="hi", start=1.0, end=3.5)
    assert seg.duration == 2.5


def test_result_full_text():
    r = _make_result(2)
    assert r.full_text == "hello world hello world"


def test_result_word_count():
    r = _make_result(2)
    assert r.word_count == 4


def test_to_dict_keys():
    r = _make_result()
    d = r.to_dict()
    for key in ("audio_file", "language", "duration_seconds", "model", "full_text", "segments"):
        assert key in d, f"Missing key: {key}"


def test_to_json_valid():
    r = _make_result()
    parsed = json.loads(r.to_json())
    assert parsed["language"] == "en"


def test_to_srt_format():
    r = _make_result(1)
    srt = r.to_srt()
    assert "1\n" in srt
    assert "-->" in srt


def test_to_vtt_format():
    r = _make_result(1)
    vtt = r.to_vtt()
    assert vtt.startswith("WEBVTT")
    assert "-->" in vtt


def test_srt_ts():
    assert _srt_ts(0.0) == "00:00:00,000"
    assert _srt_ts(3661.5) == "01:01:01,500"


def test_vtt_ts():
    assert _vtt_ts(0.0) == "00:00:00.000"


def test_segment_to_dict_has_words():
    seg = TranscriptionSegment(
        id=0, text="hi", start=0.0, end=1.0,
        words=[Word("hi", 0.0, 1.0, confidence=0.9)]
    )
    d = seg.to_dict()
    assert len(d["words"]) == 1
    assert d["words"][0]["text"] == "hi"


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
