"""Tests for pipeline.chunker"""

import os
import sys
import struct
import wave
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.chunker import AudioChunker, AudioChunk


def _write_wav(path: str, duration_s: float, sample_rate: int = 16000) -> str:
    """Write a minimal valid silent WAV file."""
    n_samples = int(duration_s * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return path


def test_short_audio_no_split():
    """Audio shorter than chunk_duration should return a single chunk pointing at the original."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = _write_wav(os.path.join(tmpdir, "short.wav"), duration_s=10.0)
        chunker = AudioChunker(chunk_duration=30.0, overlap=2.0, output_dir=tmpdir)
        chunks = chunker.split(audio)
        assert len(chunks) == 1
        assert chunks[0].file_path == audio  # original path, no copy made


def test_long_audio_splits_into_multiple():
    """A 65 s file with 30 s chunks / 2 s overlap should produce 3 chunks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = _write_wav(os.path.join(tmpdir, "long.wav"), duration_s=65.0)
        chunker = AudioChunker(chunk_duration=30.0, overlap=2.0, output_dir=tmpdir)
        chunks = chunker.split(audio)
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"


def test_chunks_cover_full_duration():
    """The last chunk must reach the end of the audio."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = _write_wav(os.path.join(tmpdir, "med.wav"), duration_s=45.0)
        chunker = AudioChunker(chunk_duration=30.0, overlap=2.0, output_dir=tmpdir)
        chunks = chunker.split(audio)
        assert chunks[-1].end_time == 45.0


def test_chunk_indices_are_sequential():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = _write_wav(os.path.join(tmpdir, "seq.wav"), duration_s=65.0)
        chunker = AudioChunker(chunk_duration=30.0, overlap=2.0, output_dir=tmpdir)
        chunks = chunker.split(audio)
        for i, c in enumerate(chunks):
            assert c.index == i


def test_chunk_files_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = _write_wav(os.path.join(tmpdir, "exist.wav"), duration_s=65.0)
        chunker = AudioChunker(chunk_duration=30.0, overlap=2.0, output_dir=tmpdir)
        chunks = chunker.split(audio)
        for chunk in chunks:
            assert os.path.exists(chunk.file_path), f"Missing: {chunk.file_path}"


def test_needs_chunking_true():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = _write_wav(os.path.join(tmpdir, "big.wav"), duration_s=60.0)
        chunker = AudioChunker(chunk_duration=30.0, overlap=2.0, output_dir=tmpdir)
        assert chunker.needs_chunking(audio) is True


def test_needs_chunking_false():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = _write_wav(os.path.join(tmpdir, "small.wav"), duration_s=10.0)
        chunker = AudioChunker(chunk_duration=30.0, overlap=2.0, output_dir=tmpdir)
        assert chunker.needs_chunking(audio) is False


def test_chunk_duration_property():
    c = AudioChunk(index=0, start_time=0.0, end_time=30.0, file_path="/fake.wav")
    assert c.duration == 30.0


def test_invalid_chunk_duration_raises():
    try:
        AudioChunker(chunk_duration=31.0)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_overlap_must_be_smaller_than_chunk():
    try:
        AudioChunker(chunk_duration=30.0, overlap=30.0)
        assert False, "Should have raised"
    except ValueError:
        pass


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
