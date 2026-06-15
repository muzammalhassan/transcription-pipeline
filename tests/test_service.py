import os
import sys
import struct
import wave
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.pipeline import TranscriptionPipeline
from mock.mock_engine import MockTranscriptionEngine


def _write_wav(path, duration_s, sample_rate=16000):
    n = int(duration_s * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return path


def test_transcribe():
    with tempfile.TemporaryDirectory() as d:
        audio = _write_wav(os.path.join(d, "test.wav"), 10.0)
        pipeline = TranscriptionPipeline(engine=MockTranscriptionEngine(seed=42))
        result = pipeline.transcribe(audio)
        assert result.full_text
        assert len(result.segments) > 0
        print("PASS: transcription returned segments")


if __name__ == "__main__":
    test_transcribe()
