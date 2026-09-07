"""
Natural-language search inside a video.

The idea: slice the video into frames tagged with their timestamps, encode each
frame as an image with WeMM, then encode the user's question as text. Because
images and text live in the same vector space, a cosine comparison between them
gives us the closest frame — and its timestamp is the answer.

We decode with PyAV because it bundles FFmpeg, unlike decord and torchcodec,
which need system libraries unavailable on macOS without Homebrew.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.providers.wemm_embed import load_model  # noqa: E402

# Longest frame side before encoding — cuts cost with no meaningful accuracy loss.
MAX_SIDE = 448


@dataclass
class Frame:
    """A single frame together with its timestamp in seconds."""
    time_s: float
    image: Image.Image


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 300) -> list[Frame]:
    """
    Extract one frame every 1/fps seconds.

    fps=1.0 means one frame per second. Raise it for finer timing at a higher cost.
    max_frames is a safety cap so a long video cannot exhaust memory.
    """
    step = 1.0 / fps
    frames: list[Frame] = []

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        next_t = 0.0

        for packet_frame in container.decode(stream):
            if packet_frame.pts is None:
                continue
            t = float(packet_frame.pts * stream.time_base)
            if t + 1e-6 < next_t:
                continue

            img = packet_frame.to_image()
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            frames.append(Frame(time_s=t, image=img))

            next_t = t + step
            if len(frames) >= max_frames:
                break

    return frames


def encode_frames(frames: list[Frame], batch_size: int = 8, progress=None) -> np.ndarray:
    """Encode the frames as images. Returns an array of shape (n_frames, 4096)."""
    model = load_model()
    vectors: list[np.ndarray] = []

    for i in range(0, len(frames), batch_size):
        batch = [{"image": f.image} for f in frames[i : i + batch_size]]
        vectors.append(np.asarray(model.encode_document(batch)))
        if progress is not None:
            progress(min(i + batch_size, len(frames)) / len(frames))

    return np.vstack(vectors)


def search(query: str, frame_vectors: np.ndarray, frames: list[Frame], top_k: int = 5,
           min_gap_s: float = 2.0) -> list[tuple[float, float, Image.Image]]:
    """
    Return the best top_k moments as (timestamp, score, frame).

    min_gap_s stops five results from collapsing onto the same moment: adjacent
    frames are nearly identical, so without this filter every hit comes from one second.
    """
    model = load_model()
    q = np.asarray(model.encode_query([query])[0])
    scores = frame_vectors @ q

    results: list[tuple[float, float, Image.Image]] = []
    for idx in np.argsort(scores)[::-1]:
        t = frames[idx].time_s
        if any(abs(t - chosen_t) < min_gap_s for chosen_t, _, _ in results):
            continue
        results.append((t, float(scores[idx]), frames[idx].image))
        if len(results) == top_k:
            break

    return results


def format_timestamp(seconds: float) -> str:
    """Format seconds as mm:ss."""
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"
