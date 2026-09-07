"""
محرك البحث داخل الفيديو باللغة الطبيعية.

الفكرة: نقطّع الفيديو لإطارات موسومة بزمنها، نرمّز كل إطار كصورة عبر WeMM،
ثم نرمّز سؤال المستخدم كنص. بما إن الصور والنصوص تسكن نفس الفضاء المتجهي،
مقارنة cosine بينها تعطينا الإطار الأقرب — وزمنه هو الجواب.

نستخدم PyAV لفك الترميز لأنه يحمل FFmpeg بداخله، بعكس decord/torchcodec
اللي يحتاجون مكتبات نظام غير متوفرة على macOS بدون brew.
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

# أقصى ضلع للإطار قبل الترميز — يقلل التكلفة بدون أثر يذكر على الدقة.
MAX_SIDE = 448


@dataclass
class Frame:
    """إطار واحد مع زمنه بالثواني."""
    time_s: float
    image: Image.Image


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 300) -> list[Frame]:
    """
    يستخرج إطارًا كل 1/fps ثانية.

    fps=1.0 يعني إطارًا كل ثانية. ارفعها لدقة زمنية أعلى وتكلفة أكبر.
    max_frames سقف أمان عشان فيديو طويل ما يفجّر الذاكرة.
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
    """يرمّز الإطارات كصور. يرجع مصفوفة (عدد الإطارات، 4096)."""
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
    يرجع أفضل top_k لحظات كـ (الزمن، الدرجة، الإطار).

    min_gap_s يمنع رجوع خمس نتائج من نفس اللحظة: الإطارات المتجاورة تتشابه
    بشدة، فبدون هذا الفلتر تطلع النتائج كلها من ثانية واحدة.
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
    """يحوّل الثواني إلى mm:ss."""
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"
