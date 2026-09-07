"""
Natural-language search inside a video.

Run it: project/run.sh
"""
from __future__ import annotations

import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import sys
from pathlib import Path

import gradio as gr

# نضيف مجلد المشروع لمسار الاستيراد عشان يشتغل من أي مجلد تشغّله منه.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_search import extract_frames, encode_frames, search, format_timestamp  # noqa: E402

CSS = """
/* يحاذي حسب لغة ما يكتبه المستخدم — الاستعلام يقبل العربية والإنجليزية. */
.auto-dir textarea, .auto-dir input { unicode-bidi: plaintext; }
footer { display: none !important; }
"""


def index_video(video_path, fps, progress=gr.Progress()):
    """يستخرج الإطارات ويرمّزها. يرجع الحالة + رسالة."""
    if not video_path:
        return None, None, "⚠️ Upload a video first."

    progress(0, desc="Extracting frames…")
    frames = extract_frames(video_path, fps=fps)
    if not frames:
        return None, None, "⚠️ Could not read any frames from this file."

    progress(0, desc=f"Encoding {len(frames)} frames with WeMM…")
    vectors = encode_frames(frames, progress=lambda p: progress(p, desc="Encoding frames…"))

    duration = frames[-1].time_s
    return frames, vectors, (
        f"✅ Ready — {len(frames)} frames covering {format_timestamp(duration)} "
        f"(one frame every {1/fps:.1f}s). Describe your scene below."
    )


def run_search(query, frames, vectors, top_k):
    """يبحث عن اللحظات المطابقة ويرجع معرضًا + جدولًا."""
    if frames is None or vectors is None:
        return [], "⚠️ Index the video first."
    if not query or not query.strip():
        return [], "⚠️ Describe what you are looking for."

    results = search(query.strip(), vectors, frames, top_k=int(top_k))
    gallery = [(image, f"{format_timestamp(t)}  •  {score:.3f}") for t, score, image in results]

    table = "| # | Time | Second | Match |\n|---|---|---|---|\n"
    for i, (t, score, _) in enumerate(results, 1):
        table += f"| {i} | **{format_timestamp(t)}** | {t:.1f}s | {score:.4f} |\n"

    best_t, best_score, _ = results[0]
    table += f"\n**Best match: second {best_t:.0f}** (`{format_timestamp(best_t)}`)"
    if best_score < 0.30:
        table += "\n\n⚠️ Low match score — this scene may not appear in the video."
    return gallery, table


with gr.Blocks(title="Video Scene Search", css=CSS, theme=gr.themes.Soft()) as demo:
    frames_state = gr.State()
    vectors_state = gr.State()

    gr.Markdown(
        "# 🎬 Search Inside a Video\n"
        "Upload a clip, index it, then describe the scene you are looking for — "
        "such as *“a white car driving down the road”* — and get the second it appears.\n"
        "\n"
        "*Queries work in any language, including Arabic.*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Clip", height=280)
            fps = gr.Slider(0.25, 4, value=1, step=0.25, label="Frames per second",
                            info="Higher = finer timing, slower indexing")
            index_btn = gr.Button("① Index clip", variant="primary")
            status = gr.Markdown("")

        with gr.Column(scale=1):
            query = gr.Textbox(label="② Describe the scene", elem_classes="auto-dir", lines=2,
                               placeholder="a white car driving down the road")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Number of results")
            search_btn = gr.Button("🔍 Search", variant="primary")
            table = gr.Markdown("")

    gallery = gr.Gallery(label="Matching moments", columns=5, height=250, object_fit="contain")

    index_btn.click(index_video, [video, fps], [frames_state, vectors_state, status])
    search_btn.click(run_search, [query, frames_state, vectors_state, top_k], [gallery, table])
    query.submit(run_search, [query, frames_state, vectors_state, top_k], [gallery, table])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
