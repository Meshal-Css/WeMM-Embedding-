"""
Wrapper لتوليد embeddings عبر WeMM-Embedding (Tencent) على sentence-transformers.

موديل multimodal مبني على Qwen3.5 — يقبل نص وصور وفيديو، ويرجّع متجه
مطبّع (L2) بطول 4096، مع دعم Matryoshka للتقليص إلى أبعاد أصغر.

ملاحظات:
- يحتاج trust_remote_code لأن كلاس الموديل يجي من ملفات المستودع نفسه.
- mode يمرّر لـ encode_query / encode_document. تنبيه: هذي النسخة من الموديل
  تسجّل prompts فاضية في config_sentence_transformers.json، والقوالب تعرض
  الاستعلام والمستند بنفس الشكل — فالوضعان ينتجان نفس المتجه حاليًا.
  أبقيناه لأنه واجهة sentence-transformers الرسمية، ويصير فعّالًا لو سجّل
  المزوّد قوالب في إصدار لاحق. تحقق بنفسك عبر: load_model().prompts
"""
import os

import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "tencent/WeMM-Embedding-9B"

_model_cache: dict[tuple[str, str], SentenceTransformer] = {}


def _pick_device() -> str:
    """يختار أفضل جهاز متاح (Apple Silicon أولًا، ثم CUDA، وإلا CPU)."""
    override = os.getenv("WEMM_DEVICE")
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model: str = DEFAULT_MODEL, device: str | None = None) -> SentenceTransformer:
    """يحمّل الموديل مرة وحدة ويحتفظ فيه بالكاش (التحميل ثقيل)."""
    device = device or _pick_device()
    key = (model, device)
    if key not in _model_cache:
        _model_cache[key] = SentenceTransformer(
            model,
            trust_remote_code=True,
            device=device,
            model_kwargs={"dtype": torch.bfloat16},
        )
    return _model_cache[key]


def embed(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    mode: str = "document",
    truncate_dim: int | None = None,
    device: str | None = None,
) -> list[list[float]]:
    """
    يرجع قائمة embeddings لكل نص في texts.

    mode: "document" أو "query". بلا أثر على هذي النسخة (prompts فاضية) —
          راجع ملاحظة الوحدة في أعلى الملف.
    truncate_dim: بُعد من matryoshka_dimensions للتقليص (64/128/256/512/1024/2048/4096).
    """
    if mode not in ("document", "query"):
        raise ValueError(f"mode لازم يكون 'document' أو 'query'، وصلني: '{mode}'")

    # التحقق من المدخلات يجي قبل التحميل — تحميل الموديل ثقيل ومكلف.
    st_model = load_model(model, device)

    encode = st_model.encode_query if mode == "query" else st_model.encode_document
    kwargs = {"show_progress_bar": False}
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
        kwargs["normalize_embeddings"] = True

    vectors = encode(texts, **kwargs)
    return vectors.tolist()
