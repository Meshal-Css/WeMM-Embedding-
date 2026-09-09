"""
سجل المزودين المتاحين. أضف أي مزود جديد هنا بعد إنشاء ملفه.
"""

from . import hf_embed, ollama_embed, wemm_embed

PROVIDERS = {
    "ollama": ollama_embed.embed,
    "hf": hf_embed.embed,
    "wemm": wemm_embed.embed,
}


def get_provider(name: str):
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(PROVIDERS)}")
    return PROVIDERS[name]
