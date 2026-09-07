from pathlib import Path


def load_samples(folder: str = "data/samples") -> list[str]:
    """يحمّل كل ملفات .txt داخل مجلد النصوص التجريبية كقائمة نصوص."""
    paths = sorted(Path(folder).glob("*.txt"))
    return [p.read_text(encoding="utf-8").strip() for p in paths]
