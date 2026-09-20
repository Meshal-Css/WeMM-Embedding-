# test reranker model
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "BAAI/bge-reranker-v2-m3"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading {MODEL} on {device}...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model = model.to(device).eval()

query = "كيف أخفّض فاتورة الكهرباء في البيت؟"

documents = [
    "يمكن شراء العقار بالتقسيط عبر التمويل العقاري.",
    "يساعد العزل الحراري وضبط المكيف على تقليل استهلاك الكهرباء.",
    "تستخدم نماذج التضمين لقياس التشابه بين النصوص.",
]

pairs = [[query, document] for document in documents]

inputs = tokenizer(
    pairs,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
).to(device)

with torch.inference_mode():
    scores = model(**inputs).logits.flatten().float().cpu().tolist()

ranked = sorted(
    zip(documents, scores, strict=False),
    key=lambda item: item[1],
    reverse=True,
)

print(f"\nالسؤال: {query}\n")
for position, (document, score) in enumerate(ranked, start=1):
    print(f"{position}. [{score:.4f}] {document}")
