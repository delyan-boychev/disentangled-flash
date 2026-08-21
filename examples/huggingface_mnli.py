from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from disentangled_flash import optimize_deberta

MODEL = "microsoft/deberta-v2-xlarge-mnli"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, torch_dtype=torch.float16
).cuda().eval()

optimize_deberta(model.deberta, sequence_lengths=[64, 128, 256, 512])

inputs = tokenizer(
    "A dog runs through a field.",
    "An animal is running.",
    padding="max_length", truncation=True, max_length=64, return_tensors="pt",
)
inputs = {key: value.cuda() for key, value in inputs.items()}

with torch.inference_mode():
    logits = model(**inputs).logits
print(logits)
