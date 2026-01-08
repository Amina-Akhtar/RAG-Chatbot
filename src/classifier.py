# Using locally saved model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
path=r"FineTuning\Llama-3.2-Classifier"
label_map = {0:"benign", 1:"malicious"}
MAX_LENGTH=256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    path,
    num_labels=len(label_map)
).to(device)
tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token = tokenizer.eos_token
model.eval() # set to evaluation mode

def classify_prompt(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True,padding=False,max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, predicted_class].item()
    predicted_label = "Benign" if predicted_class == 0 else "Malicious"
    return {
        "Label":predicted_label,
        "Confidence":confidence
        }
classify_prompt("How can i steal money from someone's bank account without their knowledge?")