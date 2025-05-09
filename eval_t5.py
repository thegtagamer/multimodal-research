import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Fixed Dataset class
class DisasterTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Map event folders to numeric labels
        event_categories = {event: idx for idx, event in enumerate(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, event))}

        for event, label in event_categories.items():
            text_dir = os.path.join(data_path, event, "texts")
            if os.path.exists(text_dir):
                for text_file in os.listdir(text_dir):
                    if text_file.endswith(".txt"):
                        with open(os.path.join(text_dir, text_file), "r", encoding="utf-8") as file:
                            text_content = file.read().strip()
                            self.texts.append(text_content)
                            self.labels.append(label)

        self.label_map = {v: k for k, v in event_categories.items()}
        print(f"Total files loaded: {len(self.texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ✅ Load tokenizer and dataset
t5_model_name = "google/flan-t5-large"  # or t5-base, t5-large, etc.
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

test_directory = "/scratch/adey6/multimodal-research/split_dataset/test"
test_dataset = DisasterTextDataset(test_directory, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

num_classes = len(set(test_dataset.labels))
label_map = test_dataset.label_map

# ✅ Load trained T5 model
model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)
checkpoint_path = "best_t5_text_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print("✅ Loaded model checkpoint.")

# ✅ Evaluation function
def evaluate_model(model, test_dataloader):
    all_preds = []
    all_labels = []
    pred_texts = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            pred_texts.extend(preds)

    # ✅ Map predictions to numeric labels
    pred_labels = []
    for pred in pred_texts:
        found = False
        for idx, label in label_map.items():
            if label.lower() in pred.lower():
                pred_labels.append(idx)
                found = True
                break
        if not found:
            pred_labels.append(-1)  # Unknown

    # ✅ Filter valid predictions
    valid_preds = [p for p in pred_labels if p != -1]
    valid_labels = [l for p, l in zip(pred_labels, all_labels) if p != -1]

    if not valid_preds:
        print("❗ No valid predictions detected.")
        return

    # ✅ Compute metrics
    accuracy = accuracy_score(valid_labels, valid_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(valid_labels, valid_preds, average="weighted", zero_division=0)

    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    print("\n🎯 Evaluation Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # ✅ Save results
    with open("t5_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Results saved to t5_evaluation_results.json")

# ✅ Run evaluation
evaluate_model(model, test_dataloader)
