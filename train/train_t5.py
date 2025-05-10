import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_scheduler
from tqdm import tqdm

# Device selection
def select_device():
    print("Select device for training:")
    print("1. CPU")
    print("2. CUDA (GPU, if available)")
    print("3. MPS (Apple Silicon, if available)")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "2" and torch.cuda.is_available():
        return torch.device("cuda")
    elif choice == "3" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Dataset class for T5
class T5TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, label_map, max_length=512):
        self.inputs = []
        self.targets = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for label_name, label_idx in label_map.items():
            text_dir = os.path.join(data_path, label_name, "texts")
            if os.path.exists(text_dir):
                for fname in os.listdir(text_dir):
                    if fname.endswith(".txt"):
                        with open(os.path.join(text_dir, fname), "r", encoding="utf-8") as f:
                            text = f.read().strip()
                            self.inputs.append(f"classify: {text}")
                            self.targets.append(label_name)

        print(f"Total examples loaded: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_encoding = self.tokenizer(
            self.inputs[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            self.targets[idx],
            truncation=True,
            padding="max_length",
            max_length=10,  # Labels are short
            return_tensors="pt"
        )
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
        }

# Load tokenizer and dataset
device = select_device()
print(f"Using device: {device}")

t5_model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

train_directory = "./data/train"
test_directory = "./data/test"

label_names = sorted(os.listdir(train_directory))
label_map = {label: idx for idx, label in enumerate(label_names)}

train_dataset = T5TextDataset(train_directory, tokenizer, label_map)
test_dataset = T5TextDataset(test_directory, tokenizer, label_map)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load model
model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

# Optimizer, scheduler
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=1000)

# Early stopping setup
best_val_loss = float("inf")
patience = 3
counter = 0

# Training function
def train_model(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs=10):
    global best_val_loss, counter

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(test_dataloader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpointing and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "./models/best_t5_text_model.pth")
            print("✅ Model saved (best so far).")
        else:
            counter += 1
            print(f"⏳ Early stopping patience count: {counter}/{patience}")
            if counter >= patience:
                print("🚨 Early stopping triggered. Training stopped.")
                break

# Run training
train_model(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs=10)

