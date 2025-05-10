import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from tqdm import tqdm

import torch._inductor
torch._inductor.config.triton.cudagraphs = False  # Disable advanced CUDA graphs
torch._inductor.config.compile_threads = 1  # Reduce memory usage

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import os
os.environ["USE_FLASH_ATTENTION"] = "0"  # ✅ Disable FlashAttention

# **Select Device**
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

# **Dataset Class for ModernBERT**
class DisasterTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length

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

# **Load Tokenizer & Dataset**
device = select_device()
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")  # ✅ Correct ModernBERT model

train_directory = "./data/train"
test_directory = "./data/test"

train_dataset = DisasterTextDataset(train_directory, tokenizer)
test_dataset = DisasterTextDataset(test_directory, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# **Load ModernBERT Model**
num_classes = len(set(train_dataset.labels))
model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-large", num_labels=num_classes).to(device)

# **Define Optimizer, Scheduler & Loss Function**
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=1000)  # Learning rate decay

# **Early Stopping Parameters**
best_val_loss = float("inf")
patience = 2
counter = 0

# **Training Function**
def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, epochs=10):
    global best_val_loss, counter

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # **Validation Step**
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_dataloader)

        print(f"Epoch {epoch+1} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        # **Model Checkpointing**
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "./models/best_modernbert_text_classifier.pth")  # Save best model
            print("✅ Model saved as validation loss improved.")
        else:
            counter += 1
            print(f"⏳ Early stopping patience count: {counter}/{patience}")

        # **Early Stopping**
        if counter >= patience:
            print("🚨 Early stopping triggered. Training stopped.")
            break

# **Train Model**
train_model(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, epochs=10)

