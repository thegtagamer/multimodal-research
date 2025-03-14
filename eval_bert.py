import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def select_device():
    print("Select device for evaluation:")
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

# Define Dataset Class (same as in training script)
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

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Define Paths
test_directory = "/Users/abhi/multi-modal/v2/split_dataset/test"

# Create Test Dataset and DataLoader
test_dataset = DisasterTextDataset(test_directory, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Get number of classes dynamically
num_classes = len(set(test_dataset.labels))

# Define Custom BERT Model (Same as Training Script)
class CustomBERTLargeClassifier(nn.Module):
    def __init__(self, num_labels):
        super(CustomBERTLargeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-large-uncased")
        self.dropout = nn.Dropout(0.3)  # Regularization
        self.hidden_layer = nn.Linear(1024, 1024)  # Fully connected 1024-neuron layer
        self.relu = nn.ReLU()  # Activation
        self.classifier = nn.Linear(1024, num_labels)  # Output layer

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.pooler_output)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

# Evaluate Model
def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)  # Custom Model directly returns logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

#     # Confusion Matrix Plot
#     plt.figure(figsize=(8,6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")
#     plt.title("Confusion Matrix")
#     plt.show()

# # Run Evaluation
if __name__ == "__main__":
    device = select_device()
    print(f"Using device: {device}")

    # Load Custom Model
    model = CustomBERTLargeClassifier(num_labels=num_classes)
    model.load_state_dict(torch.load("bert_text_classifier.pth", map_location="cpu"))  # Ensure CPU loading

    model.to(device)

    # Evaluate Model
    evaluate_model(model, test_dataloader, device)
