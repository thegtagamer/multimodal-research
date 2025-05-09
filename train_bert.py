import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Ask user for device preference
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

# Define Dataset Class
class DisasterTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, sample_n=4):
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
        print("Sample texts:")
        for i in range(min(sample_n, len(self.texts))):
            print(f"Sample {i+1}: {self.texts[i][:200]}...")  # Print first 200 chars of each sample

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

# Define Dataset Path
train_directory = "split_dataset/train"  # Path to train data
test_directory = "split_dataset/test"  # Path to test data

# Create Dataset and DataLoader
train_dataset = DisasterTextDataset(train_directory, tokenizer, sample_n=4)
test_dataset = DisasterTextDataset(test_directory, tokenizer, sample_n=4)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define Custom BERT Model with a 1024-Neuron Linear Layer
class CustomBERTLargeClassifier(nn.Module):
    def __init__(self, num_labels):
        super(CustomBERTLargeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-large-uncased")  # Load BERT-Large
        self.dropout = nn.Dropout(0.3)  # Higher dropout for regularization
        self.hidden_layer = nn.Linear(1024, 1024)  # Fully connected 1024-neuron layer
        self.relu = nn.ReLU()  # Activation function
        self.classifier = nn.Linear(1024, num_labels)  # Final classification layer

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.pooler_output)  # Apply dropout
        x = self.hidden_layer(x)  # Pass through 1024-neuron linear layer
        x = self.relu(x)  # Apply activation
        x = self.classifier(x)  # Final classification layer
        return x

# Initialize Model
num_classes = len(set(train_dataset.labels))
model = CustomBERTLargeClassifier(num_labels=num_classes)

# Select Device
device = select_device()
print(f"Using device: {device}")
model.to(device)

# Define Training Parameters
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Training Function
def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=3):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", leave=True):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Evaluate on validation (test) data
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

# Train Model
train_model(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=3)

# Save Model
torch.save(model.state_dict(), "bert_text_classifier.pth")
