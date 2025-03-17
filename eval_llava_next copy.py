import os
import argparse
import json
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
import copy

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# We won't use conv_templates.get_prompt() since it triggers the Llama 3 tokenizer error.

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===== Evaluation Dataset: expects each category folder contains an "images" folder =====
class DisasterImageDatasetEval:
    def __init__(self, data_path):
        self.data = []  # list of dicts: {"image_path": ..., "label": ...}
        self.event_categories = {}
        for idx, event in enumerate(sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])):
            self.event_categories[event] = idx
        for event in self.event_categories.keys():
            image_dir = os.path.join(data_path, event, "images")
            if os.path.exists(image_dir):
                for f in os.listdir(image_dir):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.data.append({"image_path": os.path.join(image_dir, f), "label": event})
        print(f"Loaded {len(self.data)} images from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        return image, sample["label"]

# ===== Build a Manual Prompt for Inference =====
def build_classification_prompt():
    # Instead of using conv_templates.get_prompt(), we create a prompt manually.
    # This format is acceptable for LLaVA-NeXT models (e.g., for the llama3 variant):
    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-NeXT Zero-shot Classification")
    parser.add_argument("--test_dir", type=str, default="split_dataset/test", help="Path to test data")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load evaluation dataset
    dataset = DisasterImageDatasetEval(args.test_dir)
    
    # Load official LLaVA-NeXT model and processor (8B variant)
    pretrained = "lmms-lab/llama3-llava-next-8b"
    model_name = "llava_llama3"
    device_map = "auto"
    tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
    model.to(device)
    model.eval()
    model.tie_weights()
    
    prompt_template = build_classification_prompt()
    
    all_true = []
    all_pred = []
    
    for idx in tqdm(range(len(dataset)), desc="Evaluating LLaVA-NeXT"):
        image, true_label = dataset[idx]
        # Process the image using the official image processor
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [img.to(dtype=torch.float16, device=device) for img in image_tensor]
        image_sizes = [image.size]
        
        # Tokenize prompt: the prompt already contains the <image> placeholder.
        input_ids = tokenizer_image_token(prompt_template, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.7,
                max_new_tokens=64,
            )
        generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip().lower()
        
        # Basic matching: if a known label appears in the response, choose it; otherwise, default to "non_damage"
        pred_label = "non_damage"
        for label in sorted(dataset.event_categories.keys()):
            if label.lower() in generated_text:
                pred_label = label
                break
        
        all_true.append(true_label)
        all_pred.append(pred_label)
    
    # Compute evaluation metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    accuracy = accuracy_score(all_true, all_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, labels=list(dataset.event_categories.keys()), average='macro')
    cm = confusion_matrix(all_true, all_pred, labels=list(dataset.event_categories.keys()))
    
    print("\n===== Evaluation Metrics =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    out_data = {"true_labels": all_true, "predictions": all_pred}
    with open("llava_next_eval_predictions.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print("Saved predictions to llava_next_eval_predictions.json")

if __name__ == "__main__":
    main()
