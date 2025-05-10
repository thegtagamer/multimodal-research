import os
import random
import json
import re
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ CONFIGURATION
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MAX_SAMPLES_PER_CLASS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISASTER_CATEGORIES = ["flood", "non_damage", "damaged_infrastructure", "human_damage", "fires", "damaged_nature"]

# ✅ Load model and processor
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_ID)

print(f"✅ Using device: {DEVICE}")
print("✅ Model and processor loaded.")

# ✅ Load text from file
def load_text(text_path):
    with open(text_path, "r", encoding="utf-8") as file:
        return file.read().strip()

# ✅ Get random test samples
def get_test_samples(test_directory):
    test_samples = []
    for category in DISASTER_CATEGORIES:
        text_dir = os.path.join(test_directory, category, "texts")
        image_dir = os.path.join(test_directory, category, "images")
        text_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(text_dir) if f.endswith(".txt")}
        image_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))}
        common_files = list(set(text_files.keys()) & set(image_files.keys()))
        sampled_files = random.sample(common_files, min(MAX_SAMPLES_PER_CLASS, len(common_files)))
        for file_name in sampled_files:
            txt_path = os.path.join(text_dir, text_files[file_name])
            img_path = os.path.join(image_dir, image_files[file_name])
            test_samples.append((txt_path, img_path, category.lower()))
    return test_samples

# ✅ Clean LLaVA output
def clean_llava_output(raw_output):
    raw_output = raw_output.lower()
    if "assistant:" in raw_output:
        raw_output = raw_output.split("assistant:")[-1]
    if "er:" in raw_output:
        raw_output = raw_output.split("er:")[-1]
    for prefix in [
        "classify this image into one of these categories:",
        "strictly respond only with the category name given below. Do not come up with anything else other than what is mentioned below",
        "flood, non_damage, damaged_infrastructure, human_damage, fires, damaged_nature.",
    ]:
        raw_output = raw_output.replace(prefix.lower(), "")
        raw_output = raw_output.replace("\\", "")  # �~\~E REMOVE all backslashes
    return raw_output.strip()

# ✅ Run single inference
def run_llava_inference(text_path, image_path, instruction):
    try:
        text_content = load_text(text_path)
        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction + "\n" + text_content},
                    {"type": "image"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(DEVICE, torch.float16)
        output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        decoded_output = processor.decode(output[0][2:], skip_special_tokens=True).strip()
        cleaned_prediction = clean_llava_output(decoded_output)
        return cleaned_prediction
    except Exception as e:
        print(f"❌ Error with {image_path}: {e}")
        return None

# ✅ Evaluate
def evaluate_model(test_samples):
    instruction = (
        "Classify this multimodal input (text + image) into one of these categories: "
        "flood, non_damage, damaged_infrastructure, human_damage, fires, damaged_nature. "
        "Respond only with the category name."
    )
    true_labels, predicted_labels = [], []

    for txt_path, img_path, true_label in tqdm(test_samples, desc="Evaluating"):
        print(f"\n🔍 Processing image: {img_path}")
        prediction = run_llava_inference(txt_path, img_path, instruction)
        if prediction:
            print(f"✅ True: {true_label} | 🔍 Predicted: {prediction}")
            true_labels.append(true_label)
            predicted_labels.append(prediction)
        else:
            print(f"❌ Skipped {img_path}")

    if true_labels and predicted_labels:
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average="macro", zero_division=0
        )

        results_text = (
            "\n🎯 **Evaluation Metrics:**\n"
            f"✅ Accuracy: {accuracy:.4f}\n"
            f"✅ Macro Precision: {precision:.4f}\n"
            f"✅ Macro Recall: {recall:.4f}\n"
            f"✅ Macro F1-score: {f1:.4f}\n\n"
            "**Predictions:**\n"
        )
        for txt_path, img_path, true, pred in zip([x[0] for x in test_samples], [x[1] for x in test_samples], true_labels, predicted_labels):
            text_snippet = load_text(txt_path)[:200].replace("\n", " ")
            results_text += f"📄 Text: {text_snippet}... | 🖼 Image: {img_path} | ✅ True: {true} | 🔍 Predicted: {pred}\n"

        print(results_text)
        with open("./results/llava_multimodal_evaluation_results.txt", "w") as file:
            file.write(results_text)
        print("📁 Results saved to `llava_multimodal_evaluation_results.txt`")
    else:
        print("\n⚠️ No valid predictions collected.")

# ✅ MAIN EXECUTION
test_directory = "./data/test"
test_samples = get_test_samples(test_directory)
evaluate_model(test_samples)

