import os
import random
import time
import json
import requests
import re
import csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ CONFIGURATION
OPENAI_API_KEY = "OPENAI_KEY_HERE"  # replace with your key
OPENAI_MODEL = "gpt-4o"
MAX_RETRIES = 5
DISASTER_CATEGORIES = ["flood", "non_damage", "damaged_infrastructure", "human_damage", "fires", "damaged_nature"]
DEBUG = True  # set to False to reduce logs
SLEEP_TIME = 5.0  # seconds between requests

def load_text(text_path):
    with open(text_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def get_few_shot_examples(data_path, num_examples=1):
    examples = []
    for category in DISASTER_CATEGORIES:
        text_dir = os.path.join(data_path, category, "texts")
        text_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]
        sampled_texts = random.sample(text_files, min(num_examples, len(text_files)))
        for txt in sampled_texts:
            txt_path = os.path.join(text_dir, txt)
            examples.append((txt_path, category))
    return examples

def create_few_shot_prompt(few_shot_examples):
    prompt = (
        "You are a disaster classification assistant. Classify the following text into one of:\n"
        "[flood, non_damage, damaged_infrastructure, human_damage, fires, damaged_nature].\n\n"
        "Respond **only** with a JSON object:\n"
        "{\n"
        '  "classification": "category_name"\n'
        "}\n\n"
        "Examples:\n"
    )
    for txt_path, label in few_shot_examples:
        text_content = load_text(txt_path)[:200].replace("\n", " ").strip()
        prompt += f'Text: {text_content}\n{{"classification": "{label}"}}\n\n'
    prompt += "Now classify this text:\n"
    return prompt

def batch_process_gpt4o_requests(test_data, few_shot_examples):
    predicted_labels = []
    prompt_base = create_few_shot_prompt(few_shot_examples)

    for txt_path, _ in test_data:
        text_data = load_text(txt_path)
        full_prompt = prompt_base + text_data
        messages = [
            {"role": "system", "content": "You are a disaster classification assistant."},
            {"role": "user", "content": full_prompt}
        ]

        if DEBUG:
            print("\n==========================")
            print(f"📝 Input Text Path: {txt_path}")
            print(f"📝 Input Text: {text_data[:200]}...")
            print(f"📜 Final Prompt Sent to GPT:\n{full_prompt}")

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": OPENAI_MODEL, "messages": messages, "temperature": 0.7, "top_p": 0.8, "max_tokens": 1000}
                )
                if DEBUG:
                    print(f"🔗 API Status: {response.status_code}")
                response.raise_for_status()
                result = response.json()
                break
            except requests.exceptions.RequestException:
                delay = min(2 ** attempt, 10)
                jitter = random.uniform(0, delay * 0.1)
                sleep_time = delay + jitter
                if DEBUG:
                    print(f"⚠️ Sync retrying in {sleep_time:.2f}s... (Attempt {attempt +1})")
                time.sleep(sleep_time)
        else:
            print("🚨 Sync maximum retries reached, skipping.")
            predicted_labels.append("unknown")
            continue

        raw_content = result["choices"][0]["message"]["content"].strip()
        if DEBUG:
            print(f"💬 Raw GPT Response:\n{raw_content}")

        match = re.search(r'\{.*?\}', raw_content, re.DOTALL)
        if match:
            json_text = match.group()
            if DEBUG:
                print(f"✅ Extracted JSON:\n{json_text}")
            try:
                prediction = json.loads(json_text)
                predicted_label = prediction.get("classification", "unknown")
            except json.JSONDecodeError:
                if DEBUG:
                    print("❌ JSON decoding failed.")
                predicted_label = "unknown"
        else:
            if DEBUG:
                print("❌ No JSON block found in response.")
            predicted_label = "unknown"

        if DEBUG:
            print(f"🔍 Final Predicted Label: {predicted_label}")
        predicted_labels.append(predicted_label)
        time.sleep(SLEEP_TIME)  # Slow down requests

    return predicted_labels

def evaluate_model(test_data, few_shot_examples):
    true_labels = [label for _, label in test_data]
    predicted_labels = batch_process_gpt4o_requests(test_data, few_shot_examples)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="macro", zero_division=0)

    results_text = (
        "\n🎯 **Evaluation Metrics:**\n"
        f"✅ Accuracy: {accuracy:.4f}\n"
        f"✅ Macro Precision: {precision:.4f}\n"
        f"✅ Macro Recall: {recall:.4f}\n"
        f"✅ Macro F1-score: {f1:.4f}\n\n"
        "**Test Sample Predictions:**\n"
    )

    csv_rows = [["Text Path", "True Label", "Predicted Label", "Snippet"]]

    for i, (txt_path, true_label) in enumerate(test_data):
        predicted_label = predicted_labels[i]
        text_snippet = load_text(txt_path)[:200]
        results_text += f"📄 Text: {text_snippet}... | ✅ True: {true_label} | 🔍 Predicted: {predicted_label}\n"
        csv_rows.append([txt_path, true_label, predicted_label, text_snippet])

    print(results_text)

    with open("./results/gpt_text_evaluation_results.txt", "w") as file:
        file.write(results_text)
    print("📁 Results saved to `gpt_text_evaluation_results.txt`")

    with open("./results/gpt_text_evaluation_results.csv", "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    print("📁 CSV saved to `gpt_text_evaluation_results.csv`")

# ✅ Load Training and Testing Data
train_directory = "./data/train"
test_directory = "./data/test"

# ✅ Select Few-Shot Examples (1 per class)
few_shot_examples = get_few_shot_examples(train_directory, num_examples=1)

# ✅ Randomly pick 5 test samples per class
test_samples = []
for category in DISASTER_CATEGORIES:
    text_dir = os.path.join(test_directory, category, "texts")
    text_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]
    sampled_files = random.sample(text_files, min(5, len(text_files)))
    for txt in sampled_files:
        txt_path = os.path.join(text_dir, txt)
        test_samples.append((txt_path, category))

# ✅ Run Evaluation
evaluate_model(test_samples, few_shot_examples)
