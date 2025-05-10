import os
import random
import base64
import json
import requests
import asyncio
import aiohttp
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ CONFIGURATION
OPENAI_API_KEY = "OPENAI_KEY_HERE"  # replace with your key
OPENAI_MODEL = "gpt-4o"
MAX_RETRIES = 5
DISASTER_CATEGORIES = ["flood", "non_damage", "damaged_infrastructure", "human_damage", "fires", "damaged_nature"]

# ✅ Encode image as Base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# ✅ Load text from file
def load_text(text_path):
    with open(text_path, "r", encoding="utf-8") as file:
        return file.read().strip()

# ✅ Select category-aware few-shot examples
def get_few_shot_examples(data_path, num_examples=1):
    examples = []
    for category in DISASTER_CATEGORIES:
        text_dir = os.path.join(data_path, category, "texts")
        image_dir = os.path.join(data_path, category, "images")
        text_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(text_dir) if f.endswith(".txt")}
        image_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))}
        common_files = list(set(text_files.keys()) & set(image_files.keys()))
        sampled_files = random.sample(common_files, min(num_examples, len(common_files)))
        for file_name in sampled_files:
            txt_path = os.path.join(text_dir, text_files[file_name])
            img_path = os.path.join(image_dir, image_files[file_name])
            examples.append((txt_path, img_path, category))
    return examples

# ✅ Create prompt
def create_few_shot_prompt(few_shot_examples):
    prompt = (
        "You are a disaster classification assistant. Classify the following multimodal (text + image) inputs "
        "into one of: [flood, non_damage, damaged_infrastructure, human_damage, fires, damaged_nature].\n\n"
        "Respond **only** with a JSON object like:\n"
        "{\n"
        '  "classification": "category_name"\n'
        "}\n\nExamples:\n"
    )
    for txt_path, img_path, label in few_shot_examples:
        text_content = load_text(txt_path)[:200].replace("\n", " ").strip()
        prompt += f"- Text: {text_content}\n  Image: {img_path}\n  Classification: {label}\n"
    return prompt

# ✅ Async API request
async def async_openai_request(session, api_key, model, messages):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 1000
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError:
            delay = min(2 ** attempt, 10)
            jitter = random.uniform(0, delay * 0.1)
            await asyncio.sleep(delay + jitter)
    return None

# ✅ Batch requests
async def batch_process_gpt4o_requests(test_data, few_shot_examples):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        prompt = create_few_shot_prompt(few_shot_examples)
        for txt_path, img_path, _ in test_data:
            text_data = load_text(txt_path)
            image_data = encode_image(img_path)
            messages = [
                {"role": "system", "content": "You are a disaster classification assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": text_data},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]}
            ]
            tasks.append(async_openai_request(session, OPENAI_API_KEY, OPENAI_MODEL, messages))

        responses = await asyncio.gather(*tasks)
        for response in responses:
            if response and "choices" in response:
                try:
                    content = response["choices"][0]["message"]["content"].strip()
                    match = re.search(r'\{.*?\}', content, re.DOTALL)
                    if match:
                        prediction = json.loads(match.group())
                        predicted_label = prediction.get("classification", "non_damage")
                    else:
                        predicted_label = "non_damage"
                except json.JSONDecodeError:
                    predicted_label = "non_damage"
                results.append(predicted_label)
            else:
                results.append("non_damage")
    return results

# ✅ Evaluate
def evaluate_model(test_data, few_shot_examples):
    true_labels = [label for _, _, label in test_data]
    predicted_labels = asyncio.run(batch_process_gpt4o_requests(test_data, few_shot_examples))

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="macro", zero_division=0)

    results_text = (
        "\n🎯 **Evaluation Metrics:**\n"
        f"✅ Accuracy: {accuracy:.4f}\n"
        f"✅ Macro Precision: {precision:.4f}\n"
        f"✅ Macro Recall: {recall:.4f}\n"
        f"✅ Macro F1-score: {f1:.4f}\n\n"
        "**Predictions:**\n"
    )

    for i, (txt_path, img_path, true_label) in enumerate(test_data):
        predicted_label = predicted_labels[i]
        text_snippet = load_text(txt_path)[:200].replace("\n", " ")
        results_text += f"📄 Text: {text_snippet}... | 🖼 Image: {img_path} | ✅ True: {true_label} | 🔍 Predicted: {predicted_label}\n"

    print(results_text)
    with open("./results/gpt_multimodal_evaluation_results.txt", "w") as file:
        file.write(results_text)
    print("📁 Results saved to `gpt_multimodal_evaluation_results.txt`")

# ✅ Load data
train_directory = "./data/train"
test_directory = "./data/test"

# ✅ Few-shot examples (1 per class)
few_shot_examples = get_few_shot_examples(train_directory, num_examples=1)

# ✅ Randomly pick 5 test samples per class
test_samples = []
for category in DISASTER_CATEGORIES:
    text_dir = os.path.join(test_directory, category, "texts")
    image_dir = os.path.join(test_directory, category, "images")
    text_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(text_dir) if f.endswith(".txt")}
    image_files = {f.rsplit('.', 1)[0]: f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))}
    common_files = list(set(text_files.keys()) & set(image_files.keys()))
    sampled_files = random.sample(common_files, min(5, len(common_files)))
    for file_name in sampled_files:
        txt_path = os.path.join(text_dir, text_files[file_name])
        img_path = os.path.join(image_dir, image_files[file_name])
        test_samples.append((txt_path, img_path, category))

# ✅ Run evaluation
evaluate_model(test_samples, few_shot_examples)
