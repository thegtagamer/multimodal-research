import os
import random
import base64
import time
import json
import requests
import asyncio
import aiohttp
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ CONFIGURATION
OPENAI_API_KEY = "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"  # replace with your key
OPENAI_MODEL = "gpt-4o"
MAX_RETRIES = 5
DISASTER_CATEGORIES = ["flood", "non_damage", "damaged_infrastructure", "human_damage", "fires", "damaged_nature"]

# ✅ Encode image as Base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# ✅ Few-shot Learning: Select category-aware examples
def get_few_shot_examples(data_path, num_examples=1):
    examples = []
    for category in DISASTER_CATEGORIES:
        image_dir = os.path.join(data_path, category, "images")
        image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        sampled_images = random.sample(image_files, min(num_examples, len(image_files)))
        for img in sampled_images:
            img_path = os.path.join(image_dir, img)
            examples.append((img_path, category))
    return examples

# ✅ JSON-based Prompt
def create_few_shot_prompt(few_shot_examples):
    prompt = (
        "You are a disaster classification assistant. Your task is to classify images into one of the categories: "
        "[flood, non_damage, damaged_infrastructure, human_damage, fires, damaged_nature].\n\n"
        "Respond **only** with a JSON object like this:\n"
        "{\n"
        '  "classification": "category_name"\n'
        "}\n\n"
        "**Examples:**\n"
    )
    for img_path, label in few_shot_examples:
        prompt += f"- Image: {img_path}\n  Classification: {label}\n"
    prompt += "\nNow classify the following image:\n"
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
            sleep_time = delay + jitter
            print(f"⚠️ Retry in {sleep_time:.2f}s (Attempt {attempt +1})")
            await asyncio.sleep(sleep_time)
    print("🚨 Max retries reached. Skipping.")
    return None

# ✅ Batch requests
async def batch_process_gpt4o_requests(test_data, few_shot_examples):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        prompt = create_few_shot_prompt(few_shot_examples)
        for img_path, _ in test_data:
            image_data = encode_image(img_path)
            messages = [
                {"role": "system", "content": "You are a disaster classification assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
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
    true_labels = [label for _, label in test_data]
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

    for i, (img_path, true_label) in enumerate(test_data):
        predicted_label = predicted_labels[i]
        results_text += f"🖼 Image: {img_path} | ✅ True: {true_label} | 🔍 Predicted: {predicted_label}\n"

    print(results_text)
    with open("gpt_image_evaluation_results.txt", "w") as file:
        file.write(results_text)
    print("📁 Results saved to `gpt_image_evaluation_results.txt`")

# ✅ Load data
train_directory = "./split_dataset/train"
test_directory = "./split_dataset/test"

# ✅ Select Few-Shot Examples (1 per class)
few_shot_examples = get_few_shot_examples(train_directory, num_examples=1)

# ✅ Randomly pick 5 test samples per class
test_samples = []
for category in DISASTER_CATEGORIES:
    image_dir = os.path.join(test_directory, category, "images")
    image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    sampled_files = random.sample(image_files, min(5, len(image_files)))
    for img in sampled_files:
        img_path = os.path.join(image_dir, img)
        test_samples.append((img_path, category))

# ✅ Run Evaluation
evaluate_model(test_samples, few_shot_examples)
