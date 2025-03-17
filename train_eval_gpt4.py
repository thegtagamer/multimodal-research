import os
import random
import base64
import time
import json
import requests
import asyncio
import aiohttp
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ OpenAI API Setup
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o"
MAX_RETRIES = 5
DISASTER_CATEGORIES = ["flood", "non_damage", "damaged_infrastructure", "human_damage", "fires", "damaged_nature"]

# ✅ Function to send OpenAI API requests asynchronously
async def async_openai_request(session, api_key, model, messages):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,  # ✅ Lower temperature for better consistency
        "top_p": 0.7,
        "max_tokens": 50
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            delay = min(2 ** attempt, 10)  # Exponential Backoff
            jitter = random.uniform(0, delay * 0.1)
            sleep_time = delay + jitter
            print(f"⚠️ API Rate Limit! Retrying in {sleep_time:.2f}s...")
            await asyncio.sleep(sleep_time)

    print("🚨 Maximum retries reached. Skipping this request.")
    return None

# ✅ Encode image as Base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# ✅ Few-shot Learning: Select category-aware examples
def get_few_shot_examples(data_path, num_examples=5):
    examples = []
    for category in DISASTER_CATEGORIES:
        image_dir = os.path.join(data_path, category, "images")
        image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

        if len(image_files) < num_examples:
            print(f"⚠️ Warning: Not enough samples in {category}, using available {len(image_files)} images.")

        sampled_images = random.sample(image_files, min(num_examples, len(image_files)))
        
        for img in sampled_images:
            img_path = os.path.join(image_dir, img)
            examples.append((img_path, category))  # (image path, label)
    
    return examples

# ✅ JSON-based Prompt for GPT-4o
def create_few_shot_prompt(few_shot_examples):
    prompt = (
        "You are a disaster classification assistant. Your task is to classify images into one of the categories: "
        "[flood, non_damage, damaged_infrastructure, human_damage, fires, damaged_nature].\n\n"
        "Respond **only** with a JSON object formatted like this:\n"
        "{\n"
        '  "classification": "category_name"\n'
        "}\n\n"
        "**Examples:**\n"
    )

    for img_path, label in few_shot_examples:
        prompt += f"\nExample:\n- Image: {img_path}\n- Classification: {label}\n"

    prompt += "\nNow, classify the following image:\n"
    return prompt

# ✅ Parallelized GPT-4o Requests
async def batch_process_gpt4o_requests(test_data, few_shot_examples):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for img_path, _ in test_data:
            image_data = encode_image(img_path)
            prompt = create_few_shot_prompt(few_shot_examples)

            messages = [
                {"role": "system", "content": "You are a disaster classification assistant."},
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}  # ✅ Fixed format
                    ]
                }
            ]

            tasks.append(async_openai_request(session, OPENAI_API_KEY, OPENAI_MODEL, messages))

        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            if response and "choices" in response:
                try:
                    prediction = json.loads(response["choices"][0]["message"]["content"].strip())
                    predicted_label = prediction.get("classification", "non_damage")
                except json.JSONDecodeError:
                    predicted_label = "non_damage"  # Default label if JSON parsing fails

                results.append(predicted_label)
            else:
                results.append("non_damage")  # Default label if response fails

    return results

# ✅ Evaluation Metrics
# ✅ Evaluation Metrics (Updated to Save Results)
def evaluate_model(test_data, few_shot_examples):
    true_labels = [label for _, label in test_data]
    predicted_labels = asyncio.run(batch_process_gpt4o_requests(test_data, few_shot_examples))

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="weighted", zero_division=0)

    results_text = (
        "\n🎯 **Evaluation Metrics:**\n"
        f"✅ Accuracy: {accuracy:.4f}\n"
        f"✅ Precision: {precision:.4f}\n"
        f"✅ Recall: {recall:.4f}\n"
        f"✅ F1-score: {f1:.4f}\n\n"
        "**Test Sample Predictions:**\n"
    )

    for i, (img_path, true_label) in enumerate(test_data):
        predicted_label = predicted_labels[i]
        results_text += f"🖼 Image: {img_path} | ✅ True: {true_label} | 🔍 Predicted: {predicted_label}\n"

    print(results_text)

    # ✅ Save Results to File
    with open("gpt_evaluation_results.txt", "w") as file:
        file.write(results_text)

    print("📁 Results saved to `evaluation_results.txt`")



# ✅ Load Training and Testing Data
train_directory = "./split_dataset/train"
test_directory = "./split_dataset/test"

# ✅ Select Few-Shot Examples
few_shot_examples = get_few_shot_examples(train_directory, num_examples=7)

# ✅ Select Test Samples
test_samples = get_few_shot_examples(test_directory, num_examples=100)  # Pick 10 per category

# ✅ Run Evaluation
evaluate_model(test_samples, few_shot_examples)
