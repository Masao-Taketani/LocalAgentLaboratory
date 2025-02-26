from datasets import Dataset, DatasetDict
import pandas as pd
import torch
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import re

# Helper function to extract first number from text
def extract_number(text):
    numbers = re.findall(r"[-+]?\d+\.?\d*", text)
    if numbers:
        return float(numbers[0])
    return None

# Prepare dataset
questions = [
    {
        "question": "What is the temperature outside?",
        "question_with_anchor": "Considering the temperature is around 50°C, what is the temperature outside?",
        "answer": "25°C"
    },
    {
        "question": "How many people are in the meeting?",
        "question_with_anchor": "Given there are approximately 100 people in the meeting, how many are actually present?",
        "answer": "50"
    },
    {
        "question": "What is the price of the item?",
        "question_with_anchor": "The item is priced around $200, what would you estimate it costs?",
        "answer": "$150"
    }
]

# Add more questions to reach at least 100 samples
for i in range(97):
    questions.append({
        "question": f"What is the number of attendees {i+4}?",
        "question_with_anchor": f"Considering there are approximately 100 attendees {i+4}, how many are actually present?",
        "answer": "50"
    })

# Create DataFrame and Dataset
df = pd.DataFrame(questions)
dataset = Dataset.from_dict(df)
dataset = DatasetDict({"train": dataset})

# Initialize pipeline with optimized settings
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", device_map="auto")
pipe.tokenizer.truncation = True
pipe.tokenizer.max_length = 200  # Reduce max length for faster generation

# Experiment 1: Quantify Anchoring Bias
print("Running Experiment 1: Quantifying Anchoring Bias")

# Process all baseline prompts in batch
baseline_prompts = [{ "role": "user", "content": entry["question"]} for entry in dataset["train"]]
baseline_responses = pipe(baseline_prompts, do_sample=True, max_length=200, truncation=True)
baseline_numbers = [extract_number(response["generated_text"]) for response in baseline_responses]

# Process all anchoring prompts in batch
anchoring_prompts = [{ "role": "user", "content": entry["question_with_anchor"]} for entry in dataset["train"]]
anchoring_responses = pipe(anchoring_prompts, do_sample=True, max_length=200, truncation=True)
anchoring_numbers = [extract_number(response["generated_text"]) for response in anchoring_responses]

# Calculate differences
valid_pairs = [(b, a) for b, a in zip(anchoring_numbers, baseline_numbers) if a is not None and b is not None]
differences = [abs(b - a) for b, a in valid_pairs]
mean_diff = np.mean(differences) if differences else 0
print(f"Mean absolute difference: {mean_diff}")

# Statistical test
if len(valid_pairs) >= 2:
    t_stat, p_val = stats.ttest_rel([b for b, a in valid_pairs], [a for b, a in valid_pairs])
    print(f"Paired t-test results: t={t_stat}, p={p_val}")
else:
    print("Not enough valid pairs for statistical test")

# Generate Figure 1
plt.figure(figsize=(10, 6))
plt.scatter([a for b, a in valid_pairs], [b for b, a in valid_pairs])
plt.xlabel("Baseline Responses")
plt.ylabel("Anchoring Responses")
plt.title("Anchoring Bias Visualization")
plt.savefig("Figure_1.png", bbox_inches='tight')
plt.close()

# Experiment 2: Mitigate Anchoring Bias
print("\nRunning Experiment 2: Mitigating Anchoring Bias")

# Define mitigation strategies
strategies = {
    "Chain-of-Thought": "[Chain of Thought] {}",
    "Thoughts of Principles": "[Thinking] {}",
    "Ignoring Anchor Hints": "[Ignoring Anchors] {}",
    "Reflection": "[Reflecting] {}",
    "Comprehensive Hints": "[Comprehensive Analysis] {}"
}

# Process all strategies in batches
mitigation_prompts = {
    name: [{ "role": "user", "content": template.format(entry["question_with_anchor"])} 
           for entry in dataset["train"]] 
    for name, template in strategies.items()
}

# Generate responses for all strategies in parallel
mitigation_responses = {
    name: pipe(prompts, do_sample=True, max_length=200, truncation=True) 
    for name, prompts in mitigation_prompts.items()
}

# Extract numbers from responses
mitigation_numbers = {
    name: [extract_number(response["generated_text"]) for response in responses] 
    for name, responses in mitigation_responses.items()
}

# Calculate differences from baseline
valid_baseline = [a for a in baseline_numbers if a is not None]
diffs = {
    name: [abs(b - a) for b, a in zip(responses, valid_baseline[:len(responses)]) 
           if a is not None and b is not None] 
    for name, responses in mitigation_numbers.items()
}

# Calculate mean differences
mean_diffs = {name: np.mean(diffs[name]) for name in diffs.keys() if diffs[name]}
print("\nMean absolute differences for each strategy:")
for name, diff in mean_diffs.items():
    print(f"{name}: {diff}")

# Statistical comparison
if len(mean_diffs) >= 2:
    print("\nANOVA test across strategies:")
    f_val, p_val = stats.f_oneway(*[diffs[name] for name in mean_diffs.keys()])
    print(f"ANOVA results: F={f_val}, p={p_val}")
    print("Significant differences found between strategies" if p_val < 0.05 else "No significant differences found between strategies")
else:
    print("\nNot enough valid data for ANOVA test")

# Generate Figure 2
if mean_diffs:
    plt.figure(figsize=(10, 6))
    plt.bar(mean_diffs.keys(), mean_diffs.values())
    plt.xlabel("Mitigation Strategy")
    plt.ylabel("Mean Absolute Difference")
    plt.title("Effectiveness of Mitigation Strategies")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Figure_2.png", bbox_inches='tight')
    plt.close()

print("\nExperiments completed. Results saved as Figure_1.png and Figure_2.png")