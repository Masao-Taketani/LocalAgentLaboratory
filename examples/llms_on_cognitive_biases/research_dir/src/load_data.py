from datasets import Dataset, DatasetDict
import pandas as pd

def prepare_anchoring_dataset():
    # Create a list of questions with baseline and anchoring versions
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

    # Create a pandas DataFrame
    df = pd.DataFrame(questions)

    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_dict(df)

    # Create a DatasetDict with train split
    dataset = DatasetDict({"train": dataset})

    return dataset

# Example usage
dataset = prepare_anchoring_dataset()
print("Custom Dataset Structure:")
print(dataset)
print("\nFirst Example:")
print(dataset["train"][0])