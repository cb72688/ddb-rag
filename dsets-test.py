from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

logger = logging.getLogger("dataset_test")


def main():
    logger.info("Setting dataset variable")
    print("Setting dataset variable")
    dataset = load_dataset("nick007x/arxiv-papers")

    print("\n=== Dataset Info ===")
    print(f"Splits: {dataset.keys()}")
    print(f"Columns: {dataset['train'].column_names}")
    print(f"\nFirst example (first 500 characters of each field):")
    first_example = dataset['train'][0]
    for key, value in first_example.items():
        if isinstance(value, str):
            print(f"{key}: {value[:500]}")
        else:
            print(f"{key}: {value}")
    print("\n====================\n")

    logger.info("Setting model variable")
    print("Setting model variable")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased")
    logger.info("Setting tokenizer variable")
    print("Setting tokenizer variable")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode(examples):
        # arxiv-papers dataset typically has columns such as 'title', 'abstract', 'authors', 'categories'
        # 'doi', 'created'
        # BERT classifications probably only need 'title' and 'abstract' tokenized
        if "title" in examples and "abstract" in examples:
            return tokenizer(examples["title"],
                             examples["abstract"],
                             truncation=True,
                             padding="max_length",
                             max_length=8192
                             )
        elif "title" in examples and "summary" in examples:
            return tokenizer(examples["title"],
                             examples["summary"],
                             truncation=True,
                             padding="max_length",
                             max_length=8192
                             )
        else:
            # Fallback to only tokenize the title if other columns don't exist
            return tokenizer(
                examples["title"],
                truncation=True,
                padding="max_length",
                max_length=8192
            )

    logger.info("Mapping dataset")
    print("Mapping dataset (may take a while for larger datasets")

    # Map the encoding function to the dataset
    encoded_dataset = dataset.map(encode, batched=True)

    logger.info("Printing sample results")
    print("\nPrinting first 3 encoded examples:")
    # Print first 3 examples from the train split
    for i in range(min(3, len(encoded_dataset['train']))):
        print(f"\n--- Example {i+1} ---")
        example = encoded_dataset['train'][i]
        # Print only the keys and shapes of the tensor
        for key, value in example.items():
            if isinstance(value, list):
                print(f"{key}: list of length {len(value)}")
            else:
                print(f"{key}: {value}")

    print("\nSript completed successfully!")


if __name__ == "__main__":
    logger.info("Running main function...")
    print("Running main function...")
    main()
