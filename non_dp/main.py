import torch
from transformers import DistilBertTokenizer
from data_prep import load_and_prepare_dataset
from tokenization import tokenize_function
from training import train_standard, train_adapter, train_lora, train_ia3

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    datasets = load_and_prepare_dataset("imdb")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_datasets = datasets.map(tokenize_function, batched=True)

    # Train different models
    train_standard(tokenized_datasets, tokenizer, device)
    train_adapter(tokenized_datasets, tokenizer, device)
    train_lora(tokenized_datasets, tokenizer, device)
    train_ia3(tokenized_datasets, tokenizer, device)

if __name__ == "__main__":
    main()