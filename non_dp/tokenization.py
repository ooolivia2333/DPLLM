from transformers import DistilBertTokenizer

def tokenize_function(examples):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(examples['text'], truncation=True, max_length=512)
