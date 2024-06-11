from datasets import load_dataset, DatasetDict

def load_and_prepare_dataset(path, test_size=0.15, seed=42):
    dataset = load_dataset(path)
    subset_train = dataset['train'].train_test_split(test_size=test_size, seed=seed)
    subset_test = dataset['test'].train_test_split(test_size=test_size, seed=seed)
    datasets = DatasetDict({
        'train': dataset['train'],
        'test': dataset['test']
    })
    return datasets
