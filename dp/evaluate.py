import torch
import numpy as np
from tqdm import tqdm
from utils import binary_accuracy

def evaluate(args, model, test_loader, log_file):
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    model = model.eval().to(device)

    with torch.no_grad():
        for data, attention_mask, label in tqdm(test_loader):
            data = data.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            predictions = model(input_ids=data, attention_mask=attention_mask).logits

            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            losses.append(loss.item())
            accuracies.append(acc.item())

    mean_accuracy = np.mean(accuracies)
    log_file.write(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            np.mean(losses), mean_accuracy * 100
        )
    )
    return mean_accuracy