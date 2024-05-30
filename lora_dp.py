#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training sentiment prediction model on IMDB movie reviews dataset.
Architecture and reference results from https://arxiv.org/pdf/1911.11607.pdf
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from opacus import PrivacyEngine
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AutoModelForSequenceClassification
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    IA3Config,
)
from opacus.accountants.utils import get_noise_multiplier

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def binary_accuracy(preds, y):
    correct = (y.long() == torch.argmax(preds, dim=1)).float()
    acc = correct.sum() / len(correct)
    return acc

def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    attention_mask = pad_sequence(
        [elem["attention_mask"] for elem in batch],
        batch_first=True,
        padding_value=0,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, attention_mask, y

def train(args, model, train_loader, optimizer, privacy_engine, epoch):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    model = model.train().to(device)

    optimizer.zero_grad()

    for batch_idx, (data, attention_mask, label) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)

        predictions = model(input_ids=data, attention_mask=attention_mask).logits
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()

        if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())
        accuracies.append(acc.item())

        # Clear cache after each batch
        torch.cuda.empty_cache()

        # Log batch loss and accuracy
        if (batch_idx + 1) % args.log_interval == 0:
            if not args.disable_dp:
                epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}\tAccuracy: {acc.item():.6f}\t"
                    f"(ε = {epsilon:.2f}, δ = {args.delta})"
                )
            else:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}\tAccuracy: {acc.item():.6f}"
                )

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Train Loss: {np.mean(losses):.6f} "
            f"Train Accuracy: {np.mean(accuracies):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )
    else:
        print(
            f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} \t Accuracy: {np.mean(accuracies):.6f}"
        )

def evaluate(args, model, test_loader):
    criterion = nn.CrossEntropyLoss()
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
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            np.mean(losses), mean_accuracy * 100
        )
    )
    return mean_accuracy

def main():
    parser = argparse.ArgumentParser(
        description="Opacus IMDB Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        metavar="B",
        help="input batch size for training and test",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,  # We'll calculate this based on epsilon
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        metavar="SL",
        help="Longer sequences will be cut to this length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla optimizer",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root", type=str, default="../imdb", help="Where IMDB is/will be stored"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=4.0,  # Set the target epsilon
        help="Target epsilon",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    raw_dataset = load_dataset("imdb", cache_dir=args.data_root)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = raw_dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, padding="max_length", max_length=args.max_sequence_length
        ),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Update delta to be 1/size of the dataset
    args.delta = 1.0 / len(train_dataset)

    # Compute noise multiplier to achieve the desired epsilon
    if args.sigma is None:
        args.sigma = get_noise_multiplier(
            target_epsilon=args.epsilon, 
            target_delta=args.delta, 
            sample_rate=args.batch_size / len(train_dataset), 
            epochs=args.epochs
        )

    # Logging for verification
    print(f"Calculated noise multiplier (sigma): {args.sigma}")

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        collate_fn=padded_collate,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=padded_collate,
        pin_memory=True,
    )

    lora_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=['q_lin', 'v_lin'])
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3, return_dict=True)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    privacy_engine = None
    if not args.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=args.epsilon,
            target_delta=1.0 / len(train_dataset),
            epochs=args.epochs,
            max_grad_norm=args.max_per_sample_grad_norm,
            poisson_sampling=True,
        )

    mean_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, privacy_engine, epoch)
        mean_accuracy = evaluate(args, model, test_loader)

    torch.save(mean_accuracy, "run_results_imdb_classification.pt")

if __name__ == "__main__":
    main()
