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
import matplotlib.pyplot as plt
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# def capture_per_sample_gradients(model, gradient_stats, current_epoch, phase="before_clipping"):
#     grad_data = {}
#     for name, param in model.named_parameters():
#         if param.requires_grad and param.grad_sample is not None:
#             # Compute the norm for each sample
#             grad_norms = param.grad_sample.norm(2, dim=tuple(range(1, param.grad_sample.dim())))
#             grad_data[name] = grad_norms.mean().item()  # Take the mean of norms for simplicity
    
#     if current_epoch not in gradient_stats:
#         gradient_stats[current_epoch] = {"before_clipping": [], "after_clipping": []}
    
#     gradient_stats[current_epoch][phase].append(grad_data)
#     # print(f"{phase} gradients: {gradient_stats[current_epoch][phase]}")

# def capture_gradient_norms(model, gradient_stats, current_epoch, phase="before_clipping"):
#     grad_data = {}
#     for name, param in model.named_parameters():
#         if param.requires_grad and param.grad is not None:
#             grad_norm = param.grad.norm().item()
#             grad_data[name] = grad_norm
    
#     if current_epoch not in gradient_stats:
#         gradient_stats[current_epoch] = {"before_clipping": [], "after_clipping": []}
    
#     gradient_stats[current_epoch][phase].append(grad_data)
#     print(f"{phase} gradients: {gradient_stats[current_epoch][phase]}")

def plot_gradient_norms(gradient_stats, type_filters, epsilon):
    for ftype in type_filters:
        start_epoch = min(gradient_stats.keys())
        end_epoch = max(gradient_stats.keys())

        start_grads_before = gradient_stats[start_epoch]['before_clipping']
        start_grads_after = gradient_stats[start_epoch]['after_clipping']
        end_grads_before = gradient_stats[end_epoch]['before_clipping']
        end_grads_after = gradient_stats[end_epoch]['after_clipping']

        layer_names = set()
        for grads in start_grads_before:
            layer_names.update(grads.keys())

        layer_names = sorted(layer_names)

        cleaned_layer_names = [
            name.replace("module.distilbert.transformer.", "")
                .replace("module.base_model.model.distilbert.transformer.", "")
            for name in layer_names
        ]

        start_norms_before = [
            np.mean([grad[layer] for grad in start_grads_before if layer in grad])
            for layer in layer_names
        ]
        start_norms_after = [
            np.mean([grad[layer] for grad in start_grads_after if layer in grad])
            for layer in layer_names
        ]
        end_norms_before = [
            np.mean([grad[layer] for grad in end_grads_before if layer in grad])
            for layer in layer_names
        ]
        end_norms_after = [
            np.mean([grad[layer] for grad in end_grads_after if layer in grad])
            for layer in layer_names
        ]

        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.2

        x = np.arange(len(cleaned_layer_names))

        rects2 = ax.bar(x, start_norms_after, width, label="Start Epoch - After Clipping")
        rects4 = ax.bar(x + 2 * width, end_norms_after, width, label="End Epoch - After Clipping")

        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient Norm")
        ax.set_title(f"Gradient Norms by Layer - {ftype.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(cleaned_layer_names, rotation=90)
        ax.legend()

        fig.tight_layout()

        file_name = f"dp_adapter_epsilon_{epsilon}_{ftype}_norms.png"
        fig.savefig(file_name)
        print(f"Gradient norms plot saved to {file_name}")

def make_adapter(in_dim, bottleneck_dim, out_dim):
    adapter_layers = torch.nn.Sequential(
        torch.nn.Linear(in_dim, bottleneck_dim),
        torch.nn.GELU(),
        torch.nn.Linear(bottleneck_dim, out_dim),
    )
    return adapter_layers

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def calc_sample_norms(named_params):
    return torch.stack([p.grad_sample.view(len(p.grad_sample), -1).norm(2, dim=1) for _, p in named_params], dim=1).norm(2, dim=1)

def calc_clipping_factors(norms, max_norm):
    return torch.clamp(max_norm / (norms + 1e-6), max=1.0)

def capture_per_sample_gradients(model, gradient_stats, current_epoch, phase="before_clipping"):
    grad_data = {}
    for name, param in model.named_parameters():
        if param.requires_grad and hasattr(param, 'grad_sample'):
            grad_norms = param.grad_sample.view(param.grad_sample.size(0), -1).norm(2, dim=1)
            grad_data[name] = grad_norms.mean().item()

    if current_epoch not in gradient_stats:
        gradient_stats[current_epoch] = {"before_clipping": [], "after_clipping": []}

    gradient_stats[current_epoch][phase].append(grad_data)

def capture_per_sample_gradients_after_clipping(model, gradient_stats, current_epoch, max_norm):
    grad_data = {}
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad and hasattr(param, 'grad_sample')]
    all_norms = calc_sample_norms(named_params)
    clipping_factors = calc_clipping_factors(all_norms, max_norm)
    
    for (name, param) in named_params:
        grad_norms = (param.grad_sample.view(param.grad_sample.size(0), -1) * clipping_factors.unsqueeze(1).to(param.grad_sample.device)).norm(2, dim=1)
        grad_data[name] = grad_norms.mean().item()

    if current_epoch not in gradient_stats:
        gradient_stats[current_epoch] = {"before_clipping": [], "after_clipping": []}

    gradient_stats[current_epoch]["after_clipping"].append(grad_data)

# Usage in the training loop
def train(args, model, train_loader, optimizer, privacy_engine, epoch, gradient_stats, log_file):
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

        # Capture gradient norms before clipping
        if epoch == 1 or epoch == args.epochs:
            capture_per_sample_gradients(model, gradient_stats, epoch, phase="before_clipping")

        if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Capture gradient norms after clipping
            if epoch == 1 or epoch == args.epochs:
                capture_per_sample_gradients_after_clipping(model, gradient_stats, epoch, args.max_per_sample_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

        # Clear cache after each batch
        torch.cuda.empty_cache()

        # Log batch loss and accuracy
        if (batch_idx + 1) % args.log_interval == 0:
            if not args.disable_dp:
                epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
                log_file.write(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}\tAccuracy: {acc.item():.6f}\t"
                    f"(ε = {epsilon:.2f}, δ = {args.delta})\n"
                )
            else:
                log_file.write(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}\tAccuracy: {acc.item():.6f}\n"
                )

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        log_file.write(
            f"Train Epoch: {epoch} \t"
            f"Train Loss: {np.mean(losses):.6f} "
            f"Train Accuracy: {np.mean(accuracies):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})\n"
        )
    else:
        log_file.write(
            f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} \t Accuracy: {np.mean(accuracies):.6f}\n"
        )


def evaluate(args, model, test_loader, log_file):
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
    log_file.write(
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
        default=5,
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

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    # frozen other parts
    for param in model.parameters():
        param.requires_grad = False

    total_size = 0
    bottleneck_size = 32

    for block_idx in range(6):

        ###################################################
        # insert 1st adapter layer into transformer block
        ###################################################

        orig_layer_1 = model.distilbert.transformer.layer[block_idx].attention.out_lin

        adapter_layers_1 = make_adapter(
            in_dim=orig_layer_1.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_1.out_features)

        new_1 = torch.nn.Sequential(orig_layer_1, *adapter_layers_1)
        model.distilbert.transformer.layer[block_idx].attention.out_lin = new_1

        total_size += count_parameters(adapter_layers_1)

        ###################################################
        # insert 2nd adapter layer into transformer block
        ###################################################

        orig_layer_2 = model.distilbert.transformer.layer[block_idx].ffn.lin2

        adapter_layers_2 = make_adapter(
            in_dim=orig_layer_2.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_2.out_features)

        new_2 = torch.nn.Sequential(orig_layer_2, *adapter_layers_2)
        model.distilbert.transformer.layer[block_idx].ffn.lin2 = new_2

        total_size += count_parameters(adapter_layers_2)


    print("Number of adapter parameters added:", total_size)

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

    gradient_stats = {}  # Initialize gradient statistics dictionary

    mean_accuracy = 0

    log_file_name = f"dp_adapter_epsilon_{args.epsilon}_training.log"
    with open(log_file_name, "w") as log_file:
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, privacy_engine, epoch, gradient_stats, log_file)
            mean_accuracy = evaluate(args, model, test_loader, log_file)

    # Plot gradient norms after training
    plot_gradient_norms(gradient_stats, ['attention', 'ffn'], args.epsilon)

    torch.save(mean_accuracy, "run_results_imdb_classification.pt")

if __name__ == "__main__":
    main()
