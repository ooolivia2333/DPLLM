import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

def plot_gradient_norms(gradient_stats, type_filters, epsilon, save_dir='figs'):
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    for ftype in type_filters:
        start_epoch = min(gradient_stats.keys())
        end_epoch = max(gradient_stats.keys())

        start_grads_after = gradient_stats[start_epoch]['after_clipping']
        end_grads_after = gradient_stats[end_epoch]['after_clipping']
        start_grads_before = gradient_stats[start_epoch]['before_clipping']
        end_grads_before = gradient_stats[end_epoch]['before_clipping']

        layer_names = set()
        for grads in start_grads_after:
            layer_names.update(grads.keys())

        layer_names = sorted(layer_names)

        cleaned_layer_names = [
            name.replace("module.distilbert.transformer.", "")
                .replace("module.base_model.model.distilbert.transformer.", "")
            for name in layer_names
        ]

        # Calculate means and stds for start and end epochs after clipping
        start_means_after = [
            np.mean([grad[layer] for grad in start_grads_after if layer in grad])
            for layer in layer_names
        ]
        end_means_after = [
            np.mean([grad[layer] for grad in end_grads_after if layer in grad])
            for layer in layer_names
        ]
        start_stds_after = [
            np.std([grad[layer] for grad in start_grads_after if layer in grad])
            for layer in layer_names
        ]
        end_stds_after = [
            np.std([grad[layer] for grad in end_grads_after if layer in grad])
            for layer in layer_names
        ]
        # Calculate stds for start and end epochs before clipping
        start_stds_before = [
            np.std([grad[layer] for grad in start_grads_before if layer in grad])
            for layer in layer_names
        ]
        end_stds_before = [
            np.std([grad[layer] for grad in end_grads_before if layer in grad])
            for layer in layer_names
        ]

        width = 0.35
        x = np.arange(len(cleaned_layer_names))

        # Plot mean after clipping for start and end epochs in the same graph
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, start_means_after, width, label="Start Epoch - After Clipping")
        ax.bar(x + width/2, end_means_after, width, label="End Epoch - After Clipping")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient Norm Mean")
        ax.set_title(f"Gradient Norm Mean by Layer - After Clipping - {ftype.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(cleaned_layer_names, rotation=90)
        ax.legend()
        fig.tight_layout()
        file_name = os.path.join(save_dir, f"dp_adapter_epsilon_{epsilon}_{ftype}_mean_after_clipping.png")
        fig.savefig(file_name)
        print(f"Gradient norms mean after clipping plot saved to {file_name}")

        # Plot standard deviation after clipping for start and end epochs in the same graph
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, start_stds_after, width, label="Start Epoch - After Clipping")
        ax.bar(x + width/2, end_stds_after, width, label="End Epoch - After Clipping")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient Norm Std")
        ax.set_title(f"Gradient Norm Std by Layer - After Clipping - {ftype.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(cleaned_layer_names, rotation=90)
        ax.legend()
        fig.tight_layout()
        file_name = os.path.join(save_dir, f"dp_adapter_epsilon_{epsilon}_{ftype}_std_after_clipping.png")
        fig.savefig(file_name)
        print(f"Gradient norms std after clipping plot saved to {file_name}")

        # Plot standard deviation before clipping for start and end epochs in the same graph
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, start_stds_before, width, label="Start Epoch - Before Clipping")
        ax.bar(x + width/2, end_stds_before, width, label="End Epoch - Before Clipping")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient Norm Std")
        ax.set_title(f"Gradient Norm Std by Layer - Before Clipping - {ftype.capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(cleaned_layer_names, rotation=90)
        ax.legend()
        fig.tight_layout()
        file_name = os.path.join(save_dir, f"dp_adapter_epsilon_{epsilon}_{ftype}_std_before_clipping.png")
        fig.savefig(file_name)
        print(f"Gradient norms std before clipping plot saved to {file_name}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    correct = (y.long() == torch.argmax(preds, dim=1)).float()
    return correct.sum() / len(correct)

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
