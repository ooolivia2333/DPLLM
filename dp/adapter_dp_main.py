import argparse
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from train import train
from evaluate import evaluate
from utils import padded_collate, plot_gradient_norms
from adapter import make_adapter, count_parameters
import os
import torch.optim as optim

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(
        description="Opacus IMDB Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32, metavar="B",
        help="input batch size for training and test"
    )
    parser.add_argument(
        "-n", "--epochs", type=int, default=5, metavar="N",
        help="number of epochs to train"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, metavar="LR",
        help="learning rate"
    )
    parser.add_argument(
        "--sigma", type=float, default=None, metavar="S",
        help="Noise multiplier"
    )
    parser.add_argument(
        "-c", "--max-per-sample-grad_norm", type=float, default=1.0, metavar="C",
        help="Clip per-sample gradients to this norm"
    )
    parser.add_argument(
        "--delta", type=float, default=1e-5, metavar="D",
        help="Target delta"
    )
    parser.add_argument(
        "--max-sequence-length", type=int, default=256, metavar="SL",
        help="Longer sequences will be cut to this length"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="GPU ID for this process"
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False,
        help="Save the trained model"
    )
    parser.add_argument(
        "--disable-dp", action="store_true", default=False,
        help="Disable privacy training and just train with vanilla optimizer"
    )
    parser.add_argument(
        "--secure-rng", action="store_true", default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost"
    )
    parser.add_argument(
        "--data-root", type=str, default="../imdb",
        help="Where IMDB is/will be stored"
    )
    parser.add_argument(
        "-j", "--workers", default=2, type=int, metavar="N",
        help="number of data loading workers"
    )
    parser.add_argument(
        "--accumulation-steps", type=int, default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, metavar="N",
        help="how many batches to wait before logging training status"
    )
    parser.add_argument(
        "--epsilon", type=float, default=4.0,
        help="Target epsilon"
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

    args.delta = 1.0 / len(train_dataset)

    if args.sigma is None:
        args.sigma = get_noise_multiplier(
            target_epsilon=args.epsilon, 
            target_delta=args.delta, 
            sample_rate=args.batch_size / len(train_dataset), 
            epochs=args.epochs
        )

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
    for param in model.parameters():
        param.requires_grad = False

    total_size = 0
    bottleneck_size = 32

    for block_idx in range(6):
        orig_layer_1 = model.distilbert.transformer.layer[block_idx].attention.out_lin
        adapter_layers_1 = make_adapter(
            in_dim=orig_layer_1.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_1.out_features
        )
        new_1 = torch.nn.Sequential(orig_layer_1, *adapter_layers_1)
        model.distilbert.transformer.layer[block_idx].attention.out_lin = new_1
        total_size += count_parameters(adapter_layers_1)

        orig_layer_2 = model.distilbert.transformer.layer[block_idx].ffn.lin2
        adapter_layers_2 = make_adapter(
            in_dim=orig_layer_2.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_2.out_features
        )
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

    gradient_stats = {}

    log_file_name = f"dp_adapter_epsilon_{args.epsilon}_training.log"
    with open(log_file_name, "w") as log_file:
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, privacy_engine, epoch, gradient_stats, log_file)
            mean_accuracy = evaluate(args, model, test_loader, log_file)

    plot_gradient_norms(gradient_stats, ['attention', 'ffn'], args.epsilon)

    torch.save(mean_accuracy, "run_results_imdb_classification.pt")

if __name__ == "__main__":
    main()
