import torch
from tqdm import tqdm
from utils import binary_accuracy, capture_per_sample_gradients, capture_per_sample_gradients_after_clipping

def train(args, model, train_loader, optimizer, privacy_engine, epoch, gradient_stats, log_file):
    criterion = torch.nn.CrossEntropyLoss()
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

        if epoch == 1 or epoch == args.epochs:
            capture_per_sample_gradients(model, gradient_stats, epoch, phase="before_clipping")

        if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if epoch == 1 or epoch == args.epochs:
                capture_per_sample_gradients_after_clipping(model, gradient_stats, epoch, args.max_per_sample_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

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
            f"Train Loss: {loss.item():.6f} "
            f"Train Accuracy: {acc.item():.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})\n"
        )
    else:
        log_file.write(
            f"Train Epoch: {epoch} \t Loss: {loss.item():.6f} \t Accuracy: {acc.item():.6f}\n"
        )
