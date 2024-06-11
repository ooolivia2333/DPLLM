import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification
from metrics import compute_metrics
import transformers
from adapter import make_adapter, count_parameters
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
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, compute_metrics, device, model_type='standard', log_file='training_logs.txt'):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, compute_metrics=compute_metrics)
        self.gradient_stats = {}  # Dictionary to store gradients by epoch
        self.device = device
        self.start_epoch = 0
        self.end_epoch = args.num_train_epochs - 1
        self.log_file = log_file
        self.model_type = model_type

    def training_step(self, model, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()  # Ensure the loss is scalar

        loss.backward()

        # Capture gradients at the start and end epochs
        current_epoch = int(self.state.epoch)
        if current_epoch == self.start_epoch or current_epoch == self.end_epoch:
            self.capture_per_sample_gradients(model, current_epoch)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        model.zero_grad()
        return loss.detach()

    def capture_per_sample_gradients(self, model, current_epoch):
        grad_data = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norms = param.grad.view(param.grad.size(0), -1).norm(2, dim=1)
                grad_data[name] = grad_norms.mean().item()

        if current_epoch not in self.gradient_stats:
            self.gradient_stats[current_epoch] = []

        self.gradient_stats[current_epoch].append(grad_data)

    def log_metrics(self, metrics, prefix=""):
        with open(self.log_file, 'a') as logf:
            logf.write(f"{prefix} Metrics: {metrics}\n")

    def log(self, logs):
        super().log(logs)
        self.log_metrics(logs, prefix=f"Epoch {int(self.state.epoch)}")

    def plot_gradient_norms(self, save_dir='figs'):
        os.makedirs(save_dir, exist_ok=True)

        start_epoch = self.start_epoch
        end_epoch = self.end_epoch

        start_grads = self.gradient_stats.get(start_epoch, [])
        end_grads = self.gradient_stats.get(end_epoch, [])

        layer_names = set()
        for grads in start_grads:
            layer_names.update(grads.keys())

        layer_names = sorted(layer_names)

        cleaned_layer_names = [
            name.replace("module.distilbert.transformer.", "")
                .replace("module.base_model.model.distilbert.transformer.", "")
                .replace("module.distilbert.", "")
                .replace("module.base_model.model", "")
            for name in layer_names
        ]

        start_means = [
            np.mean([grad[layer] for grad in start_grads if layer in grad])
            for layer in layer_names
        ]
        end_means = [
            np.mean([grad[layer] for grad in end_grads if layer in grad])
            for layer in layer_names
        ]
        start_stds = [
            np.std([grad[layer] for grad in start_grads if layer in grad])
            for layer in layer_names
        ]
        end_stds = [
            np.std([grad[layer] for grad in end_grads if layer in grad])
            for layer in layer_names
        ]

        width = 0.35
        x = np.arange(len(cleaned_layer_names))

        # Plot mean for start and end epochs in the same graph
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, start_means, width, label="Start Epoch")
        ax.bar(x + width/2, end_means, width, label="End Epoch")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient Norm Mean")
        ax.set_title("Gradient Norm Mean by Layer")
        ax.set_xticks(x)
        ax.set_xticklabels(cleaned_layer_names, rotation=90)
        ax.legend()
        fig.tight_layout()
        file_name = os.path.join(save_dir, f"{self.model_type}_gradient_norm_mean.png")
        fig.savefig(file_name)
        print(f"Gradient norms mean plot saved to {file_name}")

        # Plot standard deviation for start and end epochs in the same graph
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, start_stds, width, label="Start Epoch")
        ax.bar(x + width/2, end_stds, width, label="End Epoch")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient Norm Std")
        ax.set_title("Gradient Norm Std by Layer")
        ax.set_xticks(x)
        ax.set_xticklabels(cleaned_layer_names, rotation=90)
        ax.legend()
        fig.tight_layout()
        file_name = os.path.join(save_dir, f"{self.model_type}_gradient_norm_std.png")
        fig.savefig(file_name)
        print(f"Gradient norms std plot saved to {file_name}")

def train_standard(tokenized_datasets, tokenizer, device):
    standard_num_epochs = 3
    standard_train_batch_size = 32
    standard_eval_batch_size = 64
    standard_lr = 5e-5
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased").to(device)
    model.train()

    transformers.logging.set_verbosity_info()
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=standard_num_epochs,
        per_device_train_batch_size=standard_train_batch_size,
        per_device_eval_batch_size=standard_eval_batch_size,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        optim='adamw_hf',
        learning_rate=standard_lr
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        device=device,
        log_file='standard_training_logs.txt',
        model_type='standard'
    )

    trainer.train()
    trainer.plot_gradient_norms()

def train_adapter(tokenized_datasets, tokenizer, device):
    bottleneck_size = 32
    adapter_num_epochs = 5
    adapter_train_batch_size = 32
    adapter_eval_batch_size = 64
    adapter_lr = 5e-4

    adapter_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    for param in adapter_model.parameters():
        param.requires_grad = False

    total_size = 0
    for block_idx in range(6):
        orig_layer_1 = adapter_model.distilbert.transformer.layer[block_idx].attention.out_lin
        adapter_layers_1 = make_adapter(
            in_dim=orig_layer_1.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_1.out_features)

        new_1 = torch.nn.Sequential(orig_layer_1, *adapter_layers_1)
        adapter_model.distilbert.transformer.layer[block_idx].attention.out_lin = new_1
        total_size += count_parameters(adapter_layers_1)

        orig_layer_2 = adapter_model.distilbert.transformer.layer[block_idx].ffn.lin2
        adapter_layers_2 = make_adapter(
            in_dim=orig_layer_2.out_features,
            bottleneck_dim=bottleneck_size,
            out_dim=orig_layer_2.out_features)

        new_2 = torch.nn.Sequential(orig_layer_2, *adapter_layers_2)
        adapter_model.distilbert.transformer.layer[block_idx].ffn.lin2 = new_2
        total_size += count_parameters(adapter_layers_2)

    print("Number of adapter parameters added:", total_size)
    adapter_model.to(device)
    adapter_model.train()

    transformers.logging.set_verbosity_info()

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=adapter_num_epochs,
        per_device_train_batch_size=adapter_train_batch_size,
        per_device_eval_batch_size=adapter_eval_batch_size,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        optim='adamw_hf',
        learning_rate=adapter_lr
    )

    trainer = CustomTrainer(
        model=adapter_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        device=device,
        log_file='adapter_training_logs.txt',
        model_type='adapter'
    )

    trainer.train()
    trainer.plot_gradient_norms()

def train_lora(tokenized_datasets, tokenizer, device):
    lora_num_epochs = 3
    lora_train_batch_size = 32
    lora_eval_batch_size = 64
    lora_lr = 5e-4

    lora_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=['q_lin', 'v_lin'])
    lora_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3, return_dict=True)
    lora_model = get_peft_model(lora_model, lora_config)
    lora_model.print_trainable_parameters()

    lora_model.to(device)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=lora_num_epochs,
        per_device_train_batch_size=lora_train_batch_size,
        per_device_eval_batch_size=lora_eval_batch_size,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        optim='adamw_hf',
        learning_rate=lora_lr
    )

    trainer = CustomTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        device=device,
        log_file='lora_training_logs.txt',
        model_type='lora'
    )

    trainer.train()
    trainer.plot_gradient_norms()

def train_ia3(tokenized_datasets, tokenizer, device):
    ia3_num_epochs = 3
    ia3_train_batch_size = 32
    ia3_eval_batch_size = 64
    ia3_lr = 7e-3

    ia3_config = IA3Config(task_type="SEQ_CLS", inference_mode=False, target_modules=['q_lin', 'v_lin', 'out_lin'], feedforward_modules=['out_lin'])
    ia3_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3, return_dict=True)
    ia3_model = get_peft_model(ia3_model, ia3_config)
    ia3_model.print_trainable_parameters()

    ia3_model.to(device)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=ia3_num_epochs,
        per_device_train_batch_size=ia3_train_batch_size,
        per_device_eval_batch_size=ia3_eval_batch_size,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        optim='adamw_hf',
        learning_rate=ia3_lr
    )

    trainer = CustomTrainer(
        model=ia3_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        device=device,
        log_file='ia3_training_logs.txt',
        model_type='ia3'
    )

    trainer.train()
    trainer.plot_gradient_norms()
