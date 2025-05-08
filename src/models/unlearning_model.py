import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import LoRAModel
import argparse
from tqdm.auto import tqdm
#import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import copy


class UnlearningModel(torch.nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        args: argparse.Namespace,
    ):
        super().__init__()
        self._device: torch.device = torch.device(
            args.device
            if args.device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._args: argparse.Namespace = args

        lora_alpha = float(args.lora_rank) if args.lora_alpha is None else args.lora_alpha

        self._llm: LoRAModel = LoRAModel(model, args.lora_rank, lora_alpha)
        self._tokenizer = tokenizer

        self.logdir, self._writers = args.logdir, {}
        
        # Initialize loss tracking lists
        self.forget_losses = []
        self.retain_losses = []
        self.total_losses = []
        self.current_step = 0

        self._optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        self.to(self._device)

    def extract_model(self) -> AutoModelForCausalLM:
        return self._llm.extract_model()

    def unlearn(
        self,
        train_data: DataLoader,
        args: argparse.Namespace,
    ):
        train_steps = 0
        for epoch in range(args.epochs):
            self.train()
            epoch_message = f"Epoch={epoch + 1}/{args.epochs}"
            data_and_progress = tqdm(
                train_data, epoch_message, unit="batch", leave=False
            )

            total_loss = 0.0
            npo_loss = 0.0
            retain_loss = 0.0
            kl_retain_loss = 0.0
            forget_count = 0
            retain_count = 0

            for inputs, answer_mask, ranges, tasks in data_and_progress:
                inputs.input_ids = inputs.input_ids.to(self._device)
                inputs.attention_mask = inputs.attention_mask.to(self._device)
                answer_mask = answer_mask.to(self._device)
                ranges = ranges.to(self._device)
                tasks = tasks.to(self._device)

                losses = self.train_step(inputs, answer_mask, tasks)
                
                # Record losses
                self.current_step += 1
                self.forget_losses.append(losses["npo_loss"])
                self.retain_losses.append(losses["retain_loss"])
                self.total_losses.append(losses["total_loss"])

                train_steps += 1
                if (
                    args.lora_merge_every > 0
                    and train_steps % args.lora_merge_every == 0
                ):
                    self._llm.merge_loras()

                total_loss += losses["total_loss"]
                npo_loss += losses["npo_loss"]
                retain_loss += losses["retain_loss"]
                kl_retain_loss += losses["kl_retain_loss"]
                forget_count += losses["forget_count"]
                retain_count += losses["retain_count"]

                data_and_progress.set_postfix(
                    {"loss": total_loss / (forget_count + retain_count)}
                )

            self.add_logs(
                "train",
                {
                    "total_loss": total_loss / (forget_count + retain_count),
                    "npo_loss": npo_loss / forget_count,
                    "retain_loss": retain_loss / retain_count,
                    "kl_retain_loss": kl_retain_loss / retain_count,
                },
                epoch + 1,
            )

            if (args.evaluate_every >= 1) and ((epoch + 1) % args.evaluate_every == 0):
                self.eval(epoch + 1)

            if (args.save_every >= 1) and (((epoch + 1) % args.save_every) == 0):
                print("Saving checkpoint")
                self.save_checkpoint(os.path.join(args.logdir, f"checkpoint_{epoch}"))
        
        # Plot losses at the end of training
        self.plot_losses()
        pass

    def train_step(self, inputs, answer_mask, tasks):
        # reference output
        self._llm.only_backbone(True)
        with torch.no_grad():
            reference_output = self._llm(
                torch.as_tensor(inputs.input_ids),
                attention_mask=torch.as_tensor(inputs.attention_mask),
            )

        # actual output
        self._llm.only_backbone(False)
        outputs = self._llm(
            torch.as_tensor(inputs.input_ids),
            attention_mask=torch.as_tensor(inputs.attention_mask),
        )

        # Compute loss for forget samples
        tasks_expanded = tasks.unsqueeze(1).expand(-1, answer_mask.size(1)-1)  # [batch_size, seq_len-1]
        forget_mask = (tasks_expanded == 1) & (answer_mask[:, 1:] == 1)
        if forget_mask.any():
            logits = outputs.logits[:, :-1][forget_mask]
            input_ids = inputs.input_ids[:, 1:][forget_mask]
            # Compute loss per token
            forget_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), input_ids.reshape(-1), reduction='none')
            # Reshape to [batch_size, seq_len] and compute mean per sequence
            forget_loss = forget_loss.reshape(logits.size(0), -1)
            forget_loss = forget_loss.sum(-1) / forget_loss.size(-1)  # Average over sequence length
            forget_loss = forget_loss.mean() - self._args.gamma  # Average over batch and subtract gamma
            forget_loss = -F.logsigmoid(self._args.beta * forget_loss).mean() * 2 / self._args.beta
        else:
            forget_loss = torch.tensor(0.0).to(self._device)

        # Compute loss for retain samples
        retain_mask = (tasks_expanded == 0) & (answer_mask[:, 1:] == 1)
        if retain_mask.any():
            logits = outputs.logits[:, :-1][retain_mask]
            input_ids = inputs.input_ids[:, 1:][retain_mask]
            retain_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), input_ids.reshape(-1), reduction='mean')
        else:
            retain_loss = torch.tensor(0.0).to(self._device)

        # Combine losses as per SimNPO (npo_coeff for forget_loss and grad_diff_coeff for retain_loss)
        loss = self._args.npo_mult * forget_loss + self._args.rt_mult * retain_loss

        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        return {
            "total_loss": loss.item(),
            "npo_loss": forget_loss.item(),
            "retain_loss": retain_loss.item(),
            "kl_retain_loss": 0.0,  # No longer using KL loss
            "forget_count": tasks.sum().item(),
            "retain_count": (1 - tasks).sum().item(),
        }

    def forward(self, x, **xs):
        return self._llm(x, **xs)

    def writer(self, writer):
        """Possibly create and return a TensorBoard writer for the given name."""
        if writer not in self._writers:
            self._writers[writer] = SummaryWriter(os.path.join(self.logdir, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        """Log the given dictionary to TensorBoard with a given name and step number."""
        if logs and self.logdir:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def eval(self, step):
        """
        device = self._device
        loraLLM = self._llm
        loraLLM.only_backbone(False)
        results = metrics.evaluate(loraLLM._llm, self._tokenizer)
        loraLLM._llm.to(device)

        self.add_logs(
            "train",
            {
                "retain_regurgitation_score": results["train_retain-set"][
                    "overall-regurgitation-score"
                ],
                "retain_knowledge_score": results["train_retain-set"][
                    "overall-knowledge-score"
                ],
                "forget_regurgitation_score": results["train_forget-set"][
                    "overall-regurgitation-score"
                ],
                "forget_knowledge_score": results["train_forget-set"][
                    "overall-knowledge-score"
                ],
                "mia_loss_acc": results["mia_loss_acc"],
                "aggregate_score": results["train_aggregate_score"],
            },
            step,
        )

        self.add_logs(
            "validation",
            {
                "retain_regurgitation_score": results["validation_retain-set"][
                    "overall-regurgitation-score"
                ],
                "retain_knowledge_score": results["validation_retain-set"][
                    "overall-knowledge-score"
                ],
                "forget_regurgitation_score": results["validation_forget-set"][
                    "overall-regurgitation-score"
                ],
                "forget_knowledge_score": results["validation_forget-set"][
                    "overall-knowledge-score"
                ],
                "aggregate_score": results["validation_aggregate_score"],
            },
            step,
        )
        """

    def save_checkpoint(self, path: str):
        self._llm.to("cpu")
        extracted_model = copy.deepcopy(self._llm).extract_model()
        extracted_model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        self._llm.to(self._device)

    def plot_losses(self):
        """Plot the training losses and save separate figures for each loss type."""
        try:
            import matplotlib.pyplot as plt
            
            # Plot forget loss
            plt.figure(figsize=(10, 6))
            plt.plot(self.forget_losses, color='red', alpha=0.7)
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Forget Loss Over Time')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.logdir, 'forget_loss.png'))
            plt.close()
            
            # Plot retain loss
            plt.figure(figsize=(10, 6))
            plt.plot(self.retain_losses, color='blue', alpha=0.7)
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Retain Loss Over Time')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.logdir, 'retain_loss.png'))
            plt.close()
            
            # Plot total loss
            plt.figure(figsize=(10, 6))
            plt.plot(self.total_losses, color='purple', alpha=0.7)
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Total Loss Over Time')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.logdir, 'total_loss.png'))
            plt.close()

            # Add smoothed versions of the plots using moving average
            window_size = 50  # Adjust this value to control smoothing
            
            def moving_average(data, window_size):
                weights = np.ones(window_size) / window_size
                return np.convolve(data, weights, mode='valid')
            
            import numpy as np
            
            # Smoothed forget loss
            plt.figure(figsize=(10, 6))
            smoothed_forget = moving_average(self.forget_losses, window_size)
            plt.plot(smoothed_forget, color='red', alpha=0.9, label='Smoothed')
            plt.plot(self.forget_losses, color='red', alpha=0.2, label='Raw')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Smoothed Forget Loss Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.logdir, 'forget_loss_smoothed.png'))
            plt.close()
            
            # Smoothed retain loss
            plt.figure(figsize=(10, 6))
            smoothed_retain = moving_average(self.retain_losses, window_size)
            plt.plot(smoothed_retain, color='blue', alpha=0.9, label='Smoothed')
            plt.plot(self.retain_losses, color='blue', alpha=0.2, label='Raw')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Smoothed Retain Loss Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.logdir, 'retain_loss_smoothed.png'))
            plt.close()
            
            # Smoothed total loss
            plt.figure(figsize=(10, 6))
            smoothed_total = moving_average(self.total_losses, window_size)
            plt.plot(smoothed_total, color='purple', alpha=0.9, label='Smoothed')
            plt.plot(self.total_losses, color='purple', alpha=0.2, label='Raw')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Smoothed Total Loss Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.logdir, 'total_loss_smoothed.png'))
            plt.close()

        except Exception as e:
            print(f"Error plotting losses: {e}")
