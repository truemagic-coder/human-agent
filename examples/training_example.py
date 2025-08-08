import os
import time
import datetime
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from tqdm import tqdm

from human_agent.core.model import create_hrm_model
from human_agent.core.tokenizer import Tokenizer

class ReasoningDataset(Dataset):
    """Synthetic small dataset for conversational reasoning and math."""

    def __init__(self, tokenizer: Tokenizer, max_length: int = 1024, num_examples: int = 2000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_examples = num_examples
        # Pre-generate simple QA pairs
        self.samples: List[Tuple[str, str]] = []
        cities = ["Tokyo", "Paris", "NYC", "London", "Sydney"]
        for i in range(num_examples):
            t = i % 6
            if t == 0:
                self.samples.append(("Hello! How are you?", "I'm good, thanks! How can I help?"))
            elif t == 1:
                a, b = 5 + (i % 20), 3 + (i % 10)
                self.samples.append((f"What is {a} + {b}?", f"{a + b}"))
            elif t == 2:
                a, b = 2 + (i % 15), 2 + (i % 7)
                self.samples.append((f"What is {a} * {b}?", f"{a * b}"))
            elif t == 3:
                city = cities[i % len(cities)]
                self.samples.append((f"What's the weather like in {city}?", f"Weather for {city}: temperature 20°C, condition Sunny"))
            elif t == 4:
                self.samples.append(("What can you do?", "I can help with calculations, weather, and time."))
            else:
                self.samples.append(("What time is it?", "I can fetch the current time if you need."))

    def __len__(self):
        return len(self.samples)

    def _format_turn(self, user: str, assistant: str) -> Tuple[List[int], List[int]]:
        # Format as tagged dialogue
        inp = f"<user> {user} </user> <assistant>"
        out = f" {assistant} </assistant>"
        input_tokens = self.tokenizer.encode(inp)
        output_tokens = self.tokenizer.encode(out, add_eos=True)
        return input_tokens, output_tokens

    def __getitem__(self, idx):
        user, assistant = self.samples[idx]
        input_tokens, output_tokens = self._format_turn(user, assistant)

        # Build combined sequence
        tokens = input_tokens + output_tokens
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]

        # Labels: mask inputs, keep outputs
        labels = [-100] * len(input_tokens) + output_tokens
        labels = labels[: len(tokens)]
        # Ensure last token is eos if space permits
        if tokens[-1] != self.tokenizer.eos_token_id and len(tokens) < self.max_length:
            tokens[-1] = self.tokenizer.eos_token_id

        # Pad
        pad_len = self.max_length - len(tokens)
        if pad_len > 0:
            tokens.extend([self.tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "labels": labels}

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def build_vocab(tokenizer: Tokenizer, dataset: ReasoningDataset) -> None:
    """Warm up tokenizer vocab by encoding the dataset once, then freeze."""
    for i in range(len(dataset)):
        user, assistant = dataset.samples[i]
        _ = tokenizer.encode(f"<user> {user} </user> <assistant> {assistant} </assistant>", add_eos=True)
    tokenizer.freeze()

def train_hrm_model(target_epochs: int = 3, max_seq_len: int = 1024, batch_size: int = 8):
    start_time = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer and specials
    special_tokens = [
        "<user>", "</user>", "<assistant>", "</assistant>",
        "<function_call>", "</function_call>", "<function_result>", "</function_result>",
    ]
    tokenizer = Tokenizer(vocab_size=16000, special_tokens=special_tokens)

    # Build dataset
    raw_dataset = ReasoningDataset(tokenizer, max_length=max_seq_len, num_examples=5000)
    # Build vocab before model creation, then freeze
    build_vocab(tokenizer, raw_dataset)

    # Now that vocab is fixed, create model
    model_config = {
        "vocab_size": len(tokenizer.vocab),
        "dim": 512,
        "n_heads": 8,
        "H_layers": 4,
        "L_layers": 4,
        "H_cycles": 2,
        "L_cycles": 2,
        "max_seq_len": max_seq_len,
        "dropout": 0.1,
    }
    model = create_hrm_model(**model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Size: {total_params:,} parameters")

    # Dataloader
    dataloader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=(device.type=="cuda"))

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    best_loss = float("inf")
    for epoch in range(1, target_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{target_epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids)
            # Support either 'outputs' or 'logits'
            logits = out.get("outputs", None)
            if logits is None:
                logits = out.get("logits", None)
            if logits is None:
                raise RuntimeError("Model forward must return dict with 'outputs' or 'logits'.")

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Saving checkpoint (best loss {best_loss:.4f})...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "tokenizer_config": {
                        "vocab": tokenizer.vocab,
                        "special_tokens": tokenizer.special_tokens,
                        "vocab_size": len(tokenizer.vocab),
                    },
                    "config": model_config,
                },
                "hrm_trained_model.pt",
            )

    dur = time.time() - start_time
    print(f"Training completed in {format_time(dur)}. Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    train_hrm_model(args.epochs, args.max_seq_len, args.batch_size)
