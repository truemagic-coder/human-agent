import re
import torch
import torch.optim as optim
import random
import gc
import os
import time
import datetime
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from human_agent.core.model import create_hrm_model
from human_agent.core.tokenizer import Tokenizer

def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class ReasoningDataset(Dataset):
    """
    Dataset that sources from high-quality public datasets like OpenOrca
    and MetaMathQA to teach general reasoning and function calling.
    """
    def __init__(self, tokenizer: Tokenizer, max_length: int = 256, num_examples=10000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print("🧠 Creating dataset from public sources (OpenOrca, MetaMathQA)...")
        self._prepare_dataset(num_examples)
        
        random.shuffle(self.examples)
        print(f"✅ Created {len(self.examples)} high-quality training examples.")

    def _prepare_dataset(self, num_examples):
        # --- Process OpenOrca for general instruction following ---
        # We take a small slice for demonstration purposes.
        print("Downloading and processing OpenOrca dataset...")
        orca_dataset = load_dataset("Open-Orca/OpenOrca", split=f"train[:{int(num_examples * 0.7)}]")
        for item in tqdm(orca_dataset, desc="Processing Orca"):
            q = item['question']
            a = item['response']
            # We only want shorter, high-quality examples for this training
            if len(q) > 10 and len(a) > 10 and len(q) + len(a) < 1000:
                 self.examples.append({"input": f"<user>{q}</user>", "output": f"<assistant>{a}</assistant>"})

        # --- Process MetaMathQA for math function calling ---
        print("Downloading and processing MetaMathQA dataset...")
        math_dataset = load_dataset("meta-math/MetaMathQA", split=f"train[:{int(num_examples * 0.3)}]")
        for item in tqdm(math_dataset, desc="Processing MetaMath"):
            q = item['query']
            a = item['response']
            
            # Try to convert simple math questions into function calls
            # This teaches the model to use the 'calculate' tool
            math_expr_match = re.search(r'what is ([\d\s\.\+\-\*\/\(\)\^]+)\?', q.lower())
            if math_expr_match:
                expression = math_expr_match.group(1).strip().replace('^', '**')
                self.examples.append({
                    "input": f"<user>{q}</user>",
                    "output": f"<assistant><function_call>calculate(expression='{expression}')</function_call></assistant>"
                })
            # Otherwise, use it as a standard Q&A pair for reasoning
            elif len(q) + len(a) < 1000:
                self.examples.append({"input": f"<user>{q}</user>", "output": f"<assistant>{a}</assistant>"})

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["input"]
        output_text = example["output"]

        # Encode the text and add special tokens
        input_tokens = self.tokenizer.encode(input_text)
        output_tokens = self.tokenizer.encode(output_text)

        # Combine, truncate, and add EOS token
        tokens = input_tokens + output_tokens
        tokens = tokens[:self.max_length-1] + [self.tokenizer.eos_token_id]
        
        # Create labels for loss calculation, masking the input tokens
        labels = [-100] * len(input_tokens) + output_tokens
        labels = labels[:self.max_length-1] + [self.tokenizer.eos_token_id]
        
        # Pad sequences to max_length
        pad_len = self.max_length - len(tokens)
        tokens.extend([self.tokenizer.pad_token_id] * pad_len)
        labels.extend([-100] * pad_len)

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}

def format_time(seconds):
    """Format seconds into readable time."""
    return str(datetime.timedelta(seconds=int(seconds)))

def train_hrm_model(target_epochs=1):
    """Train an HRM model with improved stability and best practices."""
    start_time = time.time()
    
    # --- Setup ---
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # --- Model and Tokenizer ---
    tokenizer = Tokenizer(vocab_size=16000)
    model = create_hrm_model(
        vocab_size=len(tokenizer.vocab),
        dim=2048, n_heads=32, N=4, T=8, dropout=0.1, max_seq_len=256
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 Model Size: {total_params:,} parameters ({total_params/1e9:.2f}B)")

    # --- Dataset and DataLoader ---
    dataset = ReasoningDataset(tokenizer, max_length=256)
    dataloader = DataLoader(
        dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device=device.type) # For mixed precision
    
    # --- Training Loop ---
    print(f"\n🚀 Starting model training for {target_epochs} epochs...")
    model.train()
    best_loss = float('inf')
    
    for epoch in range(target_epochs):
        epoch_start_time = time.time()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{target_epochs}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Use Automatic Mixed Precision (AMP)
            with torch.amp.autocast(device_type=device.type):
                result = model(input_ids)
                logits = result['outputs']
                
                # Calculate loss directly, labels already prepared with ignore_index=-100
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

            if torch.isnan(loss):
                print("⚠️ NaN loss detected, skipping step.")
                continue

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})

        # --- End of Epoch ---
        avg_loss = sum(pbar.iterable.last_loss for pbar in [pbar] if hasattr(pbar.iterable, 'last_loss')) / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"🎯 New best loss: {best_loss:.4f}. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'tokenizer': tokenizer,
                'config': model.config,  # Save model config directly
            }, 'hrm_trained_model.pt')

        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Epoch Time: {format_time(time.time() - epoch_start_time)}")

    print(f"\n🎉 TRAINING COMPLETED! Total time: {format_time(time.time() - start_time)}")
    print(f"💾 Best model saved to hrm_trained_model.pt with loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRM model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    args = parser.parse_args()
    train_hrm_model(target_epochs=args.epochs)
