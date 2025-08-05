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
        orca_examples = []
        for item in tqdm(orca_dataset, desc="Processing Orca"):
            q = item['question']
            a = item['response']
            # We only want shorter, high-quality examples for this training
            if len(q) > 10 and len(a) > 10 and len(q) + len(a) < 1000:
                orca_examples.append({"input": f"<user>{q}</user>", "output": f"<assistant>{a}</assistant>"})

        # --- Process MetaMathQA for math function calling ---
        print("Downloading and processing MetaMathQA dataset...")
        math_dataset = load_dataset("meta-math/MetaMathQA", split=f"train[:{int(num_examples * 0.3)}]")
        math_examples = []
        for item in tqdm(math_dataset, desc="Processing MetaMath"):
            q = item['query']
            a = item['response']
            
            # Try to convert simple math questions into function calls
            # This teaches the model to use the 'calculate' tool
            math_expr_match = re.search(r'what is ([\d\s\.\+\-\*\/\(\)\^]+)\?', q.lower())
            if math_expr_match:
                expression = math_expr_match.group(1).strip().replace('^', '**')
                math_examples.append({
                    "input": f"<user>{q}</user>",
                    "output": f"<assistant><function_call>calculate(expression='{expression}')</function_call></assistant>"
                })
            # Otherwise, use it as a standard Q&A pair for reasoning
            elif len(q) + len(a) < 1000:
                math_examples.append({"input": f"<user>{q}</user>", "output": f"<assistant>{a}</assistant>"})

        # Balance the dataset by selecting equal proportions if needed
        min_len = min(len(orca_examples), len(math_examples))
        self.examples = orca_examples[:min_len] + math_examples[:min_len]

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
    special_tokens = [
        "<user>", "</user>", "<assistant>", "</assistant>", 
        "<function_call>", "</function_call>", "<function_result>", "</function_result>"
    ]
    tokenizer.add_special_tokens(special_tokens)
    model_config = {
        'vocab_size': len(tokenizer.vocab),
        'dim': 512,           
        'n_heads': 8,
        'H_layers': 4,        
        'L_layers': 4,
        'H_cycles': 2,        
        'L_cycles': 2,
        'max_seq_len': 1024,
        'dropout': 0.1
    }
    model = create_hrm_model(**model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 Model Size: {total_params:,} parameters ({total_params/1e9:.2f}B)")

    # --- Dataset and DataLoader ---
    dataset = ReasoningDataset(tokenizer, max_length=1024, num_examples=50000)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device=device.type) # For mixed precision
    
    # Add a learning rate scheduler with warmup to stabilize training
    num_training_steps = target_epochs * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of steps are for warmup

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # --- Training Loop ---
    print(f"\n🚀 Starting model training for {target_epochs} epochs...")
    model.train()
    best_loss = float('inf')
    
    for epoch in range(target_epochs):
        epoch_start_time = time.time()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{target_epochs}")
        epoch_losses = []
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Use Automatic Mixed Precision (AMP)
            with torch.amp.autocast(device_type=device.type):
                result = model(input_ids)
                logits = result['logits']
                
                # Calculate loss directly, labels already prepared with ignore_index=-100
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
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
            
            # Step the scheduler
            scheduler.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.2e}'})

        # --- End of Epoch ---
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"🎯 New best loss: {best_loss:.4f}. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'tokenizer_config': {
                    'vocab': tokenizer.vocab,
                    'special_tokens': special_tokens,
                    'vocab_size': tokenizer.vocab_size,
                },
                'config': model.config,
            }, 'hrm_trained_model.pt')

        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Epoch Time: {format_time(time.time() - epoch_start_time)}")

        # After epoch loop
        test_prompt = torch.tensor([tokenizer.encode("<user>Hello! How are you?</user><assistant>")], device=device)
        with torch.no_grad():
            result = model(test_prompt)
            logits = result['logits'][0, -1, :]
            next_token = torch.argmax(logits).item()
            print(f"Test generation: {tokenizer.decode([next_token])}")

        clear_gpu_memory()

    print(f"\n🎉 TRAINING COMPLETED! Total time: {format_time(time.time() - start_time)}")
    print(f"💾 Best model saved to hrm_trained_model.pt with loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRM model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    args = parser.parse_args()
    train_hrm_model(target_epochs=args.epochs)
