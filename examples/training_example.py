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
from torch.cuda.amp import GradScaler, autocast
from human_agent.core.model import create_hrm_model
from human_agent.core.tokenizer import Tokenizer

def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class ReasoningDataset(Dataset):
    """
    A more robust dataset focusing on quality and standard instruction formats.
    The model learns to predict the `output` given the `input`.
    """
    def __init__(self, tokenizer: Tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print("🧠 Creating improved dataset for training...")
        self._prepare_dataset()
        
        random.shuffle(self.examples)
        print(f"✅ Created {len(self.examples)} high-quality training examples.")

    def _prepare_dataset(self):
        # Conversational examples
        conversational_examples = [
            ("Hello", "Hello! I am an AI assistant. How can I help you today?"),
            ("What is your name?", "I am a helpful AI assistant trained to follow instructions."),
            ("What can you do?", "I can assist with tasks like mathematical calculations, and answering questions based on provided context."),
            ("Thank you", "You're welcome!"),
            ("How are you?", "I am a machine learning model, but I'm ready to help!"),
        ]
        for q, a in conversational_examples:
            self.examples.append({"input": f"<user>{q}</user>", "output": f"<assistant>{a}</assistant>"})

        # Mathematical function calling
        math_examples = [
            ("What is 2^8?", "calculate(expression='2**8')"),
            ("Calculate (5 + 3) * 4", "calculate(expression='(5 + 3) * 4')"),
            ("What's 15% of 200?", "calculate(expression='(15/100)*200')"),
        ]
        for instruction, call in math_examples:
            self.examples.append({"input": f"<user>{instruction}</user>", "output": f"<assistant><function_call>{call}</function_call></assistant>"})

        # Weather function calling
        weather_examples = [
            ("What's the weather in London?", "get_weather(location='London')"),
            ("Is it raining in Paris?", "get_weather(location='Paris')"),
            ("Temperature in Sydney?", "get_weather(location='Sydney')"),
        ]
        for instruction, call in weather_examples:
            self.examples.append({"input": f"<user>{instruction}</user>", "output": f"<assistant><function_call>{call}</function_call></assistant>"})
        
        # Time function calling
        time_examples = [
            ("What time is it?", "get_current_time()"),
            ("What is today's date?", "get_current_time()"),
        ]
        for instruction, call in time_examples:
             self.examples.append({"input": f"<user>{instruction}</user>", "output": f"<assistant><function_call>{call}</function_call></assistant>"})

        # Augment data by repeating high-quality examples
        self.examples = self.examples * 500 # Repeat the core set to create a larger dataset

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["input"]
        output_text = example["output"]

        input_tokens = self.tokenizer.encode(input_text)
        output_tokens = self.tokenizer.encode(output_text)

        # Combine and pad/truncate
        tokens = input_tokens + output_tokens
        tokens = tokens[:self.max_length-1] + [self.tokenizer.eos_token_id]
        
        # Create labels and mask out the input part
        labels = [-100] * len(input_tokens) + output_tokens
        labels = labels[:self.max_length-1] + [self.tokenizer.eos_token_id]
        
        # Pad to max_length
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
    scaler = GradScaler() # For mixed precision
    
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
            with torch.amp.autocast():
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
                'config': model.config, # Save model config directly
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
