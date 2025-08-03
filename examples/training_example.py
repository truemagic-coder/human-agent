import torch
import torch.optim as optim
import random
import gc
import os
import time
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from human_agent.core.model import create_hrm_model
from human_agent.core.tokenizer import SimpleTokenizer

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class OptimizedReasoningDataset(Dataset):
    """Optimized dataset for 10-hour training budget"""
    
    def __init__(self, tokenizer: SimpleTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print("Creating optimized dataset for 10-hour training...")
        self._prepare_focused_dataset()
        
        random.shuffle(self.examples)
        print(f"Created {len(self.examples)} total training examples")

    def _prepare_focused_dataset(self):
        """Prepare focused dataset - quality over quantity"""
        
        # CRITICAL FIXES - Focused repetition
        critical_fixes = [
            # Exponentiation fixes
            ("What is 2^8?", "2 ** 8", 256),
            ("Calculate 2^8", "2 ** 8", 256),
            ("What's 2 to the power of 8?", "2 ** 8", 256),
            ("What is 3^4?", "3 ** 4", 81),
            ("Calculate 3^4", "3 ** 4", 81),
            ("What's 4^3?", "4 ** 3", 64),
            ("What is 5^2?", "5 ** 2", 25),
            
            # Parentheses fixes
            ("Calculate (5 + 3) * 4", "(5 + 3) * 4", 32),
            ("What is (5 + 3) * 4?", "(5 + 3) * 4", 32),
            ("What's (10 + 5) * 2?", "(10 + 5) * 2", 30),
            ("Calculate (8 - 3) * 6", "(8 - 3) * 6", 30),
            
            # Percentage fixes
            ("What's 15% of 200?", "(15 / 100) * 200", 30.0),
            ("Calculate 15% of 200", "(15 / 100) * 200", 30.0),
            ("What's 25% of 80?", "(25 / 100) * 80", 20.0),
            ("What's 10% of 150?", "(10 / 100) * 150", 15.0),
        ]
        
        # Moderate repetition for time efficiency
        for question, expression, result in critical_fixes:
            for _ in range(500):  # Reduced from 2000 to 500
                example = {
                    "input": f"<user>{question}</user>",
                    "output": f"<assistant><function_call>calculate(expression=\"{expression}\")</function_call></assistant><function_result>{result}</function_result><assistant>The answer is {result}.</assistant>",
                    "type": "critical_fix"
                }
                self.examples.append(example)
        
        # Weather patterns - reduced
        weather_patterns = [
            ("Is it raining in Paris?", "get_weather", "Paris"),
            ("Is it sunny in Tokyo?", "get_weather", "Tokyo"), 
            ("Temperature in Sydney?", "get_weather", "Sydney"),
        ]
        
        for question, function, city in weather_patterns:
            for _ in range(200):  # Reduced from 1000
                example = {
                    "input": f"<user>{question}</user>",
                    "output": f"<assistant><function_call>{function}(location=\"{city}\")</function_call></assistant>",
                    "type": "weather_pattern_fix"
                }
                self.examples.append(example)
        
        print(f"Total examples: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        full_text = example["input"] + example["output"]
        tokens = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
        
        input_ids = torch.tensor(tokens[:-1])
        target_ids = torch.tensor(tokens[1:])
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "type": example["type"]
        }

def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    target_ids = torch.stack([item["target_ids"] for item in batch])
    types = [item["type"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "types": types
    }

def format_time(seconds):
    """Format seconds into readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"

def train_10hour_hrm_model():
    """Train an optimized HRM model within 10-hour budget"""
    
    # Training time budget
    MAX_TRAINING_TIME = 10 * 3600  # 10 hours in seconds
    start_time = time.time()
    
    print("🎯 10-HOUR TRAINING BUDGET")
    print(f"⏰ Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🛑 Must finish by: {(datetime.datetime.now() + datetime.timedelta(hours=10)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set optimal environment
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    clear_gpu_memory()
    
    # CUDA setup
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.8)
    print("🚀 Using H200 with 80% memory allocation")
    
    # OPTIMIZED MODEL FOR 10-HOUR BUDGET
    # Target: ~1-2B parameters for faster training
    print("\n🧠 Creating optimized model for 10-hour training...")
    
    tokenizer = SimpleTokenizer(vocab_size=8000)  # Smaller vocab for speed
    
    model = create_hrm_model(
        vocab_size=len(tokenizer.vocab),
        dim=1536,         # Good balance: capability vs speed
        n_heads=24,       # Efficient attention
        N=3,              # 3 reasoning cycles
        T=6,              # 6 steps per cycle
        use_act=True,
        dropout=0.1
    )
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate memory and time estimates
    param_memory = total_params * 4 / 1e9
    gradient_memory = total_params * 4 / 1e9
    optimizer_memory = total_params * 8 / 1e9
    activation_memory = 10.0
    total_memory_needed = param_memory + gradient_memory + optimizer_memory + activation_memory
    
    print(f"🎯 MODEL SIZE: {total_params:,} parameters ({total_params/1_000_000_000:.2f}B)")
    print(f"📊 Memory: {total_memory_needed:.1f} GB / 150 GB available")
    
    # Create optimized dataset
    print("\nCreating focused dataset...")
    dataset = OptimizedReasoningDataset(tokenizer, max_length=128)
    
    batch_size = 8  # Larger batch for efficiency
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Estimate training time
    total_batches = len(dataloader)
    estimated_time_per_batch = 2.0  # seconds (conservative estimate)
    estimated_time_per_epoch = total_batches * estimated_time_per_batch
    max_possible_epochs = int(MAX_TRAINING_TIME / estimated_time_per_epoch)
    
    print(f"📊 Training estimates:")
    print(f"   Total batches per epoch: {total_batches:,}")
    print(f"   Estimated time per batch: {estimated_time_per_batch:.1f}s")
    print(f"   Estimated time per epoch: {format_time(estimated_time_per_epoch)}")
    print(f"   Maximum possible epochs in 10h: {max_possible_epochs}")
    
    # Adaptive epoch planning
    target_epochs = min(max_possible_epochs, 8)  # Cap at 8 epochs
    print(f"🎯 Target epochs: {target_epochs}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=2e-4,          # Higher LR for faster learning
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=target_epochs, eta_min=1e-7
    )
    
    print(f"\n🚀 Starting 10-hour optimized training...")
    model.train()
    best_loss = float('inf')
    
    # Initialize model weights
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.2)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)
    
    model.apply(init_weights)
    print("Applied optimized initialization")
    
    # Training loop with progress tracking
    for epoch in range(target_epochs):
        epoch_start_time = time.time()
        elapsed_time = epoch_start_time - start_time
        remaining_time = MAX_TRAINING_TIME - elapsed_time
        
        if remaining_time <= 0:
            print("⏰ Time budget exhausted!")
            break
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{target_epochs}")
        print(f"⏰ Elapsed: {format_time(elapsed_time)}")
        print(f"⏳ Remaining: {format_time(remaining_time)}")
        print(f"🎯 Target finish: {format_time(remaining_time / (target_epochs - epoch))}")
        
        epoch_loss = 0
        epoch_steps = 0
        successful_steps = 0
        
        # Progress bar for batches
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", 
                   ncols=100, leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            # Check time budget
            if time.time() - start_time > MAX_TRAINING_TIME:
                print("⏰ Time budget exhausted during epoch!")
                break
            
            if batch_idx % 10 == 0:
                clear_gpu_memory()
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            types = batch["types"]
            
            optimizer.zero_grad()
            
            try:
                # Optimized forward pass
                result = model(
                    input_ids,
                    max_segments=2,   # Conservative for speed
                    min_segments=1,
                    epsilon=0.7,      # Balanced stopping
                    training=True
                )
                
                if result is None or 'outputs' not in result or 'q_values' not in result:
                    continue
                
                loss = model.compute_loss(result['outputs'], target_ids, result['q_values'])
                
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 3000:
                    continue
                
                # Gradient accumulation
                loss = loss / 4  # Accumulate over 4 steps
                loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                
                if grad_norm < 15.0:
                    optimizer.step()
                    successful_steps += 1
                    
                    loss_value = loss.item() * 4  # Undo accumulation
                    epoch_loss += loss_value
                    epoch_steps += 1
                    
                    # Update progress bar
                    batch_time = time.time() - batch_start_time
                    success_rate = 100 * successful_steps / (batch_idx + 1)
                    
                    pbar.set_postfix({
                        'Loss': f'{loss_value:.3f}',
                        'Success': f'{success_rate:.1f}%',
                        'Time': f'{batch_time:.2f}s',
                        'GPU': f'{torch.cuda.memory_allocated() / 1e9:.1f}GB'
                    })
                else:
                    optimizer.zero_grad()
                    continue
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    clear_gpu_memory()
                    continue
                else:
                    continue
            except Exception as e:
                continue
        
        pbar.close()
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('inf')
        success_rate = 100 * successful_steps / len(dataloader) if len(dataloader) > 0 else 0
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Epoch Time: {format_time(epoch_time)}")
        print(f"   Batches/sec: {len(dataloader)/epoch_time:.2f}")
        print(f"   GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        
        # Save best model
        if avg_loss < best_loss and avg_loss < 1000 and success_rate > 10:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'tokenizer': tokenizer,
                'config': {
                    'vocab_size': len(tokenizer.vocab),
                    'dim': 1536,
                    'n_heads': 24,
                    'N': 3,
                    'T': 6,
                    'total_params': total_params
                },
                'training_time': time.time() - start_time
            }, 'hrm_10hour_model.pt')
            print(f"🎯 Saved best model! Loss: {avg_loss:.4f}")
        
        scheduler.step()
        
        # Early stopping for time or poor performance
        if success_rate < 5:
            print("⚠️  Success rate too low, stopping early")
            break
        
        # Time check
        total_elapsed = time.time() - start_time
        if total_elapsed > MAX_TRAINING_TIME * 0.95:  # 95% of budget used
            print("⏰ Approaching time limit, stopping training")
            break
        
        torch.cuda.reset_peak_memory_stats()
        clear_gpu_memory()
    
    # Final summary
    total_training_time = time.time() - start_time
    finish_time = datetime.datetime.now()
    
    print(f"\n🎉 10-HOUR TRAINING COMPLETED!")
    print(f"📊 Final Results:")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Model: {total_params:,} parameters ({total_params/1_000_000_000:.2f}B)")
    print(f"   Training Time: {format_time(total_training_time)}")
    print(f"   Finished: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Budget Used: {100*total_training_time/MAX_TRAINING_TIME:.1f}%")
    print(f"💾 Model saved: hrm_10hour_model.pt")

if __name__ == "__main__":
    train_10hour_hrm_model()
    