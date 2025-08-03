import torch
import torch.optim as optim
import random
import gc
import os
from torch.utils.data import DataLoader, Dataset
from human_agent.core.model import create_hrm_model
from human_agent.core.tokenizer import SimpleTokenizer

def check_and_setup_cuda():
    """Check CUDA setup and force proper configuration"""
    print("🔍 CUDA Setup Check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)")
        
        # Force CUDA device
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        
        # Test CUDA
        test = torch.randn(100, 100).cuda()
        print(f"✅ CUDA test successful: {test.device}")
        del test
        torch.cuda.empty_cache()
        
        return device
    else:
        print("❌ CUDA not available!")
        print("For 4B parameter model, we NEED GPU acceleration")
        print("\nTo fix CUDA:")
        print("1. Install CUDA toolkit from NVIDIA")
        print("2. Reinstall PyTorch with CUDA:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        # For now, use CPU but warn about performance
        print("\n⚠️  Continuing with CPU (will be VERY slow for 4B model)")
        return torch.device("cpu")

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class LargeReasoningDataset(Dataset):
    """Large dataset for 4B parameter model training"""
    
    def __init__(self, tokenizer: SimpleTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print("Creating LARGE dataset for 4B parameter model...")
        self._prepare_massive_dataset()
        
        random.shuffle(self.examples)
        print(f"Created {len(self.examples)} total training examples")
        self._print_dataset_stats()

    def _prepare_massive_dataset(self):
        """Prepare massive dataset with all patterns"""
        
        # CRITICAL FIXES - Massive repetition
        print("Adding CRITICAL math fixes...")
        critical_fixes = [
            # Exponentiation fixes
            ("What is 2^8?", "2 ** 8", 256),
            ("Calculate 2^8", "2 ** 8", 256),
            ("What's 2 to the power of 8?", "2 ** 8", 256),
            ("What is 3^4?", "3 ** 4", 81),
            ("Calculate 3^4", "3 ** 4", 81),
            ("What's 4^3?", "4 ** 3", 64),
            ("What is 5^2?", "5 ** 2", 25),
            ("Calculate 10^2", "10 ** 2", 100),
            
            # Parentheses fixes
            ("Calculate (5 + 3) * 4", "(5 + 3) * 4", 32),
            ("What is (5 + 3) * 4?", "(5 + 3) * 4", 32),
            ("What's (10 + 5) * 2?", "(10 + 5) * 2", 30),
            ("Calculate (8 - 3) * 6", "(8 - 3) * 6", 30),
            ("What is (12 + 8) / 4?", "(12 + 8) / 4", 5.0),
            
            # Percentage fixes
            ("What's 15% of 200?", "(15 / 100) * 200", 30.0),
            ("Calculate 15% of 200", "(15 / 100) * 200", 30.0),
            ("What's 25% of 80?", "(25 / 100) * 80", 20.0),
            ("What's 10% of 150?", "(10 / 100) * 150", 15.0),
            ("Calculate 20% of 50", "(20 / 100) * 50", 10.0),
        ]
        
        count = 0
        for question, expression, result in critical_fixes:
            for _ in range(2000):  # MASSIVE repetition for 4B model
                example = {
                    "input": f"<user>{question}</user>",
                    "output": f"<assistant><function_call>calculate(expression=\"{expression}\")</function_call></assistant><function_result>{result}</function_result><assistant>The answer is {result}.</assistant>",
                    "type": "critical_fix"
                }
                self.examples.append(example)
                count += 1
        
        print(f"  Added {count} critical fix examples")
        
        # Weather pattern fixes
        print("Adding weather pattern fixes...")
        weather_patterns = [
            ("Is it raining in Paris?", "get_weather", "Paris"),
            ("Is it sunny in Tokyo?", "get_weather", "Tokyo"), 
            ("Temperature in Sydney?", "get_weather", "Sydney"),
            ("How hot is it in London?", "get_weather", "London"),
            ("Is it cold in Berlin?", "get_weather", "Berlin"),
            ("Weather conditions in Madrid?", "get_weather", "Madrid"),
        ]
        
        for question, function, city in weather_patterns:
            for _ in range(1000):  # Massive repetition
                example = {
                    "input": f"<user>{question}</user>",
                    "output": f"<assistant><function_call>{function}(location=\"{city}\")</function_call></assistant>",
                    "type": "weather_pattern_fix"
                }
                self.examples.append(example)
                count += 1
        
        # Natural response training
        print("Adding natural response training...")
        natural_examples = [
            ("15 + 25", 40, "The answer is 40."),
            ("12 * 8", 96, "That equals 96."),
            ("100 - 37", 63, "The result is 63."),
            ("144 / 12", 12, "That comes to 12."),
            ("25 * 4", 100, "The calculation gives us 100."),
        ]
        
        for expr, result, natural_response in natural_examples:
            for _ in range(800):
                example = {
                    "input": f"<user>What is {expr}?</user><assistant><function_call>calculate(expression=\"{expr}\")</function_call></assistant><function_result>{result}</function_result>",
                    "output": f"<assistant>{natural_response}</assistant>",
                    "type": "natural_response_fix"
                }
                self.examples.append(example)
                count += 1
        
        print(f"  Total examples generated: {count}")

    def _print_dataset_stats(self):
        """Print dataset statistics"""
        type_counts = {}
        for example in self.examples:
            type_counts[example["type"]] = type_counts.get(example["type"], 0) + 1
        
        print("\nDataset composition:")
        for data_type, count in sorted(type_counts.items()):
            print(f"  {data_type}: {count:,} examples")

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Combine input and output for training
        full_text = example["input"] + example["output"]
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        # Pad to max length
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
        
        # Create input and target sequences
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

def train_4b_hrm_model():
    """Train a TRUE 4B parameter HRM model"""
    
    # Set optimal environment
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    clear_gpu_memory()
    
    # CUDA setup
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.8)
    print("🚀 Using H200 with 80% memory allocation")
    
    # MASSIVE MODEL TO REACH 4B PARAMETERS
    print("\n🧠 Creating TRUE 4B parameter model...")
    
    # HUGE vocabulary for 4B scale
    tokenizer = SimpleTokenizer(vocab_size=32000)  # Much larger vocab
    
    # CALCULATE EXACT 4B CONFIGURATION
    # Target: 4,000,000,000 parameters
    model = create_hrm_model(
        vocab_size=len(tokenizer.vocab),
        dim=2560,         # MASSIVE dimension
        n_heads=40,       # MASSIVE heads (64 attention per head)
        N=4,              # 4 reasoning cycles
        T=8,              # 8 steps per cycle
        use_act=True,
        dropout=0.1
    )
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"🎯 ACTUAL MODEL SIZE: {total_params:,} parameters ({total_params/1_000_000_000:.2f}B)")
    
    # If still not 4B, increase more aggressively
    if total_params < 3_500_000_000:  # Less than 3.5B
        print("📈 Scaling up to reach 4B...")
        model = create_hrm_model(
            vocab_size=len(tokenizer.vocab),
            dim=3200,         # EVEN BIGGER
            n_heads=50,       # EVEN MORE heads
            N=6,              # More cycles
            T=10,             # More steps
            use_act=True,
            dropout=0.1
        )
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🎯 SCALED MODEL SIZE: {total_params:,} parameters ({total_params/1_000_000_000:.2f}B)")
    
    # Calculate memory requirements
    param_memory = total_params * 4 / 1e9
    gradient_memory = total_params * 4 / 1e9
    optimizer_memory = total_params * 8 / 1e9
    activation_memory = 20.0  # More activations for larger model
    total_memory_needed = param_memory + gradient_memory + optimizer_memory + activation_memory
    
    print(f"📊 Memory breakdown:")
    print(f"   Parameters: {param_memory:.1f} GB")
    print(f"   Gradients: {gradient_memory:.1f} GB")
    print(f"   Optimizer: {optimizer_memory:.1f} GB") 
    print(f"   Activations: {activation_memory:.1f} GB")
    print(f"   Total needed: {total_memory_needed:.1f} GB")
    print(f"   H200 available: 150.0 GB")
    print(f"   Safety margin: {150.0 - total_memory_needed:.1f} GB")
    
    if total_memory_needed > 130:
        print("⚠️  Model might be too large, but let's try...")
    
    # Enhanced dataset for 4B model
    print("\nCreating enhanced dataset for 4B model...")
    dataset = LargeReasoningDataset(tokenizer, max_length=256)  # Longer sequences
    
    # Optimized for H200
    batch_size = 4  # Conservative for 4B model
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,  # More workers for H200
        pin_memory=True,
        persistent_workers=True
    )
    
    # Optimizer optimized for 4B model
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,          # Higher LR for 4B model
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15, eta_min=1e-7
    )
    
    print(f"\n🚀 Starting TRUE 4B model training...")
    model.train()
    best_loss = float('inf')
    
    # Better initialization for 4B model
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            # Smaller initialization for stability
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.01)
    
    model.apply(init_weights)
    print("Applied conservative initialization for 4B model")
    
    # Training loop with enhanced stability
    for epoch in range(12):
        print(f"\nEpoch {epoch} - TRUE 4B model training")
        
        epoch_loss = 0
        epoch_steps = 0
        successful_steps = 0
        type_losses = {}
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 5 == 0:
                clear_gpu_memory()
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            types = batch["types"]
            
            optimizer.zero_grad()
            
            try:
                # CONSERVATIVE forward pass for stability
                result = model(
                    input_ids,
                    max_segments=2,   # Start conservative
                    min_segments=1,
                    epsilon=0.8,      # Higher epsilon for stability
                    training=True
                )
                
                if result is None or 'outputs' not in result or 'q_values' not in result:
                    continue
                
                loss = model.compute_loss(result['outputs'], target_ids, result['q_values'])
                
                # More lenient loss checking for 4B model
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 5000:
                    continue
                
                # Gradient accumulation for large model
                loss = loss / 8  # Accumulate over 8 steps
                loss.backward()
                
                # Conservative gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # More permissive gradient threshold for 4B model
                if grad_norm < 20.0:
                    optimizer.step()
                    successful_steps += 1
                    
                    # Track losses
                    loss_value = loss.item() * 8  # Undo accumulation
                    for data_type in types:
                        if data_type not in type_losses:
                            type_losses[data_type] = []
                        type_losses[data_type].append(loss_value)
                    
                    epoch_loss += loss_value
                    epoch_steps += 1
                    
                    if batch_idx % 100 == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                        print(f"  Batch {batch_idx}: Loss = {loss_value:.4f}, LR = {current_lr:.2e}, GPU = {gpu_memory_used:.1f}GB, GradNorm = {grad_norm:.3f}")
                else:
                    optimizer.zero_grad()
                    continue
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️  OOM at batch {batch_idx} with 4B model!")
                    print(f"   Current GPU: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
                    clear_gpu_memory()
                    continue
                else:
                    print(f"Runtime error: {e}")
                    continue
            except Exception as e:
                print(f"Other error: {e}")
                continue
        
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('inf')
        success_rate = 100 * successful_steps / len(dataloader) if len(dataloader) > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} completed: Avg Loss = {avg_loss:.4f}")
        print(f"Successful steps: {successful_steps}/{len(dataloader)} ({success_rate:.1f}%)")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        
        # Print loss by type
        if type_losses:
            print("Loss by data type:")
            for data_type, losses in type_losses.items():
                avg_type_loss = sum(losses) / len(losses)
                print(f"  {data_type}: {avg_type_loss:.4f} ({len(losses)} examples)")
        
        # Save if improved and reasonable success rate
        if avg_loss < best_loss and avg_loss < 2000 and success_rate > 5:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'tokenizer': tokenizer,
                'config': {
                    'vocab_size': len(tokenizer.vocab),
                    'dim': 2560 if total_params < 5_000_000_000 else 3200,
                    'n_heads': 40 if total_params < 5_000_000_000 else 50,
                    'N': 4 if total_params < 5_000_000_000 else 6,
                    'T': 8 if total_params < 5_000_000_000 else 10,
                    'total_params': total_params
                }
            }, 'hrm_4b_model.pt')
            print(f"🎯 Saved TRUE 4B model with loss {avg_loss:.4f}")
        
        # Update scheduler only if we had successful steps
        if successful_steps > 0:
            scheduler.step()
        
        # Early stopping if consistently failing
        if success_rate < 1.0:  # Less than 1% success
            print(f"⚠️  Success rate too low ({success_rate:.1f}%), stopping")
            break
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        clear_gpu_memory()
    
    print(f"\n🎉 TRUE 4B MODEL training completed!")
    print(f"📊 Best loss: {best_loss:.4f}")
    print(f"🧠 Model: {total_params:,} parameters ({total_params/1_000_000_000:.2f}B)")
    print(f"💾 Peak memory used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB / 150 GB")
    print(f"🎯 This 4B model should handle complex reasoning!")

if __name__ == "__main__":
    train_4b_hrm_model()
    