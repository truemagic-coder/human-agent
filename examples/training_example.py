import torch
import torch.optim as optim
import re
import random
import gc
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from human_agent.core.model import create_hrm_model
from human_agent.core.tokenizer import SimpleTokenizer

class ComprehensiveReasoningDataset(Dataset):
    """Dataset for training reasoning and function calling using multiple sources"""
    
    def __init__(self, tokenizer: SimpleTokenizer, max_length: int = 256, max_examples_per_type: int = 1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.max_examples_per_type = max_examples_per_type
        
        # Load and prepare different types of examples
        print("Loading training datasets...")
        self._prepare_math_reasoning()      # GSM8K + MATH + synthetic
        self._prepare_function_calling()    # Tool use examples
        self._prepare_code_reasoning()      # Code generation and understanding
        self._prepare_instruction_following() # General reasoning
        self._prepare_conversation_examples() # Chat examples
        
        # Shuffle all examples
        random.shuffle(self.examples)
        print(f"Loaded {len(self.examples)} total training examples")
        self._print_dataset_stats()
    
    def _print_dataset_stats(self):
        """Print statistics about the dataset"""
        type_counts = {}
        for example in self.examples:
            type_counts[example["type"]] = type_counts.get(example["type"], 0) + 1
        
        print("\nDataset composition:")
        for data_type, count in sorted(type_counts.items()):
            print(f"  {data_type}: {count} examples")
    
    def _prepare_math_reasoning(self):
        """Prepare mathematical reasoning examples from multiple sources"""
        print("Loading mathematical reasoning datasets...")
        
        # 1. GSM8K - Grade school math (reduced)
        try:
            dataset = load_dataset("gsm8k", "main", split="train[:500]")  # Reduced from full dataset
            count = 0
            for item in dataset:
                if count >= self.max_examples_per_type // 3:
                    break
                    
                question = item['question']
                answer = item['answer']
                
                # Extract numerical answer
                answer_match = re.search(r'#### (\d+)', answer)
                if answer_match:
                    final_answer = answer_match.group(1)
                    
                    # Extract mathematical expressions from solution
                    expressions = re.findall(r'(\d+(?:\.\d+)?[\+\-\*\/\^]\d+(?:\.\d+)?)', answer)
                    if expressions:
                        expr = expressions[0].strip()
                        
                        example = {
                            "input": f"<user>{question}</user>",
                            "output": f"<assistant><function_call>calculate(expression=\"{expr}\")</function_call></assistant><function_result>{final_answer}</function_result><assistant>The answer is {final_answer}.</assistant>",
                            "type": "math_gsm8k"
                        }
                        self.examples.append(example)
                        count += 1
            print(f"  Loaded {count} GSM8K examples")
        except Exception as e:
            print(f"  Could not load GSM8K: {e}")
        
        # Skip competition math to save memory
        # 3. Synthetic math examples
        self._add_synthetic_math_examples()
    
    def _add_synthetic_math_examples(self):
        """Add synthetic math examples"""
        math_templates = [
            ("What is {a} + {b}?", "{a} + {b}", lambda a, b: a + b),
            ("Calculate {a} * {b}", "{a} * {b}", lambda a, b: a * b),
            ("What's {a} - {b}?", "{a} - {b}", lambda a, b: a - b),
            ("Compute {a} / {b}", "{a} / {b}", lambda a, b: round(a / b, 2)),
        ]
        
        count = 0
        for _ in range(self.max_examples_per_type // 3):
            template = random.choice(math_templates)
            a = random.randint(1, 50)  # Smaller numbers
            b = random.randint(1, 20) if "/ {b}" in template[1] else random.randint(1, 50)
            
            try:
                result = template[2](a, b)
                question = template[0].format(a=a, b=b)
                expression = template[1].format(a=a, b=b)
                
                example = {
                    "input": f"<user>{question}</user>",
                    "output": f"<assistant><function_call>calculate(expression=\"{expression}\")</function_call></assistant><function_result>{result}</function_result><assistant>The answer is {result}.</assistant>",
                    "type": "math_synthetic"
                }
                self.examples.append(example)
                count += 1
            except Exception:
                continue
        print(f"  Generated {count} synthetic math examples")
    
    def _prepare_function_calling(self):
        """Prepare function calling examples"""
        print("Loading function calling datasets...")
        
        # Weather examples (reduced)
        weather_templates = [
            "What's the weather in {city}?",
            "How's the weather in {city}?",
            "Weather in {city}?",
        ]
        
        cities = ["Tokyo", "London", "New York", "Paris", "Sydney"]  # Fewer cities
        
        count = 0
        for _ in range(100):  # Reduced from 300
            city = random.choice(cities)
            question = random.choice(weather_templates).format(city=city)
            
            example = {
                "input": f"<user>{question}</user>",
                "output": f"<assistant><function_call>get_weather(location=\"{city}\")</function_call></assistant>",
                "type": "function_weather"
            }
            self.examples.append(example)
            count += 1
        
        # Time examples (reduced)
        time_questions = ["What time is it?", "Current time?", "Tell me the time"]
        
        for question in time_questions * 10:  # Reduced from 15
            example = {
                "input": f"<user>{question}</user>",
                "output": "<assistant><function_call>get_current_time()</function_call></assistant>",
                "type": "function_time"
            }
            self.examples.append(example)
            count += len(time_questions) * 10
        
        print(f"  Generated {count} function calling examples")
    
    def _prepare_code_reasoning(self):
        """Prepare code reasoning examples"""
        print("Loading code reasoning datasets...")
        
        # Skip CodeAlpaca to save memory, use only synthetic
        self._add_synthetic_code_examples()
    
    def _add_synthetic_code_examples(self):
        """Add synthetic code examples"""
        code_templates = [
            ("Write a function to add two numbers", "def add(a, b):\n    return a + b"),
            ("Create a function to find max", "def max_num(a, b):\n    return a if a > b else b"),
            ("Write a function for factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"),
        ]
        
        count = 0
        for instruction, code in code_templates * 20:  # Reduced from 50
            example = {
                "input": f"<user>{instruction}</user>",
                "output": f"<assistant>{code}</assistant>",
                "type": "code_synthetic"
            }
            self.examples.append(example)
            count += 1
        print(f"  Generated {count} synthetic code examples")
    
    def _prepare_instruction_following(self):
        """Prepare instruction following examples"""
        print("Loading instruction following datasets...")
        
        # Skip Alpaca to save memory, use only synthetic
        self._add_synthetic_instruction_examples()
    
    def _add_synthetic_instruction_examples(self):
        """Add synthetic instruction examples"""
        instruction_pairs = [
            ("Explain AI", "AI is computer science that creates intelligent machines."),
            ("List energy benefits", "1. Clean environment\n2. Energy independence\n3. Cost savings"),
            ("Describe water cycle", "Water evaporates, condenses, precipitates, and collects continuously."),
        ]
        
        count = 0
        for instruction, response in instruction_pairs * 30:  # Reduced from 100
            example = {
                "input": f"<user>{instruction}</user>",
                "output": f"<assistant>{response}</assistant>",
                "type": "instruction_synthetic"
            }
            self.examples.append(example)
            count += 1
        print(f"  Generated {count} synthetic instruction examples")
    
    def _prepare_conversation_examples(self):
        """Prepare general conversation examples"""
        print("Loading conversation examples...")
        
        conversation_pairs = [
            ("Hello!", "Hi! How can I help?"),
            ("How are you?", "I'm well, thanks! How are you?"),
            ("What can you do?", "I help with math, weather, and questions."),
            ("Thank you", "You're welcome!"),
            ("Goodbye", "Goodbye!"),
        ]
        
        count = 0
        for question, answer in conversation_pairs * 20:  # Reduced from 50
            example = {
                "input": f"<user>{question}</user>",
                "output": f"<assistant>{answer}</assistant>",
                "type": "conversation"
            }
            self.examples.append(example)
            count += 1
        print(f"  Generated {count} conversation examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Combine input and output for training
        full_text = example["input"] + example["output"]
        
        # Tokenize with shorter max length
        tokens = self.tokenizer.encode(full_text, max_length=self.max_length)
        
        # Pad to max length
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
        
        # Create input (all but last token) and target (all but first token)
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

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def train_hrm_model():
    """Train the HRM model on comprehensive reasoning tasks"""
    
    # Clear GPU memory first
    clear_gpu_memory()
    
    # Set device with memory management
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
        print("Using CUDA device with memory fraction: 0.8")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create tokenizer and model with smaller dimensions
    vocab_size = 5000  # Reduced vocab size
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    
    model = create_hrm_model(
        vocab_size=len(tokenizer.vocab),
        dim=256,          # Reduced from 512
        n_heads=4,        # Reduced from 8
        N=2,              # Reduced from 3
        T=4,              # Reduced from 6
        use_act=True,
        dropout=0.1
    )
    
    model = model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset and dataloader with smaller batch size
    print("\nCreating dataset...")
    dataset = ComprehensiveReasoningDataset(tokenizer, max_length=128, max_examples_per_type=500)  # Reduced sizes
    
    dataloader = DataLoader(
        dataset, 
        batch_size=4,     # Reduced from 16
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False  # Disable pin_memory to save GPU memory
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * 15)
    
    # Training loop with memory management
    print("\nStarting training...")
    model.train()
    total_steps = 0
    best_loss = float('inf')
    
    for epoch in range(15):  # Reduced from 25
        epoch_loss = 0
        epoch_steps = 0
        type_losses = {}
        
        for batch_idx, batch in enumerate(dataloader):
            # Clear memory periodically
            if batch_idx % 50 == 0:
                clear_gpu_memory()
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            types = batch["types"]
            
            optimizer.zero_grad()
            
            try:
                # Forward pass with HRM
                result = model(
                    input_ids, 
                    max_segments=2,   # Reduced from 4
                    min_segments=1,
                    epsilon=0.1,
                    training=True
                )
                
                # Compute loss with deep supervision
                loss = model.compute_loss(result['outputs'], target_ids, result['q_values'])
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced from 1.0
                
                optimizer.step()
                scheduler.step()
                
                # Track losses by type
                for data_type in types:
                    if data_type not in type_losses:
                        type_losses[data_type] = []
                    type_losses[data_type].append(loss.item())
                
                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1
                
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}, Segments = {result['num_segments']}")
                    if torch.cuda.is_available():
                        print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch {batch_idx}, skipping...")
                    clear_gpu_memory()
                    continue
                else:
                    raise e
        
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('inf')
        
        # Print loss by type
        print(f"\nEpoch {epoch} completed: Avg Loss = {avg_loss:.4f}")
        print("Loss by type:")
        for data_type, losses in type_losses.items():
            if losses:
                avg_type_loss = sum(losses) / len(losses)
                print(f"  {data_type}: {avg_type_loss:.4f} ({len(losses)} batches)")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'tokenizer': tokenizer,
            }, 'hrm_best_model.pt')
            print(f"Saved best model with loss {avg_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'tokenizer': tokenizer,
            }, f'hrm_checkpoint_epoch_{epoch+1}.pt')
            print(f"Saved checkpoint: hrm_checkpoint_epoch_{epoch+1}.pt")
        
        # Clear memory after each epoch
        clear_gpu_memory()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'final_loss': avg_loss,
    }, 'hrm_final_model.pt')
    print("\nTraining completed! Final model saved as 'hrm_final_model.pt'")
    print(f"Best loss achieved: {best_loss:.4f}")

if __name__ == "__main__":
    # Set environment variable for better memory management
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Install required packages if not already installed
    try:
        import datasets
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets", "transformers"])
    
    train_hrm_model()
    