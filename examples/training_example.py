import torch
import torch.optim as optim
import re
import random
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from human_agent.core.model import create_hrm_model
from human_agent.core.tokenizer import SimpleTokenizer

class ComprehensiveReasoningDataset(Dataset):
    """Dataset for training reasoning and function calling using multiple sources"""
    
    def __init__(self, tokenizer: SimpleTokenizer, max_length: int = 512, max_examples_per_type: int = 2000):
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
        
        # 1. GSM8K - Grade school math
        try:
            dataset = load_dataset("gsm8k", "main", split="train")
            count = 0
            for item in dataset:
                if count >= self.max_examples_per_type // 2:
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
        
        # 2. Competition Math
        try:
            dataset = load_dataset("competition_math", split="train")
            count = 0
            for item in dataset:
                if count >= self.max_examples_per_type // 4:
                    break
                    
                problem = item['problem']
                solution = item['solution']
                
                # Look for numerical answers
                answer_patterns = [
                    r'\\boxed{(\d+)}',
                    r'answer is (\d+)',
                    r'= (\d+)$'
                ]
                
                final_answer = None
                for pattern in answer_patterns:
                    match = re.search(pattern, solution)
                    if match:
                        final_answer = match.group(1)
                        break
                
                if final_answer and len(problem) < 300:  # Keep problems reasonably short
                    example = {
                        "input": f"<user>{problem}</user>",
                        "output": f"<assistant>Let me solve this step by step. The answer is {final_answer}.</assistant>",
                        "type": "math_competition"
                    }
                    self.examples.append(example)
                    count += 1
            print(f"  Loaded {count} Competition Math examples")
        except Exception as e:
            print(f"  Could not load Competition Math: {e}")
        
        # 3. Synthetic math examples
        self._add_synthetic_math_examples()
    
    def _add_synthetic_math_examples(self):
        """Add synthetic math examples"""
        math_templates = [
            ("What is {a} + {b}?", "{a} + {b}", lambda a, b: a + b),
            ("Calculate {a} * {b}", "{a} * {b}", lambda a, b: a * b),
            ("What's {a} - {b}?", "{a} - {b}", lambda a, b: a - b),
            ("Compute {a} / {b}", "{a} / {b}", lambda a, b: round(a / b, 2)),
            ("What is {a} squared?", "{a}^2", lambda a, b: a ** 2),
            ("Find {a} percent of {b}", "{a}/100 * {b}", lambda a, b: round((a/100) * b, 2)),
        ]
        
        count = 0
        for _ in range(self.max_examples_per_type // 4):
            template = random.choice(math_templates)
            a = random.randint(1, 100)
            b = random.randint(1, 50) if "/ {b}" in template[1] else random.randint(1, 100)
            
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
        
        # Weather examples
        weather_templates = [
            "What's the weather in {city}?",
            "How's the weather in {city} today?",
            "Tell me about the weather in {city}",
            "Weather forecast for {city}?",
            "Is it raining in {city}?",
            "Temperature in {city}?"
        ]
        
        cities = [
            "Tokyo", "London", "New York", "Paris", "Sydney", "Toronto", "Berlin", "Mumbai",
            "Los Angeles", "Chicago", "Miami", "Seattle", "Boston", "San Francisco", "Denver",
            "Rome", "Madrid", "Amsterdam", "Vienna", "Copenhagen", "Stockholm", "Helsinki"
        ]
        
        count = 0
        for _ in range(300):
            city = random.choice(cities)
            question = random.choice(weather_templates).format(city=city)
            
            example = {
                "input": f"<user>{question}</user>",
                "output": f"<assistant><function_call>get_weather(location=\"{city}\")</function_call></assistant>",
                "type": "function_weather"
            }
            self.examples.append(example)
            count += 1
        
        # Time examples
        time_questions = [
            "What time is it?", "Current time please", "Tell me the time",
            "What's the current date and time?", "What day is today?",
            "Current timestamp", "Show me the time", "Time right now?"
        ]
        
        for question in time_questions * 15:
            example = {
                "input": f"<user>{question}</user>",
                "output": "<assistant><function_call>get_current_time()</function_call></assistant>",
                "type": "function_time"
            }
            self.examples.append(example)
            count += len(time_questions) * 15
        
        print(f"  Generated {count} function calling examples")
    
    def _prepare_code_reasoning(self):
        """Prepare code reasoning examples"""
        print("Loading code reasoning datasets...")
        
        try:
            # CodeAlpaca dataset
            dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            count = 0
            for item in dataset:
                if count >= self.max_examples_per_type // 2:
                    break
                
                instruction = item['instruction']
                output = item['output']
                
                # Focus on shorter, cleaner examples
                if len(instruction) < 200 and len(output) < 500 and 'def ' in output:
                    example = {
                        "input": f"<user>{instruction}</user>",
                        "output": f"<assistant>{output}</assistant>",
                        "type": "code_generation"
                    }
                    self.examples.append(example)
                    count += 1
            print(f"  Loaded {count} CodeAlpaca examples")
        except Exception as e:
            print(f"  Could not load CodeAlpaca: {e}")
            self._add_synthetic_code_examples()
    
    def _add_synthetic_code_examples(self):
        """Add synthetic code examples"""
        code_templates = [
            ("Write a function to add two numbers", "def add(a, b):\n    return a + b"),
            ("Create a function to find the maximum of two numbers", "def max_num(a, b):\n    return a if a > b else b"),
            ("Write a function to calculate factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"),
            ("Create a function to check if a number is even", "def is_even(n):\n    return n % 2 == 0"),
        ]
        
        count = 0
        for instruction, code in code_templates * 50:
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
        
        try:
            # Alpaca dataset
            dataset = load_dataset("tatsu-lab/alpaca", split="train")
            count = 0
            for item in dataset:
                if count >= self.max_examples_per_type:
                    break
                
                instruction = item['instruction']
                input_text = item.get('input', '')
                output = item['output']
                
                # Create prompt
                if input_text:
                    prompt = f"{instruction}\n\nInput: {input_text}"
                else:
                    prompt = instruction
                
                # Keep examples reasonably short
                if len(prompt) < 300 and len(output) < 400:
                    example = {
                        "input": f"<user>{prompt}</user>",
                        "output": f"<assistant>{output}</assistant>",
                        "type": "instruction_following"
                    }
                    self.examples.append(example)
                    count += 1
            print(f"  Loaded {count} Alpaca examples")
        except Exception as e:
            print(f"  Could not load Alpaca: {e}")
            self._add_synthetic_instruction_examples()
    
    def _add_synthetic_instruction_examples(self):
        """Add synthetic instruction examples"""
        instruction_pairs = [
            ("Explain what artificial intelligence is", "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence."),
            ("List three benefits of renewable energy", "1. Environmental protection by reducing greenhouse gas emissions\n2. Energy independence and security\n3. Long-term cost savings and job creation"),
            ("Describe the water cycle", "The water cycle is the continuous movement of water through evaporation, condensation, precipitation, and collection processes on Earth."),
        ]
        
        count = 0
        for instruction, response in instruction_pairs * 100:
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
            ("Hello!", "Hi there! How can I help you today?"),
            ("How are you?", "I'm doing well, thank you for asking! How are you?"),
            ("What can you do?", "I can help with math calculations, weather information, coding questions, and general inquiries."),
            ("Thank you", "You're welcome! Is there anything else I can help you with?"),
            ("Goodbye", "Goodbye! Have a great day!"),
            ("Can you help me?", "Of course! I'd be happy to help. What do you need assistance with?"),
            ("What's your name?", "I'm an AI assistant here to help answer your questions and assist with various tasks."),
            ("How does this work?", "I can process your questions and provide helpful responses. Feel free to ask me anything!"),
        ]
        
        count = 0
        for question, answer in conversation_pairs * 50:
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
        
        # Tokenize
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

def train_hrm_model():
    """Train the HRM model on comprehensive reasoning tasks"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create tokenizer and model
    vocab_size = 8000  # Increased for better coverage
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    
    model = create_hrm_model(
        vocab_size=len(tokenizer.vocab),
        dim=512,
        n_heads=8,
        N=3,          # 3 high-level cycles for complex reasoning
        T=6,          # 6 low-level steps per cycle
        use_act=True,
        dropout=0.1
    )
    
    model = model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = ComprehensiveReasoningDataset(tokenizer, max_length=256, max_examples_per_type=1500)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=16,  # Increased batch size for better training
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * 25)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    total_steps = 0
    best_loss = float('inf')
    
    for epoch in range(25):  # 25 epochs for comprehensive training
        epoch_loss = 0
        epoch_steps = 0
        type_losses = {}
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            types = batch["types"]
            
            optimizer.zero_grad()
            
            # Forward pass with HRM
            result = model(
                input_ids, 
                max_segments=4,
                min_segments=1,
                epsilon=0.1,
                training=True
            )
            
            # Compute loss with deep supervision
            loss = model.compute_loss(result['outputs'], target_ids, result['q_values'])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}, Segments = {result['num_segments']}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        avg_loss = epoch_loss / epoch_steps
        
        # Print loss by type
        print(f"\nEpoch {epoch} completed: Avg Loss = {avg_loss:.4f}")
        print("Loss by type:")
        for data_type, losses in type_losses.items():
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
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'final_loss': avg_loss,
    }, 'hrm_final_model.pt')
    print("\nTraining completed! Final model saved as 'hrm_final_model.pt'")
    print(f"Best loss achieved: {best_loss:.4f}")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import datasets
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets", "transformers"])
    
    train_hrm_model()