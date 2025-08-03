import torch
import torch.optim as optim
import re
import random
import gc
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from human_agent.core.model import create_hrm_model
from human_agent.core.tokenizer import SimpleTokenizer

class EnhancedReasoningDataset(Dataset):
    """Enhanced dataset for training reasoning and function calling with improved capabilities"""
    
    def __init__(self, tokenizer: SimpleTokenizer, max_length: int = 256, max_examples_per_type: int = 1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.max_examples_per_type = max_examples_per_type
        
        # Load and prepare different types of examples
        print("Loading enhanced training datasets...")
        self._prepare_enhanced_math_reasoning()     # Fixed math with proper operators
        self._prepare_multi_turn_conversations()    # Multi-turn dialogue examples
        self._prepare_natural_response_training()   # Better response generation
        self._prepare_function_calling()            # Tool use examples
        self._prepare_error_handling()              # Error handling and edge cases
        self._prepare_conversation_examples()       # Basic chat examples
        self._prepare_complex_reasoning()           # Advanced reasoning tasks
        
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
    
    def _prepare_enhanced_math_reasoning(self):
        """Prepare enhanced mathematical reasoning with fixed operators"""
        print("Loading enhanced mathematical reasoning...")
        
        # 1. Load GSM8K with better parsing
        try:
            dataset = load_dataset("gsm8k", "main", split="train[:500]")
            count = 0
            for item in dataset:
                if count >= self.max_examples_per_type // 4:
                    break
                    
                question = item['question']
                answer = item['answer']
                
                # Extract numerical answer
                answer_match = re.search(r'#### (\d+(?:\.\d+)?)', answer)
                if answer_match:
                    final_answer = answer_match.group(1)
                    
                    # Extract and fix mathematical expressions
                    expressions = re.findall(r'(\d+(?:\.\d+)?[\+\-\*\/]\d+(?:\.\d+)?)', answer)
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
        
        # 2. Enhanced math examples with proper operators
        self._add_enhanced_math_examples()
        
        # 3. Complex math expressions
        self._add_complex_math_examples()
        
        # 4. Percentage and special calculations
        self._add_percentage_examples()
    
    def _add_enhanced_math_examples(self):
        """Add enhanced math examples with correct operators"""
        math_templates = [
            # Basic operations
            ("What is {a} + {b}?", "{a} + {b}", lambda a, b: a + b),
            ("Calculate {a} * {b}", "{a} * {b}", lambda a, b: a * b),
            ("What's {a} - {b}?", "{a} - {b}", lambda a, b: a - b),
            ("Compute {a} / {b}", "{a} / {b}", lambda a, b: round(a / b, 2) if b != 0 else "Error"),
            
            # Fixed exponentiation - use ** instead of ^
            ("What is {a} to the power of {b}?", "{a} ** {b}", lambda a, b: a ** b),
            ("Calculate {a}^{b}", "{a} ** {b}", lambda a, b: a ** b),
            ("What's {a} squared?", "{a} ** 2", lambda a: a ** 2),
            ("What's {a} cubed?", "{a} ** 3", lambda a: a ** 3),
            
            # Alternative expressions for exponentiation
            ("Find {a} raised to the {b}", "{a} ** {b}", lambda a, b: a ** b),
            ("{a} to the {b}th power", "{a} ** {b}", lambda a, b: a ** b),
        ]
        
        count = 0
        for template in math_templates:
            for _ in range(80):  # Generate examples per template
                try:
                    if template[2].__code__.co_argcount == 2:
                        a = random.randint(1, 20)
                        # Smaller exponents to avoid huge numbers
                        b = random.randint(1, 5) if "**" in template[1] else random.randint(1, 20)
                        # Avoid division by zero
                        if "/" in template[1] and b == 0:
                            b = random.randint(1, 10)
                        
                        result = template[2](a, b)
                        if result == "Error":
                            continue
                            
                        question = template[0].format(a=a, b=b)
                        expression = template[1].format(a=a, b=b)
                    else:  # Single argument
                        a = random.randint(2, 15)
                        result = template[2](a)
                        question = template[0].format(a=a)
                        expression = template[1].format(a=a)
                    
                    example = {
                        "input": f"<user>{question}</user>",
                        "output": f"<assistant><function_call>calculate(expression=\"{expression}\")</function_call></assistant><function_result>{result}</function_result><assistant>The answer is {result}.</assistant>",
                        "type": "math_enhanced"
                    }
                    self.examples.append(example)
                    count += 1
                except Exception:
                    continue
        
        print(f"  Generated {count} enhanced math examples")
    
    def _add_complex_math_examples(self):
        """Add complex mathematical expressions with proper parentheses"""
        complex_templates = [
            # Parentheses expressions
            ("Calculate ({a} + {b}) * {c}", "({a} + {b}) * {c}", lambda a, b, c: (a + b) * c),
            ("What's {a} * ({b} + {c})?", "{a} * ({b} + {c})", lambda a, b, c: a * (b + c)),
            ("Compute ({a} - {b}) / {c}", "({a} - {b}) / {c}", lambda a, b, c: round((a - b) / c, 2) if c != 0 else "Error"),
            ("Find ({a} + {b}) / {c}", "({a} + {b}) / {c}", lambda a, b, c: round((a + b) / c, 2) if c != 0 else "Error"),
            ("Calculate {a} / ({b} + {c})", "{a} / ({b} + {c})", lambda a, b, c: round(a / (b + c), 2) if (b + c) != 0 else "Error"),
            
            # Nested operations
            ("What is ({a} * {b}) + ({c} * {d})?", "({a} * {b}) + ({c} * {d})", lambda a, b, c, d: (a * b) + (c * d)),
            ("Calculate ({a} + {b}) - ({c} - {d})", "({a} + {b}) - ({c} - {d})", lambda a, b, c, d: (a + b) - (c - d)),
            ("Find ({a} * {b}) / ({c} + {d})", "({a} * {b}) / ({c} + {d})", lambda a, b, c, d: round((a * b) / (c + d), 2) if (c + d) != 0 else "Error"),
        ]
        
        count = 0
        for template in complex_templates:
            for _ in range(50):
                try:
                    if template[2].__code__.co_argcount == 3:
                        a, b, c = random.randint(1, 15), random.randint(1, 15), random.randint(1, 15)
                        # Avoid division by zero
                        if "/" in template[1] and c == 0:
                            c = random.randint(1, 10)
                        result = template[2](a, b, c)
                    else:  # 4 arguments
                        a, b, c, d = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
                        # Avoid division by zero
                        if "/" in template[1] and (c + d) == 0:
                            c, d = random.randint(1, 5), random.randint(1, 5)
                        result = template[2](a, b, c, d)
                    
                    if result == "Error":
                        continue
                        
                    question = template[0].format(a=a, b=b, c=c, d=d if template[2].__code__.co_argcount == 4 else None)
                    expression = template[1].format(a=a, b=b, c=c, d=d if template[2].__code__.co_argcount == 4 else None)
                    
                    # Clean up None in expression
                    question = question.replace(", d=None", "").replace("{d}", "")
                    expression = expression.replace(", d=None", "").replace("{d}", "")
                    
                    example = {
                        "input": f"<user>{question}</user>",
                        "output": f"<assistant><function_call>calculate(expression=\"{expression}\")</function_call></assistant><function_result>{result}</function_result><assistant>The calculation gives us {result}.</assistant>",
                        "type": "math_complex"
                    }
                    self.examples.append(example)
                    count += 1
                except Exception:
                    continue
        
        print(f"  Generated {count} complex math examples")
    
    def _add_percentage_examples(self):
        """Add percentage calculation examples"""
        percentage_templates = [
            ("What's {a}% of {b}?", "{a} / 100 * {b}", lambda a, b: round(a / 100 * b, 2)),
            ("Calculate {a} percent of {b}", "{a} / 100 * {b}", lambda a, b: round(a / 100 * b, 2)),
            ("Find {a}% of {b}", "{a} * {b} / 100", lambda a, b: round(a * b / 100, 2)),
            ("What is {a} percent of {b}?", "({a} / 100) * {b}", lambda a, b: round((a / 100) * b, 2)),
        ]
        
        count = 0
        for template in percentage_templates:
            for _ in range(40):
                a = random.randint(5, 50)  # Percentage
                b = random.randint(10, 200)  # Base number
                
                result = template[2](a, b)
                question = template[0].format(a=a, b=b)
                expression = template[1].format(a=a, b=b)
                
                example = {
                    "input": f"<user>{question}</user>",
                    "output": f"<assistant><function_call>calculate(expression=\"{expression}\")</function_call></assistant><function_result>{result}</function_result><assistant>{a}% of {b} is {result}.</assistant>",
                    "type": "math_percentage"
                }
                self.examples.append(example)
                count += 1
        
        print(f"  Generated {count} percentage examples")
    
    def _prepare_multi_turn_conversations(self):
        """Add multi-turn conversation examples"""
        print("Loading multi-turn conversation examples...")
        
        multi_turn_scenarios = [
            {
                "name": "math_sequence",
                "turns": [
                    ("Hello! I need help with some calculations.", "Hello! I'd be happy to help with your calculations."),
                    ("What is 15 * 8?", "<function_call>calculate(expression=\"15 * 8\")</function_call>", "120", "15 times 8 equals 120."),
                    ("Now add 25 to that result.", "<function_call>calculate(expression=\"120 + 25\")</function_call>", "145", "120 plus 25 equals 145."),
                    ("Perfect! Now divide that by 5.", "<function_call>calculate(expression=\"145 / 5\")</function_call>", "29.0", "145 divided by 5 equals 29."),
                    ("Thank you so much!", "You're welcome! I'm glad I could help with your calculations.")
                ]
            },
            {
                "name": "mixed_info",
                "turns": [
                    ("I need some information please.", "I'm here to help! What information do you need?"),
                    ("What's the weather like in Tokyo?", "<function_call>get_weather(location=\"Tokyo\")</function_call>", "sunny, 22°C", "The weather in Tokyo is sunny with a temperature of 22°C."),
                    ("What time is it right now?", "<function_call>get_current_time()</function_call>", "2025-08-02 15:30:00", "The current time is 15:30 on August 2nd, 2025."),
                    ("Can you calculate 25 * 4?", "<function_call>calculate(expression=\"25 * 4\")</function_call>", "100", "25 times 4 equals 100."),
                    ("Thanks for all the help!", "You're very welcome! Feel free to ask if you need anything else.")
                ]
            },
            {
                "name": "weather_comparison",
                "turns": [
                    ("I'm planning a trip and need weather info.", "I can help you with weather information for your trip planning."),
                    ("How's the weather in London?", "<function_call>get_weather(location=\"London\")</function_call>", "cloudy, 18°C", "London is currently cloudy with a temperature of 18°C."),
                    ("What about Paris?", "<function_call>get_weather(location=\"Paris\")</function_call>", "rainy, 16°C", "Paris is rainy with a temperature of 16°C."),
                    ("And Tokyo?", "<function_call>get_weather(location=\"Tokyo\")</function_call>", "sunny, 25°C", "Tokyo is sunny and 25°C."),
                    ("Tokyo sounds best! Thanks.", "Great choice! Tokyo does have the nicest weather of the three. Have a wonderful trip!")
                ]
            }
        ]
        
        count = 0
        for scenario in multi_turn_scenarios:
            for variation in range(30):  # Generate variations
                conversation_input = ""
                conversation_output = ""
                
                # Build the conversation progressively
                for turn_idx, turn in enumerate(scenario["turns"]):
                    if len(turn) == 2:  # Regular conversation turn
                        user_msg, assistant_msg = turn
                        conversation_input += f"<user>{user_msg}</user>"
                        if turn_idx < len(scenario["turns"]) - 1:  # Not the last turn
                            conversation_input += f"<assistant>{assistant_msg}</assistant>"
                        else:  # Last turn - this is what we're predicting
                            conversation_output = f"<assistant>{assistant_msg}</assistant>"
                    else:  # Function call turn
                        user_msg, func_call, func_result, assistant_response = turn
                        conversation_input += f"<user>{user_msg}</user>"
                        if turn_idx < len(scenario["turns"]) - 1:  # Not the last turn
                            conversation_input += f"<assistant>{func_call}</assistant><function_result>{func_result}</function_result><assistant>{assistant_response}</assistant>"
                        else:  # Last turn - predict the function call
                            conversation_output = f"<assistant>{func_call}</assistant><function_result>{func_result}</function_result><assistant>{assistant_response}</assistant>"
                
                example = {
                    "input": conversation_input,
                    "output": conversation_output,
                    "type": f"multi_turn_{scenario['name']}"
                }
                self.examples.append(example)
                count += 1
        
        print(f"  Generated {count} multi-turn conversation examples")
    
    def _prepare_natural_response_training(self):
        """Add examples for better natural language responses"""
        print("Loading natural response training examples...")
        
        # Math response variations
        math_response_patterns = [
            ("calculate", [
                "The calculation gives us {result}.",
                "The result is {result}.",
                "That equals {result}.",
                "The answer is {result}.",
                "This evaluates to {result}.",
                "The solution is {result}.",
                "{expr} equals {result}.",
                "Computing this gives {result}."
            ]),
        ]
        
        # Weather response variations
        weather_response_patterns = [
            ("get_weather", [
                "The weather in {location} is {condition} with a temperature of {temp}.",
                "Currently in {location}, it's {condition} and {temp}.",
                "{location} is experiencing {condition} weather at {temp}.",
                "It's {condition} in {location} with {temp}.",
                "The current conditions in {location} are {condition}, {temp}."
            ]),
        ]
        
        # Time response variations
        time_response_patterns = [
            ("get_current_time", [
                "The current time is {time}.",
                "Right now it's {time}.",
                "The time is {time}.",
                "Currently, the time is {time}.",
                "It's {time} right now."
            ]),
        ]
        
        count = 0
        
        # Generate math response examples
        for _, responses in math_response_patterns:
            for response_template in responses:
                for _ in range(25):
                    a, b = random.randint(1, 50), random.randint(1, 50)
                    op = random.choice(['+', '-', '*'])
                    expr = f"{a} {op} {b}"
                    
                    if op == '+':
                        result = a + b
                        question = f"What is {expr}?"
                    elif op == '-':
                        result = a - b
                        question = f"Calculate {expr}"
                    else:  # multiplication
                        result = a * b
                        question = f"What's {expr}?"
                    
                    natural_response = response_template.format(result=result, expr=expr)
                    
                    example = {
                        "input": f"<user>{question}</user><assistant><function_call>calculate(expression=\"{expr}\")</function_call></assistant><function_result>{result}</function_result>",
                        "output": f"<assistant>{natural_response}</assistant>",
                        "type": "natural_math_response"
                    }
                    self.examples.append(example)
                    count += 1
        
        # Generate weather response examples
        for _, responses in weather_response_patterns:
            cities = ["Tokyo", "London", "Paris", "New York", "Sydney", "Berlin", "Madrid", "Rome"]
            conditions = ["sunny", "cloudy", "rainy", "snowy", "foggy"]
            
            for response_template in responses:
                for _ in range(20):
                    city = random.choice(cities)
                    condition = random.choice(conditions)
                    temp = f"{random.randint(5, 35)}°C"
                    
                    question = f"What's the weather in {city}?"
                    weather_result = f"The weather in {city} is {condition} with a temperature of {temp.replace('°C', '°C')}"
                    natural_response = response_template.format(location=city, condition=condition, temp=temp)
                    
                    example = {
                        "input": f"<user>{question}</user><assistant><function_call>get_weather(location=\"{city}\")</function_call></assistant><function_result>{weather_result}</function_result>",
                        "output": f"<assistant>{natural_response}</assistant>",
                        "type": "natural_weather_response"
                    }
                    self.examples.append(example)
                    count += 1
        
        # Generate time response examples (was missing!)
        for _, responses in time_response_patterns:
            for response_template in responses:
                for _ in range(20):
                    question = random.choice([
                        "What time is it?",
                        "Current time please",
                        "Tell me the time",
                        "What's the current time?",
                        "Show me the time"
                    ])
                    
                    # Generate a realistic timestamp
                    import datetime
                    now = datetime.datetime.now()
                    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    
                    natural_response = response_template.format(time=time_str)
                    
                    example = {
                        "input": f"<user>{question}</user><assistant><function_call>get_current_time()</function_call></assistant><function_result>{time_str}</function_result>",
                        "output": f"<assistant>{natural_response}</assistant>",
                        "type": "natural_time_response"
                    }
                    self.examples.append(example)
                    count += 1

        print(f"  Generated {count} natural response examples")
    
    def _prepare_function_calling(self):
        """Prepare function calling examples with better variety"""
        print("Loading enhanced function calling examples...")
        
        # Enhanced weather examples with more variety
        weather_templates = [
            "What's the weather in {city}?",
            "How's the weather in {city}?",
            "Weather in {city}?",
            "What are the weather conditions in {city}?",
            "Tell me about the weather in {city}",
            "Is it sunny in {city}?",
            "What's the temperature in {city}?",
            "How's the climate in {city} today?",
            "Weather forecast for {city}?",
            "Current weather in {city}?"
        ]
        
        cities = ["Tokyo", "London", "New York", "Paris", "Sydney", "Berlin", "Madrid", "Rome", "Bangkok", "Dubai"]
        
        count = 0
        for _ in range(150):
            city = random.choice(cities)
            question = random.choice(weather_templates).format(city=city)
            
            example = {
                "input": f"<user>{question}</user>",
                "output": f"<assistant><function_call>get_weather(location=\"{city}\")</function_call></assistant>",
                "type": "function_weather"
            }
            self.examples.append(example)
            count += 1
        
        # Enhanced time examples
        time_templates = [
            "What time is it?",
            "Current time?",
            "Tell me the time",
            "What's the current time?",
            "Show me the time",
            "Time please",
            "What time is it now?",
            "Current date and time?",
            "Tell me the current date and time",
            "What's today's date and time?"
        ]
        
        for question in time_templates * 8:
            example = {
                "input": f"<user>{question}</user>",
                "output": "<assistant><function_call>get_current_time()</function_call></assistant>",
                "type": "function_time"
            }
            self.examples.append(example)
            count += len(time_templates) * 8
        
        print(f"  Generated {count} enhanced function calling examples")
    
    def _prepare_error_handling(self):
        """Add examples for error handling and edge cases"""
        print("Loading error handling examples...")
        
        error_scenarios = [
            # Math errors
            ("Calculate 5 / 0", "I can't divide by zero. Please try a different calculation."),
            ("What's abc + 5?", "I need valid numbers for calculations. Could you rephrase with numbers?"),
            ("Calculate xyz", "I don't understand that expression. Please provide a valid mathematical expression."),
            ("What's the square root?", "I need a number to calculate the square root of. For example, 'What's the square root of 16?'"),
            
            # Unclear function requests
            ("Weather", "I'd be happy to get weather information! Which city would you like to know about?"),
            ("Calculate", "I can help with calculations! What mathematical expression would you like me to evaluate?"),
            ("Time", "Would you like me to tell you the current time?"),
            
            # Unsupported requests with helpful redirects
            ("What's my horoscope?", "I can help with calculations, weather information, and current time. I don't provide horoscope readings."),
            ("Translate hello to French", "I specialize in math calculations, weather, and time information. For translations, you might want to use a translation service."),
            ("What's the meaning of life?", "That's a deep philosophical question! I'm designed to help with calculations, weather, and time information."),
            ("Tell me a joke", "I focus on practical help like calculations and weather information. What can I calculate or look up for you?"),
            
            # Ambiguous location requests
            ("Weather in the city", "Which city would you like weather information for? Please specify the city name."),
            ("What's the weather there?", "I need to know which location you're asking about. Could you specify the city?"),
            
            # Incomplete math requests
            ("What's 5 plus", "It looks like your math expression is incomplete. What would you like to add 5 to?"),
            ("Calculate times 3", "I need the complete expression. What number would you like to multiply by 3?"),
        ]
        
        count = 0
        for question, response in error_scenarios:
            for _ in range(15):  # Generate multiple variations
                example = {
                    "input": f"<user>{question}</user>",
                    "output": f"<assistant>{response}</assistant>",
                    "type": "error_handling"
                }
                self.examples.append(example)
                count += 1
        
        print(f"  Generated {count} error handling examples")
    
    def _prepare_conversation_examples(self):
        """Prepare general conversation examples"""
        print("Loading conversation examples...")
        
        conversation_pairs = [
            # Greetings
            ("Hello!", "Hello! How can I help you today?"),
            ("Hi there!", "Hi! What can I do for you?"),
            ("Hey!", "Hey! How can I assist you?"),
            ("Good morning!", "Good morning! How may I help you?"),
            ("Good afternoon!", "Good afternoon! What can I do for you?"),
            ("Good evening!", "Good evening! How can I assist you?"),
            
            # Status questions
            ("How are you?", "I'm doing well, thank you! How are you doing?"),
            ("How's it going?", "Things are going well! How can I help you today?"),
            ("What's up?", "I'm here and ready to help! What do you need?"),
            
            # Capability questions
            ("What can you do?", "I can help with mathematical calculations, weather information, and tell you the current time. What would you like to know?"),
            ("What are your capabilities?", "I specialize in math calculations, weather lookups, and providing the current time. How can I assist you?"),
            ("How can you help me?", "I can solve math problems, check weather conditions, and tell you the time. What do you need help with?"),
            
            # Thanks and goodbyes
            ("Thank you!", "You're welcome! Is there anything else I can help you with?"),
            ("Thanks a lot!", "You're very welcome! Feel free to ask if you need anything else."),
            ("Thanks for your help!", "I'm happy to help! Let me know if you need anything else."),
            ("Goodbye!", "Goodbye! Have a great day!"),
            ("See you later!", "See you later! Take care!"),
            ("Bye!", "Bye! Have a wonderful day!"),
            
            # Clarification requests
            ("I need help", "I'd be happy to help! What do you need assistance with? I can help with calculations, weather, or time."),
            ("Can you assist me?", "Absolutely! I can help with math problems, weather information, or tell you the current time. What do you need?"),
            ("I have a question", "Sure! I'm here to help. What's your question? I can assist with calculations, weather, or time."),
        ]
        
        count = 0
        for question, answer in conversation_pairs:
            for _ in range(12):  # Generate variations
                example = {
                    "input": f"<user>{question}</user>",
                    "output": f"<assistant>{answer}</assistant>",
                    "type": "conversation"
                }
                self.examples.append(example)
                count += 1
        
        print(f"  Generated {count} conversation examples")
    
    def _prepare_complex_reasoning(self):
        """Add complex reasoning examples that require multiple steps"""
        print("Loading complex reasoning examples...")
        
        complex_scenarios = [
            # Sequential calculations
            {
                "question": "I have 10 apples, I eat 3, then buy 5 more, then give away half. How many do I have?",
                "steps": [
                    ("10 - 3", 7, "After eating 3 apples, you have 7 left."),
                    ("7 + 5", 12, "After buying 5 more, you have 12 apples."),
                    ("12 / 2", 6, "After giving away half, you have 6 apples.")
                ],
                "final": "You end up with 6 apples."
            },
            {
                "question": "If I work 8 hours a day for 5 days, and earn $15 per hour, how much do I make?",
                "steps": [
                    ("8 * 5", 40, "You work 40 hours total."),
                    ("40 * 15", 600, "At $15 per hour, you earn $600.")
                ],
                "final": "You make $600 for the week."
            },
            {
                "question": "A rectangle is 12 units long and 8 units wide. What's its perimeter and area?",
                "steps": [
                    ("2 * (12 + 8)", 40, "The perimeter is 40 units."),
                    ("12 * 8", 96, "The area is 96 square units.")
                ],
                "final": "The rectangle has a perimeter of 40 units and an area of 96 square units."
            }
        ]
        
        count = 0
        for scenario in complex_scenarios:
            for _ in range(10):  # Generate variations
                # Create examples for each step
                conversation = f"<user>{scenario['question']}</user>"
                
                for step_expr, step_result, step_explanation in scenario['steps']:
                    conversation += f"<assistant><function_call>calculate(expression=\"{step_expr}\")</function_call></assistant><function_result>{step_result}</function_result><assistant>{step_explanation}</assistant>"
                
                conversation += f"<assistant>{scenario['final']}</assistant>"
                
                # Extract just the final step for training
                example = {
                    "input": conversation.rsplit('<assistant>', 1)[0],
                    "output": f"<assistant>{scenario['final']}</assistant>",
                    "type": "complex_reasoning"
                }
                self.examples.append(example)
                count += 1
        
        print(f"  Generated {count} complex reasoning examples")
    
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

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def train_enhanced_hrm_model():
    """Train the enhanced HRM model with curriculum learning and stability fixes"""
    
    # Clear GPU memory first
    clear_gpu_memory()
    
    # Set device with memory management
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.85)
        print("Using CUDA device with memory fraction: 0.85")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Use smaller, more stable model configuration
    tokenizer = SimpleTokenizer(vocab_size=1000)  # Much smaller vocab
    
    # More conservative model architecture for stability
    model = create_hrm_model(
        vocab_size=len(tokenizer.vocab),
        dim=256,          # Back to original size
        n_heads=4,        # Back to original
        N=2,              # Back to original
        T=4,              # Back to original
        use_act=True,
        dropout=0.1       # Lower dropout
    )
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Create dataset with smaller examples
    print("\nCreating enhanced dataset...")
    dataset = EnhancedReasoningDataset(
        tokenizer, 
        max_length=128,   # Reduced sequence length
        max_examples_per_type=200  # Fewer examples for stability
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=4,     # Smaller batch size
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # Much more conservative optimizer settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=5e-5,          # Much lower learning rate
        weight_decay=0.01, # Lower weight decay
        betas=(0.9, 0.999), # Standard betas
        eps=1e-8
    )
    
    # Simple step scheduler instead of OneCycle
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=5,
        gamma=0.8
    )
    
    # Simplified curriculum learning
    curriculum_phases = [
        {
            "epochs": range(0, 5),
            "focus": ["conversation", "math_enhanced"],
            "description": "Basic conversation and simple math"
        },
        {
            "epochs": range(5, 10),
            "focus": ["function_weather", "function_time", "natural_math_response"],
            "description": "Function calling and responses"
        },
        {
            "epochs": range(10, 15),
            "focus": None,  # All types
            "description": "Full curriculum"
        }
    ]
    
    # Training loop with stability improvements
    print("\nStarting stable training...")
    model.train()
    best_loss = float('inf')
    
    # Initialize model weights properly
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)
    
    model.apply(init_weights)
    print("Applied conservative weight initialization")
    
    for epoch in range(15):  # Fewer epochs
        # Determine current curriculum phase
        current_phase = None
        for phase in curriculum_phases:
            if epoch in phase["epochs"]:
                current_phase = phase
                break
        
        if current_phase["focus"]:
            print(f"\nEpoch {epoch} - Phase: {current_phase['description']}")
            print(f"Focusing on: {', '.join(current_phase['focus'])}")
            
            # Filter examples based on curriculum
            filtered_indices = [
                i for i, example in enumerate(dataset.examples) 
                if example["type"] in current_phase["focus"]
            ]
            
            # Create subset dataset
            from torch.utils.data import Subset
            subset_dataset = Subset(dataset, filtered_indices)
            current_dataloader = DataLoader(
                subset_dataset,
                batch_size=4,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False
            )
        else:
            print(f"\nEpoch {epoch} - Phase: {current_phase['description']}")
            current_dataloader = dataloader
        
        epoch_loss = 0
        epoch_steps = 0
        type_losses = {}
        
        for batch_idx, batch in enumerate(current_dataloader):
            if batch_idx % 20 == 0:
                clear_gpu_memory()
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            types = batch["types"]
            
            optimizer.zero_grad()
            
            try:
                # Conservative forward pass settings
                result = model(
                    input_ids, 
                    max_segments=2,   # Reduced
                    min_segments=1,
                    epsilon=0.1,      # Higher for more stable stopping
                    training=True
                )
                
                # Compute loss with stability checks
                loss = model.compute_loss(result['outputs'], target_ids, result['q_values'])
                
                # Check for extreme loss values
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                    print(f"Extreme loss detected: {loss.item()}, skipping batch...")
                    continue
                
                # Backward pass with gradient accumulation
                loss = loss / 2  # Gradient accumulation factor
                loss.backward()
                
                # Conservative gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # Only step if gradients are reasonable
                if grad_norm < 10.0:  # Skip if gradients are too large
                    optimizer.step()
                else:
                    print(f"Large gradient norm {grad_norm:.2f}, skipping step...")
                    optimizer.zero_grad()
                    continue
                
                # Track losses
                loss_value = loss.item() * 2  # Undo accumulation scaling
                for data_type in types:
                    if data_type not in type_losses:
                        type_losses[data_type] = []
                    type_losses[data_type].append(loss_value)
                
                epoch_loss += loss_value
                epoch_steps += 1
                
                if batch_idx % 20 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Batch {batch_idx}: Loss = {loss_value:.4f}, LR = {current_lr:.2e}, Segments = {result['num_segments']}, GradNorm = {grad_norm:.2f}")
                    if torch.cuda.is_available():
                        print(f"    GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch {batch_idx}, skipping...")
                    clear_gpu_memory()
                    continue
                else:
                    print(f"Runtime error: {e}")
                    continue
            except Exception as e:
                print(f"Unexpected error at batch {batch_idx}: {e}")
                continue
        
        # Step scheduler after epoch
        scheduler.step()
        
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('inf')
        
        # Print detailed statistics
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} completed: Avg Loss = {avg_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Processed {epoch_steps} batches successfully")
        
        if type_losses:
            print("Loss by type:")
            for data_type, losses in sorted(type_losses.items()):
                if losses:
                    avg_type_loss = sum(losses) / len(losses)
                    print(f"  {data_type}: {avg_type_loss:.4f} ({len(losses)} batches)")
        
        # Save best model with reasonable loss
        if avg_loss < best_loss and avg_loss < 100:  # Only save if loss is reasonable
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
                    'dim': 256,
                    'n_heads': 4,
                    'N': 2,
                    'T': 4,
                    'total_params': total_params
                }
            }, 'hrm_trained_model.pt')  # Changed back to original name
            print(f"🎯 Saved new best model with loss {avg_loss:.4f}")
        
        # Save checkpoints every 3 epochs
        if (epoch + 1) % 3 == 0 and avg_loss < 100:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'tokenizer': tokenizer,
            }, f'hrm_checkpoint_epoch_{epoch+1}.pt')  # Changed back to original naming
            print(f"💾 Saved checkpoint: hrm_checkpoint_epoch_{epoch+1}.pt")
        
        # Early stopping if loss becomes unreasonable
        if avg_loss > 1000:
            print("⚠️  Loss too high, stopping training early")
            break
        
        # Clear memory after each epoch
        clear_gpu_memory()
    
    # Save final model only if reasonable
    if avg_loss < 100:
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'final_loss': avg_loss,
            'config': {
                'vocab_size': len(tokenizer.vocab),
                'dim': 256,
                'n_heads': 4,
                'N': 2,
                'T': 4,
                'total_params': total_params
            }
        }, 'hrm_trained_model.pt')  # Changed back to original name
        
        print("\n🎉 Enhanced training completed!")
        print("📊 Best loss achieved: {best_loss:.4f}")
        print("💾 Models saved:")
        print("   - hrm_trained_model.pt (best performing)")
        print("   - hrm_checkpoint_epoch_*.pt (checkpoints)")
        print("🧠 Model specs: {total_params:,} parameters, 256 dim, 4 heads, N=2, T=4")
    else:
        print(f"\n⚠️  Training unstable, final loss too high: {avg_loss:.4f}")
        print("Consider further reducing learning rate or model complexity")

if __name__ == "__main__":
    # Set environment variables for optimal performance
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Install required packages if needed
    try:
        import datasets
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets", "transformers"])
    
    train_enhanced_hrm_model()

