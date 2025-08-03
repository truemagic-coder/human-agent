import re
from typing import List, Optional

class SimpleTokenizer:
    """Simple tokenizer for HRM model"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<function_call>': 4,
            '<function_result>': 5,
            '<user>': 6,
            '<assistant>': 7,
            '<system>': 8,
        }
        
        # Build basic vocabulary
        self.vocab = {token: idx for token, idx in self.special_tokens.items()}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        # Add common tokens, numbers, punctuation
        common_tokens = []
        
        # Add alphabet and common words
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            common_tokens.append(c)
        
        # Add digits
        for d in '0123456789':
            common_tokens.append(d)
            
        # Add punctuation and special characters
        punct = '.,!?;:"()[]{}=-+*/\\|`~@#$%^&_<> \n\t'
        for p in punct:
            common_tokens.append(p)
            
        # Add common words
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'function', 'call', 'result', 'return', 'def', 'import', 'from', 'as',
            'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally',
            'class', 'self', 'None', 'True', 'False', 'print', 'len', 'str', 'int'
        ]
        
        common_tokens.extend(common_words)
        
        # Add tokens to vocab
        current_idx = len(self.special_tokens)
        for token in common_tokens:
            if token not in self.vocab and current_idx < vocab_size:
                self.vocab[token] = current_idx
                self.reverse_vocab[current_idx] = token
                current_idx += 1
                
        self.pad_token_id = self.special_tokens['<pad>']
        self.eos_token_id = self.special_tokens['<eos>']
        self.bos_token_id = self.special_tokens['<bos>']
        
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token ids"""
        # Simple word-level tokenization
        tokens = []
        
        # Split by whitespace and punctuation, keeping delimiters
        words = re.findall(r'\w+|[^\w\s]|\s+', text)
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Character-level fallback
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.special_tokens['<unk>'])
        
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        # Basic cleanup
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
