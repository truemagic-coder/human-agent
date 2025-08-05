import re
from typing import List, Optional, Dict

class Tokenizer:
    """
    A robust tokenizer that correctly handles special tokens and provides
    a clean interface for encoding and decoding text for the HRM model.
    """
    
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        
        # Core special tokens required for any model
        self.base_special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
        
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.special_tokens_set = set()
        
        # Initialize with base tokens
        self.add_special_tokens(list(self.base_special_tokens.keys()))
        
        # Add common control characters
        for i in [9, 10, 13]:  # Tab, newline, carriage return
            self._add_token(chr(i))
        
        # Build the rest of the vocabulary
        self._build_vocab()
        
        self.pad_token_id = self.vocab['<pad>']
        self.eos_token_id = self.vocab['<eos>']
        self.bos_token_id = self.vocab['<bos>']
        
        # A regex to find special tokens or split by word boundaries
        self.special_tokens_regex = None
        self._compile_special_tokens_regex()

    def _build_vocab(self):
        """Builds a basic vocabulary from common characters and words."""
        # Add all ASCII printable characters to ensure basic coverage
        for i in range(32, 127):
            self._add_token(chr(i))
        
        # Add common English words for better tokenization efficiency
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it',
            'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this',
            'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or',
            'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            # Expanded list for better coverage
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could',
            'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come',
            'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how',
            'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because'
        ]
        for word in common_words:
            self._add_token(word)

    def _add_token(self, token: str):
        """Adds a token to the vocabulary if it doesn't exist and there's space."""
        if token not in self.vocab and len(self.vocab) < self.vocab_size:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
            return True
        return False

    def add_special_tokens(self, tokens: List[str]):
        """Adds a list of special tokens to the vocabulary."""
        for token in tokens:
            if self._add_token(token):
                self.special_tokens_set.add(token)
        # Re-compile the regex to include the new tokens
        self._compile_special_tokens_regex()

    def _compile_special_tokens_regex(self):
        """Compiles a regex to efficiently find special tokens."""
        if self.special_tokens_set:
            # Escape special characters for regex and join them with '|'
            escaped_tokens = [re.escape(token) for token in sorted(list(self.special_tokens_set), key=len, reverse=True)]
            self.special_tokens_regex = re.compile(f"({'|'.join(escaped_tokens)})")

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encodes text to a list of token IDs, correctly handling special tokens.
        """
        tokens = []
        
        # First, split the text by our special tokens pattern
        parts = self.special_tokens_regex.split(text)
        
        for part in parts:
            if not part:
                continue
            # If the part is a special token, encode it directly
            if part in self.special_tokens_set:
                tokens.append(self.vocab[part])
            else:
                # Otherwise, use a simple word/character tokenization for the rest
                # This regex splits by non-space sequences or space sequences
                sub_parts = re.findall(r'\S+|\s+', part)
                for sub_part in sub_parts:
                    if sub_part in self.vocab:
                        tokens.append(self.vocab[sub_part])
                    else:
                        # Fallback to character-level for unknown words
                        for char in sub_part:
                            tokens.append(self.vocab.get(char, self.vocab['<unk>']))
        
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of token IDs back to a string.
        This method now correctly handles all tokens, including special ones.
        """
        # Decode all tokens, including special ones
        tokens = [self.reverse_vocab.get(token_id, '<unk>') for token_id in token_ids]
        
        # Join tokens without additional spaces
        text = "".join(tokens)
        
        # Basic cleanup to handle spaces around punctuation
        # This assumes spaces are preserved as tokens
        text = re.sub(r' ([.,!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces to single
        return text.strip()
    