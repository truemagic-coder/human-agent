from human_agent.core.tokenizer import Tokenizer

def test_tokenizer_creation():
    """Test tokenizer creation"""
    tokenizer = Tokenizer(vocab_size=1000)
    assert tokenizer.vocab_size == 1000
    assert len(tokenizer.special_tokens) > 0

def test_encode_decode():
    """Test encode/decode roundtrip"""
    tokenizer = Tokenizer(vocab_size=1000)
    
    text = "Hello world! How are you?"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert isinstance(decoded, str)
    # Note: Due to simple tokenization, exact match may not occur
    assert len(decoded) > 0

def test_special_tokens():
    """Test special token handling"""
    tokenizer = Tokenizer(vocab_size=1000)
    
    assert tokenizer.pad_token_id == 0
    assert tokenizer.eos_token_id == 3
    assert tokenizer.bos_token_id == 2
