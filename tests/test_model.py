import torch
from human_agent.core.model import create_hrm_model

def test_model_creation():
    """Test that HRM model can be created"""
    model = create_hrm_model(vocab_size=1000, dim=128, N=2, T=4)
    assert model is not None
    assert model.vocab_size == 1000
    assert model.dim == 128

def test_model_forward():
    """Test model forward pass"""
    model = create_hrm_model(vocab_size=100, dim=64, N=1, T=2)
    
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, 100, (batch_size, seq_len))
    
    result = model(x, max_segments=2, training=False)
    
    assert 'final_output' in result
    assert result['final_output'].shape == (batch_size, seq_len, 100)
    assert result['num_segments'] <= 2

def test_model_loss():
    """Test loss computation"""
    model = create_hrm_model(vocab_size=50, dim=32, N=1, T=2)
    
    batch_size = 2
    seq_len = 8
    inputs = torch.randint(0, 50, (batch_size, seq_len))
    targets = torch.randint(0, 50, (batch_size, seq_len))
    
    result = model(inputs, max_segments=2, training=True)
    loss = model.compute_loss(result['outputs'], targets, result['q_values'])
    
    assert loss.item() > 0
    assert not torch.isnan(loss)
