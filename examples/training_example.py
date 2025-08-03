import torch
import torch.optim as optim
from human_agent.core.model import create_hrm_model

def main():
    """Example training script for Human Agent"""
    # Create model for a simple reasoning task
    vocab_size = 100
    model = create_hrm_model(
        vocab_size=vocab_size,
        dim=256,
        N=2,      # 2 high-level cycles  
        T=4,      # 4 low-level steps per cycle
        use_act=True
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Training loop example
    model.train()
    for epoch in range(10):
        # Generate dummy data (replace with actual task data)
        batch_size = 4
        seq_len = 64
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        optimizer.zero_grad()
        
        # Forward pass with deep supervision
        result = model(inputs, max_segments=4, training=True)
        
        # Compute loss
        loss = model.compute_loss(result['outputs'], targets, result['q_values'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Segments = {result['num_segments']}")

if __name__ == "__main__":
    main()
