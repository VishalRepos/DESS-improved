#!/usr/bin/env python3
"""
Unit test for Contrastive Learning module
"""
import sys
sys.path.append('./Codebase')

import torch
from models.Contrastive_Module import SimplifiedContrastiveLoss, ContrastivePairEncoder


def test_simplified_contrastive_loss():
    print("Testing SimplifiedContrastiveLoss...")
    
    hidden_dim = 768
    num_triplets = 10
    
    # Create dummy data
    entity_spans = torch.randn(num_triplets, hidden_dim)
    opinion_spans = torch.randn(num_triplets, hidden_dim)
    
    # Initialize module
    contrastive_loss = SimplifiedContrastiveLoss(hidden_dim, temperature=0.07)
    
    # Forward pass
    loss = contrastive_loss(entity_spans, opinion_spans)
    
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    
    # Test backward pass
    loss.backward()
    print("  ✅ Backward pass successful")
    
    # Test empty input
    empty_entity = torch.randn(0, hidden_dim)
    empty_opinion = torch.randn(0, hidden_dim)
    empty_loss = contrastive_loss(empty_entity, empty_opinion)
    assert empty_loss.item() == 0.0, "Empty input should give zero loss"
    print("  ✅ Empty input handled correctly")
    
    print("✅ SimplifiedContrastiveLoss test passed!\n")


def test_contrastive_pair_encoder():
    print("Testing ContrastivePairEncoder...")
    
    hidden_dim = 768
    projection_dim = 256
    batch_size = 2
    num_entities = 5
    num_opinions = 5
    
    # Create dummy data
    entity_reprs = torch.randn(batch_size, num_entities, hidden_dim)
    opinion_reprs = torch.randn(batch_size, num_opinions, hidden_dim)
    
    # Positive pairs: [(entity_idx, opinion_idx), ...]
    positive_pairs = [
        torch.tensor([[0, 1], [2, 3]]),  # Batch 0: 2 triplets
        torch.tensor([[1, 2]])            # Batch 1: 1 triplet
    ]
    positive_pairs = torch.nn.utils.rnn.pad_sequence(
        positive_pairs, batch_first=True, padding_value=0
    )
    
    # Initialize module
    encoder = ContrastivePairEncoder(hidden_dim, projection_dim, temperature=0.07)
    
    # Forward pass
    loss = encoder(entity_reprs, opinion_reprs, positive_pairs)
    
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    
    # Test backward pass
    loss.backward()
    print("  ✅ Backward pass successful")
    
    print("✅ ContrastivePairEncoder test passed!\n")


def test_temperature_effect():
    print("Testing temperature effect...")
    
    hidden_dim = 768
    num_triplets = 10
    
    entity_spans = torch.randn(num_triplets, hidden_dim)
    opinion_spans = torch.randn(num_triplets, hidden_dim)
    
    temperatures = [0.05, 0.07, 0.1, 0.2]
    losses = []
    
    for temp in temperatures:
        model = SimplifiedContrastiveLoss(hidden_dim, temperature=temp)
        loss = model(entity_spans, opinion_spans)
        losses.append(loss.item())
        print(f"  Temperature {temp}: Loss = {loss.item():.4f}")
    
    print("  ✅ All temperatures work correctly")
    print("✅ Temperature test passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Contrastive Learning Module - Unit Tests")
    print("="*60 + "\n")
    
    try:
        test_simplified_contrastive_loss()
        test_contrastive_pair_encoder()
        test_temperature_effect()
        
        print("="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
