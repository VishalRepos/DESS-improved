#!/usr/bin/env python3
"""
Unit test for Boundary Refinement module
"""
import sys
sys.path.append('./Codebase')

import torch
from models.Boundary_Refinement import SimplifiedBoundaryRefinement, BoundaryRefinement


def test_simplified_boundary_refinement():
    print("Testing SimplifiedBoundaryRefinement...")
    
    batch_size = 2
    num_spans = 10
    span_len = 5
    hidden_dim = 768
    
    # Create dummy data
    span_features = torch.randn(batch_size, num_spans, span_len, hidden_dim)
    span_masks = torch.ones(batch_size, num_spans, span_len)
    # Add some padding
    span_masks[:, :, 3:] = 0  # Last 2 tokens are padding
    
    # Initialize module
    refiner = SimplifiedBoundaryRefinement(hidden_dim, dropout=0.1)
    
    # Forward pass
    refined = refiner(span_features, span_masks)
    
    print(f"  Input shape: {span_features.shape}")
    print(f"  Output shape: {refined.shape}")
    print(f"  Expected: [{batch_size}, {num_spans}, {hidden_dim}]")
    
    assert refined.shape == (batch_size, num_spans, hidden_dim), "Output shape mismatch"
    assert not torch.isnan(refined).any(), "Output contains NaN"
    assert not torch.isinf(refined).any(), "Output contains Inf"
    
    # Test backward pass
    loss = refined.sum()
    loss.backward()
    print("  ✅ Backward pass successful")
    
    # Test without mask
    refined_no_mask = refiner(span_features, None)
    assert refined_no_mask.shape == (batch_size, num_spans, hidden_dim)
    print("  ✅ Works without mask")
    
    print("✅ SimplifiedBoundaryRefinement test passed!\n")


def test_boundary_refinement():
    print("Testing BoundaryRefinement (with LSTM)...")
    
    batch_size = 2
    num_spans = 10
    span_len = 5
    hidden_dim = 768
    
    # Create dummy data
    span_features = torch.randn(batch_size, num_spans, span_len, hidden_dim)
    span_masks = torch.ones(batch_size, num_spans, span_len)
    span_masks[:, :, 3:] = 0
    
    # Initialize module
    refiner = BoundaryRefinement(hidden_dim, dropout=0.1)
    
    # Forward pass
    refined = refiner(span_features, span_masks)
    
    print(f"  Input shape: {span_features.shape}")
    print(f"  Output shape: {refined.shape}")
    
    assert refined.shape == (batch_size, num_spans, hidden_dim), "Output shape mismatch"
    assert not torch.isnan(refined).any(), "Output contains NaN"
    assert not torch.isinf(refined).any(), "Output contains Inf"
    
    # Test backward pass
    loss = refined.sum()
    loss.backward()
    print("  ✅ Backward pass successful")
    
    print("✅ BoundaryRefinement test passed!\n")


def test_attention_weights():
    print("Testing attention weight distribution...")
    
    batch_size = 1
    num_spans = 1
    span_len = 5
    hidden_dim = 768
    
    # Create span with distinct start and end
    span_features = torch.randn(batch_size, num_spans, span_len, hidden_dim)
    span_masks = torch.ones(batch_size, num_spans, span_len)
    
    refiner = SimplifiedBoundaryRefinement(hidden_dim, dropout=0.0)
    
    # Get attention scores
    start_scores = refiner.start_attention(span_features).squeeze(-1)
    end_scores = refiner.end_attention(span_features).squeeze(-1)
    
    start_weights = torch.softmax(start_scores, dim=-1)
    end_weights = torch.softmax(end_scores, dim=-1)
    
    print(f"  Start attention weights: {start_weights[0, 0].detach().numpy()}")
    print(f"  End attention weights: {end_weights[0, 0].detach().numpy()}")
    print(f"  Sum of start weights: {start_weights.sum().item():.4f}")
    print(f"  Sum of end weights: {end_weights.sum().item():.4f}")
    
    assert abs(start_weights.sum().item() - 1.0) < 1e-5, "Start weights don't sum to 1"
    assert abs(end_weights.sum().item() - 1.0) < 1e-5, "End weights don't sum to 1"
    
    print("  ✅ Attention weights are valid")
    print("✅ Attention weight test passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Boundary Refinement Module - Unit Tests")
    print("="*60 + "\n")
    
    try:
        test_simplified_boundary_refinement()
        test_boundary_refinement()
        test_attention_weights()
        
        print("="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
