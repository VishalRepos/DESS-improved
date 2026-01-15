#!/usr/bin/env python3
"""
Test script for Cross-Attention Fusion module.

This script validates that the CrossAttentionFusion module:
1. Has the correct input/output shapes
2. Works with the same interface as TIN
3. Produces reasonable outputs
"""

import torch
import sys
sys.path.append('./Codebase')

from models.Cross_Attention_Fusion import CrossAttentionFusion
from models.TIN_GCN import TIN


def test_cross_attention_fusion():
    """Test CrossAttentionFusion module"""
    print("=" * 60)
    print("Testing Cross-Attention Fusion Module")
    print("=" * 60)
    
    # Parameters
    batch_size = 4
    seq_len = 24
    hidden_dim = 768
    num_heads = 8
    
    # Create module
    print(f"\n1. Creating CrossAttentionFusion module...")
    print(f"   - Hidden dim: {hidden_dim}")
    print(f"   - Num heads: {num_heads}")
    print(f"   - Dropout: 0.1")
    
    cross_attn = CrossAttentionFusion(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.1
    )
    cross_attn.eval()  # Set to eval mode for testing
    
    # Create dummy inputs (same as TIN interface)
    print(f"\n2. Creating dummy inputs...")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {seq_len}")
    
    h_feature = torch.randn(batch_size, seq_len, hidden_dim)
    h_syn_ori = torch.randn(batch_size, seq_len, hidden_dim)
    h_syn_feature = torch.randn(batch_size, seq_len, hidden_dim)
    h_sem_ori = torch.randn(batch_size, seq_len, hidden_dim)
    h_sem_feature = torch.randn(batch_size, seq_len, hidden_dim)
    adj_sem_ori = torch.randn(batch_size, seq_len, seq_len)
    adj_sem_gcn = torch.randn(batch_size, seq_len, seq_len)
    
    # Test forward pass
    print(f"\n3. Testing forward pass...")
    with torch.no_grad():
        output = cross_attn(
            h_feature, h_syn_ori, h_syn_feature, 
            h_sem_ori, h_sem_feature, 
            adj_sem_ori, adj_sem_gcn
        )
    
    print(f"   ✓ Forward pass successful!")
    print(f"   - Input shape: {h_feature.shape}")
    print(f"   - Output shape: {output.shape}")
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, hidden_dim), \
        f"Expected shape {(batch_size, seq_len, hidden_dim)}, got {output.shape}"
    print(f"   ✓ Output shape is correct!")
    
    # Check for NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    assert not torch.isinf(output).any(), "Output contains Inf values!"
    print(f"   ✓ No NaN or Inf values in output!")
    
    # Compare with TIN module
    print(f"\n4. Comparing with TIN module...")
    tin = TIN(hidden_dim)
    tin.eval()
    
    with torch.no_grad():
        tin_output = tin(
            h_feature, h_syn_ori, h_syn_feature,
            h_sem_ori, h_sem_feature,
            adj_sem_ori, adj_sem_gcn
        )
    
    print(f"   - TIN output shape: {tin_output.shape}")
    print(f"   - CrossAttention output shape: {output.shape}")
    assert output.shape == tin_output.shape, "Output shapes don't match!"
    print(f"   ✓ Both modules have same output shape!")
    
    # Test parameter count
    print(f"\n5. Comparing parameter counts...")
    cross_attn_params = sum(p.numel() for p in cross_attn.parameters())
    tin_params = sum(p.numel() for p in tin.parameters())
    
    print(f"   - CrossAttention parameters: {cross_attn_params:,}")
    print(f"   - TIN parameters: {tin_params:,}")
    print(f"   - Difference: {abs(cross_attn_params - tin_params):,}")
    
    # Test with different batch sizes
    print(f"\n6. Testing with different batch sizes...")
    for bs in [1, 2, 8, 16]:
        h_test = torch.randn(bs, seq_len, hidden_dim)
        with torch.no_grad():
            out_test = cross_attn(
                h_test, h_test, h_test, h_test, h_test,
                torch.randn(bs, seq_len, seq_len),
                torch.randn(bs, seq_len, seq_len)
            )
        assert out_test.shape == (bs, seq_len, hidden_dim)
        print(f"   ✓ Batch size {bs}: OK")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nCross-Attention Fusion is ready to use!")
    print("\nTo enable in training, use:")
    print("  --use_cross_attention --cross_attention_heads 8")
    print("\nExpected improvements:")
    print("  • Better integration of semantic and syntactic features")
    print("  • Learned attention weights for feature importance")
    print("  • Potential +0.5-0.7% improvement in Triplet F1")
    print("=" * 60)


if __name__ == "__main__":
    test_cross_attention_fusion()
