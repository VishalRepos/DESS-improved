import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCrossAttentionFusion(nn.Module):
    """
    Lightweight Cross-Attention Fusion - Faster version with single attention pass.
    
    Uses only one cross-attention direction (semantic queries syntactic) 
    for better speed while maintaining performance.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(LightweightCrossAttentionFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Single cross-attention: semantic queries syntactic
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Simplified fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, h_feature, h_syn_ori, h_syn_feature, h_sem_ori, h_sem_feature, adj_sem_ori, adj_sem_gcn):
        """
        Args:
            h_feature: Original features (batch_size, seq_len, hidden_dim)
            h_syn_feature: Syntactic GCN output (batch_size, seq_len, hidden_dim)
            h_sem_feature: Semantic GCN output (batch_size, seq_len, hidden_dim)
            
        Returns:
            h_fusion: Fused features (batch_size, seq_len, hidden_dim)
        """
        # Add residual connections
        h_syn = self.norm1(h_feature + h_syn_feature)
        h_sem = self.norm2(h_feature + h_sem_feature)
        
        # Single cross-attention: semantic queries syntactic
        attended, _ = self.cross_attn(
            query=h_sem,
            key=h_syn,
            value=h_syn,
            need_weights=False
        )
        
        # Residual
        attended = h_sem + attended
        
        # Concatenate syntactic and attended semantic
        concat = torch.cat([h_syn, attended], dim=-1)
        h_fusion = self.fusion(concat)
        
        return h_fusion
