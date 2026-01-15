import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module to replace TIN concatenation.
    
    Instead of simple concatenation, this module uses multi-head cross-attention
    to let semantic and syntactic features query each other, learning which
    features are important for each context.
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head cross-attention: semantic queries syntactic
        self.sem_to_syn_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-head cross-attention: syntactic queries semantic
        self.syn_to_sem_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Residual connections with layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, h_feature, h_syn_ori, h_syn_feature, h_sem_ori, h_sem_feature, adj_sem_ori, adj_sem_gcn):
        """
        Args:
            h_feature: Original features (batch_size, seq_len, hidden_dim)
            h_syn_ori: Original syntactic features
            h_syn_feature: Syntactic GCN output (batch_size, seq_len, hidden_dim)
            h_sem_ori: Original semantic features
            h_sem_feature: Semantic GCN output (batch_size, seq_len, hidden_dim)
            adj_sem_ori: Original semantic adjacency matrix
            adj_sem_gcn: GCN semantic adjacency matrix
            
        Returns:
            h_fusion: Fused features (batch_size, seq_len, hidden_dim)
        """
        # Add residual connections to GCN outputs
        h_syn = self.norm1(h_feature + h_syn_feature)
        h_sem = self.norm2(h_feature + h_sem_feature)
        
        # Cross-attention: semantic queries syntactic (need_weights=False for speed)
        sem_attended, _ = self.sem_to_syn_attn(
            query=h_sem,
            key=h_syn,
            value=h_syn,
            need_weights=False
        )
        
        # Cross-attention: syntactic queries semantic (need_weights=False for speed)
        syn_attended, _ = self.syn_to_sem_attn(
            query=h_syn,
            key=h_sem,
            value=h_sem,
            need_weights=False
        )
        
        # Residual connections for attended features
        sem_attended = h_sem + sem_attended
        syn_attended = h_syn + syn_attended
        
        # Concatenate and fuse
        concat = torch.cat([sem_attended, syn_attended], dim=-1)
        h_fusion = self.fusion(concat)
        
        return h_fusion
