import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedBoundaryRefinement(nn.Module):
    """
    Simplified boundary-aware attention for span refinement.
    Focuses on start and end tokens without LSTM overhead.
    """
    
    def __init__(self, hidden_dim, dropout=0.1):
        super(SimplifiedBoundaryRefinement, self).__init__()
        
        self.start_attention = nn.Linear(hidden_dim, 1)
        self.end_attention = nn.Linear(hidden_dim, 1)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, span_features, span_masks):
        """
        Args:
            span_features: [batch, num_spans, span_len, hidden_dim]
            span_masks: [batch, num_spans, span_len] - 1 for valid, 0 for padding
        
        Returns:
            refined_spans: [batch, num_spans, hidden_dim]
        """
        # Compute attention scores
        start_scores = self.start_attention(span_features).squeeze(-1)
        end_scores = self.end_attention(span_features).squeeze(-1)
        
        # Apply mask
        if span_masks is not None:
            mask_value = -1e9
            start_scores = start_scores.masked_fill(span_masks == 0, mask_value)
            end_scores = end_scores.masked_fill(span_masks == 0, mask_value)
        
        # Softmax
        start_weights = F.softmax(start_scores, dim=-1).unsqueeze(-1)
        end_weights = F.softmax(end_scores, dim=-1).unsqueeze(-1)
        
        # Weighted sum
        start_repr = torch.sum(start_weights * span_features, dim=2)
        end_repr = torch.sum(end_weights * span_features, dim=2)
        
        # Fuse
        boundary_repr = torch.cat([start_repr, end_repr], dim=-1)
        refined = self.fusion(boundary_repr)
        refined = self.dropout(refined)
        
        # Residual
        residual = span_features.mean(dim=2)
        refined = refined + residual
        
        return refined


class BoundaryRefinement(nn.Module):
    """
    Full boundary-aware attention with BiLSTM for better context modeling.
    """
    
    def __init__(self, hidden_dim, dropout=0.1):
        super(BoundaryRefinement, self).__init__()
        
        # Bidirectional LSTM for boundary context
        self.boundary_lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2, 
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Separate attention for start and end
        self.start_attention = nn.Linear(hidden_dim, 1)
        self.end_attention = nn.Linear(hidden_dim, 1)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, span_features, span_masks):
        """
        Args:
            span_features: [batch, num_spans, span_len, hidden_dim]
            span_masks: [batch, num_spans, span_len] - 1 for valid, 0 for padding
        
        Returns:
            refined_spans: [batch, num_spans, hidden_dim]
        """
        batch_size, num_spans, span_len, hidden_dim = span_features.shape
        
        # Reshape for LSTM: [batch * num_spans, span_len, hidden_dim]
        span_flat = span_features.view(-1, span_len, hidden_dim)
        
        # Apply boundary-aware LSTM
        lstm_out, _ = self.boundary_lstm(span_flat)
        
        # Reshape back
        lstm_out = lstm_out.view(batch_size, num_spans, span_len, hidden_dim)
        
        # Compute start and end attention
        start_scores = self.start_attention(lstm_out).squeeze(-1)
        end_scores = self.end_attention(lstm_out).squeeze(-1)
        
        # Apply mask
        if span_masks is not None:
            mask_value = -1e9
            start_scores = start_scores.masked_fill(span_masks == 0, mask_value)
            end_scores = end_scores.masked_fill(span_masks == 0, mask_value)
        
        # Softmax
        start_weights = F.softmax(start_scores, dim=-1)
        end_weights = F.softmax(end_scores, dim=-1)
        
        # Weighted sum
        start_repr = torch.sum(start_weights.unsqueeze(-1) * lstm_out, dim=2)
        end_repr = torch.sum(end_weights.unsqueeze(-1) * lstm_out, dim=2)
        
        # Concatenate and fuse
        boundary_repr = torch.cat([start_repr, end_repr], dim=-1)
        refined = self.fusion(boundary_repr)
        refined = self.dropout(refined)
        
        # Layer norm with residual
        residual = lstm_out.mean(dim=2)
        refined = self.layer_norm(refined + residual)
        
        return refined
