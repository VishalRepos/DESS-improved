import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math


class EnhancedSemGCN(nn.Module):
    """
    Enhanced Semantic GCN with:
    - Multi-head attention with relative position encoding
    - Global context aggregation
    - Multi-scale feature extraction
    """

    def __init__(self, args, emb_dim=768, num_layers=2, gcn_dropout=0.1):
        super(EnhancedSemGCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        self.attention_heads = self.args.attention_heads
        self.mem_dim = self.args.hidden_dim
        
        # GCN layers with multi-scale
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        
        self.gcn_drop = nn.Dropout(gcn_dropout)
        
        # Enhanced multi-head attention with relative position
        self.attn = EnhancedMultiHeadAttention(
            self.attention_heads, 
            self.mem_dim * 2,
            max_relative_position=128
        )
        
        # Global context aggregation
        self.global_context = GlobalContextAggregation(self.emb_dim)
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtraction(self.emb_dim)

    def forward(self, inputs, encoding, seq_lens):
        tok = encoding
        src_mask = (tok != 0).unsqueeze(-2)
        maxlen = seq_lens
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]

        gcn_inputs = inputs
        
        # Enhanced attention with relative position
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [
            attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)
        ]
        
        # Aggregate attention heads
        adj_ag = None
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag_new = adj_ag.clone()
        adj_ag_new /= self.attention_heads

        # Add self-loops
        for j in range(adj_ag_new.size(0)):
            adj_ag_new[j] -= torch.diag(torch.diag(adj_ag_new[j]))
            adj_ag_new[j] += torch.eye(adj_ag_new[j].size(0)).to(adj_ag_new.device)
        adj_ag_new = mask_ * adj_ag_new

        # GCN layers with multi-scale features
        denom_ag = adj_ag_new.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs
        
        multi_scale_features = []
        for l in range(self.layers):
            Ax = adj_ag_new.bmm(outputs)
            AxW = self.W[l](Ax)
            AxW = AxW / denom_ag
            gAxW = F.relu(AxW)
            outputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
            multi_scale_features.append(outputs)
        
        # Apply multi-scale feature extraction
        outputs = self.multi_scale(multi_scale_features)
        
        # Apply global context aggregation
        outputs = self.global_context(outputs, mask_)
        
        return outputs, adj_ag_new


class EnhancedMultiHeadAttention(nn.Module):
    """Multi-head attention with relative position encoding"""
    
    def __init__(self, h, d_model, dropout=0.1, max_relative_position=128):
        super(EnhancedMultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        
        # Relative position encoding
        self.max_relative_position = max_relative_position
        self.relative_position_k = nn.Embedding(2 * max_relative_position + 1, self.d_k)
        self.relative_position_v = nn.Embedding(2 * max_relative_position + 1, self.d_k)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, : query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        seq_len = query.size(1)
        
        query, key = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]

        # Compute relative position encoding
        relative_positions = self._get_relative_positions(seq_len).to(query.device)
        relative_positions_k = self.relative_position_k(relative_positions)
        relative_positions_v = self.relative_position_v(relative_positions)
        
        # Attention with relative position
        attn = self._attention_with_relative_position(
            query, key, relative_positions_k, mask, self.dropout
        )

        return attn
    
    def _get_relative_positions(self, length):
        """Generate relative position matrix"""
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(0).expand(length, length)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat
    
    def _attention_with_relative_position(self, query, key, relative_pos_k, mask, dropout):
        """Compute attention with relative position encoding"""
        d_k = query.size(-1)
        
        # Standard attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Add relative position bias
        # Simplified: average across heads for relative position
        rel_scores = torch.matmul(
            query.mean(dim=1, keepdim=True), 
            relative_pos_k.transpose(-2, -1)
        ) / math.sqrt(d_k)
        scores = scores + rel_scores.expand_as(scores)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return p_attn


class GlobalContextAggregation(nn.Module):
    """Aggregate global context information"""
    
    def __init__(self, dim):
        super(GlobalContextAggregation, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim, dim)
        
    def forward(self, x, mask):
        # x: [batch, seq_len, dim]
        batch_size, seq_len, dim = x.size()
        
        # Global average and max pooling
        x_transposed = x.transpose(1, 2)  # [batch, dim, seq_len]
        global_avg = self.global_avg_pool(x_transposed).squeeze(-1)  # [batch, dim]
        global_max = self.global_max_pool(x_transposed).squeeze(-1)  # [batch, dim]
        
        # Combine global features
        global_context = torch.cat([global_avg, global_max], dim=-1)  # [batch, dim*2]
        global_context = self.fc(global_context)  # [batch, dim]
        
        # Gating mechanism
        gate_values = torch.sigmoid(self.gate(global_context))  # [batch, dim]
        gate_values = gate_values.unsqueeze(1).expand_as(x)  # [batch, seq_len, dim]
        
        # Apply gated global context
        output = x + gate_values * global_context.unsqueeze(1).expand_as(x)
        
        return output


class MultiScaleFeatureExtraction(nn.Module):
    """Extract and combine multi-scale features"""
    
    def __init__(self, dim):
        super(MultiScaleFeatureExtraction, self).__init__()
        self.scale_weights = nn.Parameter(torch.ones(2) / 2)  # For 2 layers
        self.fusion = nn.Linear(dim, dim)
        
    def forward(self, feature_list):
        # feature_list: list of [batch, seq_len, dim] from different GCN layers
        if len(feature_list) == 1:
            return feature_list[0]
        
        # Weighted combination of multi-scale features
        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * f for w, f in zip(weights, feature_list))
        
        # Fusion layer
        output = self.fusion(combined)
        
        return output


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Keep original SemGCN for backward compatibility
class SemGCN(nn.Module):

    def __init__(self, args, emb_dim=768, num_layers=2, gcn_dropout=0.1):
        super(SemGCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        self.attention_heads = self.args.attention_heads
        self.mem_dim = self.args.hidden_dim
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)
        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim * 2)

    def forward(self, inputs, encoding, seq_lens):
        tok = encoding
        src_mask = (tok != 0).unsqueeze(-2)
        maxlen = seq_lens
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]

        gcn_inputs = inputs
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [
            attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)
        ]
        adj_ag = None

        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag_new = adj_ag.clone()
        adj_ag_new /= self.attention_heads

        for j in range(adj_ag_new.size(0)):
            adj_ag_new[j] -= torch.diag(torch.diag(adj_ag_new[j]))
            adj_ag_new[j] += torch.eye(adj_ag_new[j].size(0)).cuda()
        adj_ag_new = mask_ * adj_ag_new

        # gcn layer
        denom_ag = adj_ag_new.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs

        for l in range(self.layers):
            Ax = adj_ag_new.bmm(outputs)
            AxW = self.W[l](Ax)
            AxW = AxW / denom_ag
            gAxW = F.relu(AxW)
            outputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return outputs, adj_ag_new


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, : query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]

        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn
