import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EnhancedSynGCN(nn.Module):
    """
    Enhanced Syntactic GCN with multiple graph convolution approaches:
    - GATv2 (Graph Attention Network v2)
    - GraphSAGE (Sample and Aggregate)
    - ChebNet (Chebyshev spectral)
    - Dynamic Edge Convolution
    - Hybrid fusion of all approaches
    """

    def __init__(self, emb_dim=768, num_layers=2, gcn_dropout=0.1, num_heads=4):
        super(EnhancedSynGCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        self.num_heads = num_heads
        
        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(GATv2Layer(emb_dim, emb_dim, num_heads, gcn_dropout))
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.sage_layers.append(SAGELayer(emb_dim, emb_dim, gcn_dropout))
        
        # Chebyshev layers
        self.cheb_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cheb_layers.append(ChebLayer(emb_dim, emb_dim, K=3))
        
        # Dynamic Edge Convolution
        self.edge_conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.edge_conv_layers.append(EdgeConvLayer(emb_dim, emb_dim, gcn_dropout))
        
        # Hybrid fusion
        self.fusion = HybridFusion(emb_dim, num_approaches=4)
        
        self.gcn_drop = nn.Dropout(gcn_dropout)

    def forward(self, adj, inputs):
        batch_size, seq_len, dim = inputs.size()
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        
        # Apply different GCN approaches
        outputs_list = []
        
        # 1. GATv2
        gat_out = inputs
        for layer in self.gat_layers:
            gat_out = layer(gat_out, adj)
        outputs_list.append(gat_out)
        
        # 2. GraphSAGE
        sage_out = inputs
        for layer in self.sage_layers:
            sage_out = layer(sage_out, adj)
        outputs_list.append(sage_out)
        
        # 3. Chebyshev
        cheb_out = inputs
        for layer in self.cheb_layers:
            cheb_out = layer(cheb_out, adj)
        outputs_list.append(cheb_out)
        
        # 4. Dynamic Edge Convolution
        edge_out = inputs
        for layer in self.edge_conv_layers:
            edge_out = layer(edge_out, adj)
        outputs_list.append(edge_out)
        
        # Hybrid fusion
        outputs = self.fusion(outputs_list)
        outputs = self.gcn_drop(outputs)
        
        return outputs, mask


class GATv2Layer(nn.Module):
    """Graph Attention Network v2 layer"""
    
    def __init__(self, in_dim, out_dim, num_heads, dropout):
        super(GATv2Layer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(1, num_heads, 2 * self.head_dim)))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, x, adj):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformation
        h = self.W(x)  # [batch, seq_len, out_dim]
        h = h.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Attention mechanism
        h_i = h.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)  # [batch, seq_len, seq_len, heads, head_dim]
        h_j = h.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)  # [batch, seq_len, seq_len, heads, head_dim]
        
        concat = torch.cat([h_i, h_j], dim=-1)  # [batch, seq_len, seq_len, heads, 2*head_dim]
        
        # Attention scores
        e = self.leakyrelu(torch.sum(concat * self.a, dim=-1))  # [batch, seq_len, seq_len, heads]
        
        # Mask with adjacency
        adj_expanded = adj.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
        e = e.masked_fill(adj_expanded == 0, -1e9)
        
        # Attention weights
        alpha = F.softmax(e, dim=2)  # [batch, seq_len, seq_len, heads]
        alpha = self.dropout(alpha)
        
        # Aggregate
        h_prime = torch.einsum('bijk,bjkd->bikd', alpha, h)  # [batch, seq_len, heads, head_dim]
        h_prime = h_prime.reshape(batch_size, seq_len, -1)  # [batch, seq_len, out_dim]
        
        return F.elu(h_prime)


class SAGELayer(nn.Module):
    """GraphSAGE layer with mean aggregation"""
    
    def __init__(self, in_dim, out_dim, dropout):
        super(SAGELayer, self).__init__()
        self.W = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        # Normalize adjacency
        denom = adj.sum(2, keepdim=True) + 1e-6
        adj_norm = adj / denom
        
        # Aggregate neighbors (mean)
        neigh_feat = torch.bmm(adj_norm, x)  # [batch, seq_len, dim]
        
        # Concatenate self and neighbor features
        combined = torch.cat([x, neigh_feat], dim=-1)  # [batch, seq_len, 2*dim]
        
        # Transform
        out = self.W(combined)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out


class ChebLayer(nn.Module):
    """Chebyshev spectral graph convolution"""
    
    def __init__(self, in_dim, out_dim, K=3):
        super(ChebLayer, self).__init__()
        self.K = K
        self.W = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(K)])
    
    def forward(self, x, adj):
        # Normalize adjacency for Chebyshev
        batch_size, seq_len, _ = x.size()
        
        # Compute normalized Laplacian
        d = adj.sum(2)  # [batch, seq_len]
        d_inv_sqrt = torch.pow(d + 1e-6, -0.5)
        d_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # [batch, seq_len, seq_len]
        
        L_norm = torch.eye(seq_len).unsqueeze(0).to(x.device) - torch.bmm(torch.bmm(d_inv_sqrt, adj), d_inv_sqrt)
        
        # Chebyshev polynomials
        Tx_0 = x
        Tx_1 = torch.bmm(L_norm, x)
        
        out = self.W[0](Tx_0)
        if self.K > 1:
            out = out + self.W[1](Tx_1)
        
        for k in range(2, self.K):
            Tx_2 = 2 * torch.bmm(L_norm, Tx_1) - Tx_0
            out = out + self.W[k](Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2
        
        return F.relu(out)


class EdgeConvLayer(nn.Module):
    """Dynamic Edge Convolution"""
    
    def __init__(self, in_dim, out_dim, dropout):
        super(EdgeConvLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, adj):
        batch_size, seq_len, dim = x.size()
        
        # Compute edge features
        x_i = x.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, seq_len, seq_len, dim]
        x_j = x.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch, seq_len, seq_len, dim]
        
        edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)  # [batch, seq_len, seq_len, 2*dim]
        
        # Apply MLP
        edge_out = self.mlp(edge_feat)  # [batch, seq_len, seq_len, out_dim]
        
        # Aggregate with adjacency
        adj_expanded = adj.unsqueeze(-1)  # [batch, seq_len, seq_len, 1]
        edge_out = edge_out * adj_expanded
        
        # Max pooling over neighbors
        out = edge_out.max(dim=2)[0]  # [batch, seq_len, out_dim]
        
        return out


class HybridFusion(nn.Module):
    """Hybrid fusion of multiple GCN approaches"""
    
    def __init__(self, dim, num_approaches=4):
        super(HybridFusion, self).__init__()
        self.attention = nn.Linear(num_approaches, num_approaches)
        self.fusion = nn.Linear(dim, dim)
    
    def forward(self, outputs_list):
        # outputs_list: list of [batch, seq_len, dim]
        stacked = torch.stack(outputs_list, dim=2)  # [batch, seq_len, num_approaches, dim]
        
        # Attention-based fusion
        # Transpose to [batch, seq_len, dim, num_approaches] for attention
        stacked_t = stacked.transpose(2, 3)  # [batch, seq_len, dim, num_approaches]
        attn_scores = self.attention(stacked_t)  # [batch, seq_len, dim, num_approaches]
        attn_scores = attn_scores.transpose(2, 3)  # [batch, seq_len, num_approaches, dim]
        attn_weights = F.softmax(attn_scores.mean(dim=-1, keepdim=True), dim=2)  # [batch, seq_len, num_approaches, 1]
        
        # Weighted combination
        fused = (stacked * attn_weights).sum(dim=2)  # [batch, seq_len, dim]
        
        # Final transformation
        out = self.fusion(fused)
        
        return out


# Keep original GCN for backward compatibility
class GCN(nn.Module):

    def __init__(self, emb_dim=768, num_layers=2, gcn_dropout=0.1):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)

    def forward(self, adj, inputs):
        # gcn layer

        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.layers):
            Ax = adj.bmm(inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](inputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return inputs, mask
