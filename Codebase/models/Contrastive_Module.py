import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedContrastiveLoss(nn.Module):
    """
    Simplified contrastive learning for entity-opinion pairing.
    Treats each triplet independently using InfoNCE loss.
    """
    
    def __init__(self, hidden_dim, temperature=0.07):
        super(SimplifiedContrastiveLoss, self).__init__()
        self.entity_proj = nn.Linear(hidden_dim, hidden_dim)
        self.opinion_proj = nn.Linear(hidden_dim, hidden_dim)
        self.temperature = temperature
    
    def forward(self, entity_spans, opinion_spans):
        """
        Args:
            entity_spans: [num_triplets, hidden_dim]
            opinion_spans: [num_triplets, hidden_dim]
        
        Returns:
            loss: scalar tensor
        """
        if len(entity_spans) == 0:
            return torch.tensor(0.0, device=entity_spans.device, requires_grad=True)
        
        # Project to contrastive space
        entity_emb = self.entity_proj(entity_spans)
        opinion_emb = self.opinion_proj(opinion_spans)
        
        # L2 normalize
        entity_emb = F.normalize(entity_emb, dim=-1)
        opinion_emb = F.normalize(opinion_emb, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(entity_emb, opinion_emb.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(len(entity_emb), device=entity_emb.device)
        
        # Symmetric InfoNCE loss (both directions)
        loss_entity = F.cross_entropy(similarity, labels)
        loss_opinion = F.cross_entropy(similarity.T, labels)
        
        return (loss_entity + loss_opinion) / 2


class ContrastivePairEncoder(nn.Module):
    """
    Full contrastive learning module with separate encoders.
    More expressive but slightly more complex.
    """
    
    def __init__(self, hidden_dim, projection_dim=256, temperature=0.07):
        super(ContrastivePairEncoder, self).__init__()
        
        # Separate encoders for entities and opinions
        self.entity_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.opinion_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.temperature = temperature
    
    def forward(self, entity_reprs, opinion_reprs, positive_pairs):
        """
        Args:
            entity_reprs: [batch_size, num_entities, hidden_dim]
            opinion_reprs: [batch_size, num_opinions, hidden_dim]
            positive_pairs: [batch_size, num_triplets, 2]
        
        Returns:
            contrastive_loss: scalar tensor
        """
        batch_size = entity_reprs.shape[0]
        device = entity_reprs.device
        
        total_loss = 0.0
        num_valid_samples = 0
        
        for b in range(batch_size):
            pos_pairs = positive_pairs[b]
            
            if len(pos_pairs) == 0:
                continue
            
            entities = entity_reprs[b]
            opinions = opinion_reprs[b]
            
            # Project and normalize
            entity_proj = F.normalize(self.entity_encoder(entities), dim=-1)
            opinion_proj = F.normalize(self.opinion_encoder(opinions), dim=-1)
            
            # Similarity matrix
            similarity = torch.matmul(entity_proj, opinion_proj.T) / self.temperature
            
            # InfoNCE loss for each positive pair
            for entity_idx, opinion_idx in pos_pairs:
                if entity_idx >= similarity.shape[0] or opinion_idx >= similarity.shape[1]:
                    continue
                
                pos_sim = similarity[entity_idx, opinion_idx]
                all_sims = similarity[entity_idx]
                
                log_prob = pos_sim - torch.logsumexp(all_sims, dim=0)
                total_loss += -log_prob
                num_valid_samples += 1
        
        if num_valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / num_valid_samples
