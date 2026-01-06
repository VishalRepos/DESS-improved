# 🎯 Contrastive Learning Implementation Plan

**Goal**: Improve entity-opinion pairing by learning better representations through contrastive learning  
**Expected Gain**: +0.5-0.8% Triplet F1  
**Timeline**: 3-4 days  
**Date**: January 6, 2026

---

## 📊 Understanding the Problem

### Current Architecture Flow:
```
Input: "But the staff was so horrible to us."
       
Entities: [staff (target), horrible (opinion)]
Sentiment: [(staff, horrible, NEGATIVE)]

Current Process:
1. Extract all entity spans → [staff, horrible, ...]
2. Create all possible pairs → [(staff, horrible), (staff, staff), ...]
3. Classify each pair independently → sentiment logits

Problem: No explicit learning that (staff, horrible) should be close in embedding space
         while (staff, random_opinion) should be far apart
```

### What Contrastive Learning Will Do:
```
Positive Pairs (from ground truth triplets):
  - (staff, horrible) → Pull together in embedding space
  
Negative Pairs (all other combinations):
  - (staff, other_opinions) → Push apart
  - (other_targets, horrible) → Push apart
  - (target, target) → Push apart
  - (opinion, opinion) → Push apart
```

---

## 🏗️ Architecture Design

### New Component: ContrastivePairEncoder

```
Entity Span → Entity Encoder → Normalized Entity Embedding
                                        ↓
                                  Similarity Matrix
                                        ↓
Opinion Span → Opinion Encoder → Normalized Opinion Embedding
                                        ↓
                                Contrastive Loss
```

### Key Design Decisions:

1. **Separate Encoders**: Entity and opinion get different projections
2. **L2 Normalization**: Embeddings normalized to unit sphere
3. **Temperature Scaling**: Controls hardness of positive/negative separation
4. **InfoNCE Loss**: Standard contrastive loss (used in SimCLR, MoCo)

---

## 📝 Implementation Steps

### **Step 1: Create ContrastivePairEncoder Module** (1 day)

**File**: `models/Contrastive_Module.py` (NEW)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastivePairEncoder(nn.Module):
    """
    Contrastive learning module for entity-opinion pairing.
    Uses InfoNCE loss to pull positive pairs together and push negatives apart.
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
            positive_pairs: [batch_size, num_triplets, 2] - indices of (entity, opinion) pairs
        
        Returns:
            contrastive_loss: scalar tensor
        """
        batch_size = entity_reprs.shape[0]
        device = entity_reprs.device
        
        total_loss = 0.0
        num_valid_samples = 0
        
        for b in range(batch_size):
            # Get positive pairs for this sample
            pos_pairs = positive_pairs[b]  # [num_triplets, 2]
            
            if len(pos_pairs) == 0:
                continue
            
            # Extract entity and opinion representations
            entities = entity_reprs[b]  # [num_entities, hidden_dim]
            opinions = opinion_reprs[b]  # [num_opinions, hidden_dim]
            
            # Project to contrastive space
            entity_proj = self.entity_encoder(entities)  # [num_entities, proj_dim]
            opinion_proj = self.opinion_encoder(opinions)  # [num_opinions, proj_dim]
            
            # L2 normalize
            entity_proj = F.normalize(entity_proj, dim=-1)
            opinion_proj = F.normalize(opinion_proj, dim=-1)
            
            # Compute similarity matrix
            similarity = torch.matmul(entity_proj, opinion_proj.T) / self.temperature
            # [num_entities, num_opinions]
            
            # Create positive pair mask
            pos_mask = torch.zeros_like(similarity, dtype=torch.bool)
            for entity_idx, opinion_idx in pos_pairs:
                if entity_idx < similarity.shape[0] and opinion_idx < similarity.shape[1]:
                    pos_mask[entity_idx, opinion_idx] = True
            
            # InfoNCE loss: for each positive pair, maximize similarity vs all negatives
            for entity_idx, opinion_idx in pos_pairs:
                if entity_idx >= similarity.shape[0] or opinion_idx >= similarity.shape[1]:
                    continue
                
                # Positive similarity
                pos_sim = similarity[entity_idx, opinion_idx]
                
                # All similarities for this entity (positives + negatives)
                all_sims = similarity[entity_idx]  # [num_opinions]
                
                # Compute log-softmax (numerically stable)
                log_prob = pos_sim - torch.logsumexp(all_sims, dim=0)
                
                # Negative log-likelihood
                total_loss += -log_prob
                num_valid_samples += 1
        
        if num_valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / num_valid_samples


class SimplifiedContrastiveLoss(nn.Module):
    """
    Simplified version: treats each triplet independently
    Easier to implement and debug
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
            loss: scalar
        """
        # Project
        entity_emb = F.normalize(self.entity_proj(entity_spans), dim=-1)
        opinion_emb = F.normalize(self.opinion_proj(opinion_spans), dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(entity_emb, opinion_emb.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(len(entity_emb), device=entity_emb.device)
        
        # Symmetric loss (both directions)
        loss_entity = F.cross_entropy(similarity, labels)
        loss_opinion = F.cross_entropy(similarity.T, labels)
        
        return (loss_entity + loss_opinion) / 2
```

---

### **Step 2: Modify D2E2S_Model.py** (1 day)

**File**: `models/D2E2S_Model.py`

#### 2.1 Add Import
```python
from models.Contrastive_Module import ContrastivePairEncoder, SimplifiedContrastiveLoss
```

#### 2.2 Add to __init__ (around line 100)
```python
# Add after self.entity_classifier and self.senti_classifier
self.contrastive_encoder = SimplifiedContrastiveLoss(
    hidden_dim=self._emb_dim,
    temperature=0.07
)
self.use_contrastive = self.args.use_contrastive  # New parameter
self.contrastive_weight = self.args.contrastive_weight  # Default: 0.1
```

#### 2.3 Modify _forward_train (around line 200)

**Current code** (around line 230):
```python
# entity_classify
size_embeddings = self.size_embeddings(entity_sizes)
entity_clf, entity_spans_pool = self._classify_entities(
    encodings, h, entity_masks, size_embeddings, self.args
)
```

**Add after entity classification**:
```python
# Contrastive learning for entity-opinion pairing
contrastive_loss = torch.tensor(0.0, device=h.device)

if self.use_contrastive and self.training:
    # Extract ground truth triplets
    contrastive_loss = self._compute_contrastive_loss(
        entity_spans_pool, entity_masks, sentiments, h
    )
```

**Modify return statement** (around line 250):
```python
# OLD:
return entity_clf, senti_clf, batch_loss

# NEW:
return entity_clf, senti_clf, batch_loss, contrastive_loss
```

#### 2.4 Add New Method: _compute_contrastive_loss

**Add this method to D2E2S_Model class** (around line 450):
```python
def _compute_contrastive_loss(self, entity_spans_pool, entity_masks, sentiments, h):
    """
    Compute contrastive loss for entity-opinion pairs.
    
    Args:
        entity_spans_pool: [batch_size, num_entities, hidden_dim]
        entity_masks: [batch_size, num_entities, seq_len]
        sentiments: [batch_size, num_pairs, 2] - ground truth pairs
        h: [batch_size, seq_len, hidden_dim]
    
    Returns:
        contrastive_loss: scalar tensor
    """
    batch_size = sentiments.shape[0]
    device = h.device
    
    # Collect all positive entity-opinion pairs across batch
    all_entity_reprs = []
    all_opinion_reprs = []
    
    for b in range(batch_size):
        # Get ground truth pairs for this sample
        pairs = sentiments[b]  # [num_pairs, 2]
        
        for entity_idx, opinion_idx in pairs:
            entity_idx = entity_idx.item()
            opinion_idx = opinion_idx.item()
            
            # Skip invalid pairs
            if entity_idx == 0 and opinion_idx == 0:
                continue
            
            # Get entity and opinion representations
            entity_repr = entity_spans_pool[b, entity_idx]  # [hidden_dim]
            opinion_repr = entity_spans_pool[b, opinion_idx]  # [hidden_dim]
            
            all_entity_reprs.append(entity_repr)
            all_opinion_reprs.append(opinion_repr)
    
    if len(all_entity_reprs) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Stack into tensors
    entity_batch = torch.stack(all_entity_reprs)  # [num_triplets, hidden_dim]
    opinion_batch = torch.stack(all_opinion_reprs)  # [num_triplets, hidden_dim]
    
    # Compute contrastive loss
    loss = self.contrastive_encoder(entity_batch, opinion_batch)
    
    return loss
```

---

### **Step 3: Modify train.py** (0.5 day)

**File**: `train.py`

#### 3.1 Update loss computation (around line 150-200)

**Find this section**:
```python
# forward step
entity_logits, senti_logits, batch_loss = model(
    encodings=encodings,
    context_masks=context_masks,
    entity_masks=entity_masks,
    entity_sizes=entity_sizes,
    sentiments=sentiments,
    senti_masks=senti_masks,
    adj=adj,
)
```

**Change to**:
```python
# forward step
entity_logits, senti_logits, batch_loss, contrastive_loss = model(
    encodings=encodings,
    context_masks=context_masks,
    entity_masks=entity_masks,
    entity_sizes=entity_sizes,
    sentiments=sentiments,
    senti_masks=senti_masks,
    adj=adj,
)
```

**Find loss computation** (around line 180):
```python
# compute loss and optimize parameters
batch_loss = compute_loss.compute(
    entity_logits=entity_logits,
    senti_logits=senti_logits,
    senti_types=senti_types,
    entity_types=entity_types,
    entity_sample_masks=entity_sample_masks,
    senti_sample_masks=senti_sample_masks,
)
```

**Add contrastive loss**:
```python
# Add contrastive loss
if args.use_contrastive:
    total_loss = batch_loss + args.contrastive_weight * contrastive_loss
else:
    total_loss = batch_loss

# Backward
total_loss.backward()
optimizer.step()
scheduler.step()
model.zero_grad()
```

---

### **Step 4: Add Parameters** (0.5 day)

**File**: `Parameter.py`

**Add these parameters** (around line 50):
```python
parser.add_argument(
    "--use_contrastive",
    action="store_true",
    default=False,
    help="Use contrastive learning for entity-opinion pairing"
)
parser.add_argument(
    "--contrastive_weight",
    type=float,
    default=0.1,
    help="Weight for contrastive loss (default: 0.1)"
)
```

---

### **Step 5: Update Evaluation** (0.5 day)

**File**: `models/D2E2S_Model.py`

**Modify _forward_eval** to handle new return signature:

```python
def _forward_eval(self, ...):
    # ... existing code ...
    
    # At the end, return without contrastive loss
    return entity_clf, senti_clf, sentiments
```

**Note**: Contrastive loss is only used during training, not evaluation.

---

### **Step 6: Testing & Validation** (1 day)

#### 6.1 Unit Test the Module

**Create**: `test_contrastive.py`

```python
import torch
from models.Contrastive_Module import SimplifiedContrastiveLoss

# Test
hidden_dim = 768
num_triplets = 10

entity_spans = torch.randn(num_triplets, hidden_dim)
opinion_spans = torch.randn(num_triplets, hidden_dim)

contrastive_loss = SimplifiedContrastiveLoss(hidden_dim)
loss = contrastive_loss(entity_spans, opinion_spans)

print(f"Contrastive Loss: {loss.item()}")
assert loss.item() > 0, "Loss should be positive"
print("✅ Test passed!")
```

#### 6.2 Small-Scale Training Test

```bash
# Test with 1 epoch to ensure no errors
python train.py --dataset 14res --epochs 1 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn \
    --use_contrastive --contrastive_weight 0.1
```

#### 6.3 Full Training

```bash
# Full training with contrastive learning
python train.py --dataset 14res --epochs 120 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn \
    --use_contrastive --contrastive_weight 0.1
```

---

## 🔍 Debugging Checklist

### Common Issues:

1. **Shape Mismatch**
   - Check entity_spans_pool dimensions
   - Verify sentiments tensor format
   - Print shapes at each step

2. **Empty Batches**
   - Handle cases where no valid triplets exist
   - Return zero loss with requires_grad=True

3. **NaN Loss**
   - Check for division by zero
   - Verify temperature > 0
   - Use torch.logsumexp for numerical stability

4. **Memory Issues**
   - Process in chunks if batch too large
   - Use gradient checkpointing if needed

---

## 📊 Expected Results

### Baseline (Enhanced SemGCN):
- Entity F1: 88.68%
- Triplet F1: 77.14%

### With Contrastive Learning:
- Entity F1: 88.5-89.0% (may vary slightly)
- Triplet F1: **77.6-78.0%** (+0.5-0.8%)

### Training Observations:
- Contrastive loss should decrease over epochs
- Initial loss: ~2-4
- Final loss: ~0.5-1.5
- Total training time: +5-10% (minimal overhead)

---

## 🎯 Hyperparameter Tuning

If initial results are not satisfactory, try:

### Temperature (τ):
- **Lower (0.05)**: Harder negatives, more aggressive separation
- **Higher (0.1)**: Softer negatives, gentler separation
- **Default**: 0.07

### Contrastive Weight (λ):
- **Lower (0.05)**: Less influence, safer
- **Higher (0.2)**: More influence, may hurt if not tuned
- **Default**: 0.1

### Projection Dimension:
- **Smaller (128)**: Faster, less expressive
- **Larger (512)**: Slower, more expressive
- **Default**: 256

---

## 📝 Code Changes Summary

### New Files:
1. `models/Contrastive_Module.py` - Contrastive learning module

### Modified Files:
1. `models/D2E2S_Model.py` - Add contrastive encoder and loss computation
2. `train.py` - Update loss computation
3. `Parameter.py` - Add new parameters

### Lines of Code:
- New: ~150 lines
- Modified: ~30 lines
- **Total**: ~180 lines

---

## 🚀 Next Steps After Implementation

1. **Validate Results**
   - Compare with baseline
   - Check if improvement is consistent across epochs
   - Verify on validation set

2. **Ablation Study**
   - Test different temperatures
   - Test different weights
   - Try full ContrastivePairEncoder vs Simplified

3. **Combine with Other Improvements**
   - Add Span Boundary Refinement
   - Add Cross-Attention Fusion
   - Ensemble multiple models

---

## 📚 References

1. **SimCLR**: A Simple Framework for Contrastive Learning (Chen et al., 2020)
2. **MoCo**: Momentum Contrast for Unsupervised Visual Representation Learning (He et al., 2020)
3. **InfoNCE**: Representation Learning with Contrastive Predictive Coding (Oord et al., 2018)

---

**Implementation Priority**: HIGH  
**Risk Level**: LOW  
**Expected ROI**: HIGH  

**Start Date**: January 6, 2026  
**Target Completion**: January 9, 2026  

---

## ✅ Implementation Checklist

- [ ] Create `models/Contrastive_Module.py`
- [ ] Modify `models/D2E2S_Model.py` - Add import
- [ ] Modify `models/D2E2S_Model.py` - Add to __init__
- [ ] Modify `models/D2E2S_Model.py` - Add _compute_contrastive_loss method
- [ ] Modify `models/D2E2S_Model.py` - Update _forward_train
- [ ] Modify `train.py` - Update loss computation
- [ ] Modify `Parameter.py` - Add parameters
- [ ] Create `test_contrastive.py` - Unit test
- [ ] Run small-scale test (1 epoch)
- [ ] Run full training (120 epochs)
- [ ] Analyze results
- [ ] Document findings

---

**Last Updated**: January 6, 2026, 22:30 IST
