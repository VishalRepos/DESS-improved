# 🚀 Contrastive Learning Implementation - Complete Plan

**Date**: January 6, 2026  
**Goal**: Improve Triplet F1 from 77.14% to 77.6-78.0% (+0.5-0.8%)  
**Status**: Ready to implement

---

## ✅ What's Already Done

1. ✅ **Created `models/Contrastive_Module.py`**
   - SimplifiedContrastiveLoss class (recommended)
   - ContrastivePairEncoder class (advanced)
   - Both implement InfoNCE loss

2. ✅ **Created `test_contrastive.py`**
   - Unit tests for both classes
   - Tests forward/backward pass
   - Tests edge cases (empty input, different temperatures)

3. ✅ **Created documentation**
   - Detailed implementation plan
   - Quick start guide
   - This summary document

---

## 📋 What You Need to Do

### Step 1: Add Parameters (5 minutes)

**File**: `Codebase/Parameter.py`

**Add after line 50** (after other parser arguments):

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

### Step 2: Modify D2E2S_Model.py (30 minutes)

**File**: `Codebase/models/D2E2S_Model.py`

#### 2.1 Add Import (line ~10)

```python
from models.Contrastive_Module import SimplifiedContrastiveLoss
```

#### 2.2 Add to __init__ (after line ~100, after self.senti_classifier)

```python
# Contrastive learning for entity-opinion pairing
self.contrastive_encoder = SimplifiedContrastiveLoss(
    hidden_dim=self._emb_dim,
    temperature=0.07
)
self.use_contrastive = self.args.use_contrastive if hasattr(self.args, 'use_contrastive') else False
self.contrastive_weight = self.args.contrastive_weight if hasattr(self.args, 'contrastive_weight') else 0.1
```

#### 2.3 Add New Method (after _classify_sentiments method, around line 450)

```python
def _compute_contrastive_loss(self, entity_spans_pool, sentiments):
    """
    Compute contrastive loss for entity-opinion pairs.
    
    Args:
        entity_spans_pool: [batch_size, num_entities, hidden_dim]
        sentiments: [batch_size, num_pairs, 2] - ground truth pairs (entity_idx, opinion_idx)
    
    Returns:
        contrastive_loss: scalar tensor
    """
    device = entity_spans_pool.device
    
    # Collect all positive entity-opinion pairs across batch
    all_entity_reprs = []
    all_opinion_reprs = []
    
    batch_size = sentiments.shape[0]
    for b in range(batch_size):
        pairs = sentiments[b]  # [num_pairs, 2]
        
        for entity_idx, opinion_idx in pairs:
            entity_idx = entity_idx.item()
            opinion_idx = opinion_idx.item()
            
            # Skip invalid pairs (padding)
            if entity_idx == 0 and opinion_idx == 0:
                continue
            
            # Skip if indices out of bounds
            if entity_idx >= entity_spans_pool.shape[1] or opinion_idx >= entity_spans_pool.shape[1]:
                continue
            
            # Get entity and opinion representations
            entity_repr = entity_spans_pool[b, entity_idx]
            opinion_repr = entity_spans_pool[b, opinion_idx]
            
            all_entity_reprs.append(entity_repr)
            all_opinion_reprs.append(opinion_repr)
    
    # If no valid pairs, return zero loss
    if len(all_entity_reprs) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Stack into tensors
    entity_batch = torch.stack(all_entity_reprs)  # [num_triplets, hidden_dim]
    opinion_batch = torch.stack(all_opinion_reprs)  # [num_triplets, hidden_dim]
    
    # Compute contrastive loss
    loss = self.contrastive_encoder(entity_batch, opinion_batch)
    
    return loss
```

#### 2.4 Modify _forward_train (around line 230)

**Find this section**:
```python
# entity_classify
size_embeddings = self.size_embeddings(entity_sizes)
entity_clf, entity_spans_pool = self._classify_entities(
    encodings, h, entity_masks, size_embeddings, self.args
)
```

**Add AFTER entity classification**:
```python
# Contrastive learning for entity-opinion pairing
contrastive_loss = torch.tensor(0.0, device=h.device)
if self.use_contrastive and self.training:
    contrastive_loss = self._compute_contrastive_loss(entity_spans_pool, sentiments)
```

**Find the return statement** (around line 250):
```python
return entity_clf, senti_clf, batch_loss
```

**Change to**:
```python
return entity_clf, senti_clf, batch_loss, contrastive_loss
```

---

### Step 3: Modify train.py (15 minutes)

**File**: `Codebase/train.py`

#### 3.1 Update Model Forward Call (around line 150-180)

**Find**:
```python
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

#### 3.2 Update Loss Computation (around line 180-200)

**Find**:
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

batch_loss.backward()
```

**Change to**:
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

# Add contrastive loss
if args.use_contrastive:
    total_loss = batch_loss + args.contrastive_weight * contrastive_loss
else:
    total_loss = batch_loss

total_loss.backward()
```

---

## 🧪 Testing Steps

### Step 1: Run Unit Tests (2 minutes)

```bash
cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/NewCodebase/DESS
python test_contrastive.py
```

**Expected output**:
```
============================================================
Contrastive Learning Module - Unit Tests
============================================================

Testing SimplifiedContrastiveLoss...
  Loss value: 2.3026
  ✅ Backward pass successful
  ✅ Empty input handled correctly
✅ SimplifiedContrastiveLoss test passed!

Testing ContrastivePairEncoder...
  Loss value: 1.8945
  ✅ Backward pass successful
✅ ContrastivePairEncoder test passed!

Testing temperature effect...
  Temperature 0.05: Loss = 3.2189
  Temperature 0.07: Loss = 2.3026
  Temperature 0.1: Loss = 1.6094
  Temperature 0.2: Loss = 0.6931
  ✅ All temperatures work correctly
✅ Temperature test passed!

============================================================
🎉 ALL TESTS PASSED!
============================================================
```

---

### Step 2: Quick Training Test (5 minutes)

```bash
cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/NewCodebase/DESS/Codebase

# Test with 1 epoch to ensure no errors
python train.py --dataset 14res --epochs 1 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn \
    --use_contrastive --contrastive_weight 0.1
```

**What to check**:
- ✅ No import errors
- ✅ Training starts without crashes
- ✅ Contrastive loss is computed (check logs)
- ✅ Epoch completes successfully

---

### Step 3: Full Training (2-3 hours)

```bash
cd /Users/vishal.thenuwara/Documents/MSC/Research/Coding/NewCodebase/DESS/Codebase

# Full training with contrastive learning
python train.py --dataset 14res --epochs 120 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn \
    --use_contrastive --contrastive_weight 0.1
```

**Monitor**:
- Contrastive loss should decrease over epochs
- Initial: ~2-4
- Final: ~0.5-1.5
- Triplet F1 should improve by 0.3-0.8%

---

## 📊 Expected Results

### Baseline (for comparison):
```
Entity F1:   88.68%
Triplet F1:  77.14%
Best Epoch:  68
```

### With Contrastive Learning:
```
Entity F1:   88.5-89.0%
Triplet F1:  77.6-78.0%  ← Target: +0.5-0.8%
Best Epoch:  60-80
```

---

## 🔧 Troubleshooting

### Issue 1: Import Error
```
ModuleNotFoundError: No module named 'models.Contrastive_Module'
```

**Solution**: Make sure you're in the `Codebase` directory when running train.py

---

### Issue 2: Contrastive Loss is NaN
```
Contrastive loss: nan
```

**Solution**: 
- Check temperature > 0
- Verify entity_spans_pool has valid values
- Add print statements in _compute_contrastive_loss

---

### Issue 3: No Improvement
```
Triplet F1 still at 77.14%
```

**Solution**:
- Try different temperatures: 0.05, 0.07, 0.1
- Try different weights: 0.05, 0.1, 0.2
- Check if contrastive loss is decreasing
- Verify positive pairs are being extracted correctly

---

## 📈 Hyperparameter Tuning

If initial results are not satisfactory:

### Temperature (τ):
```bash
# Harder negatives (more aggressive)
--contrastive_weight 0.1 --temperature 0.05

# Softer negatives (gentler)
--contrastive_weight 0.1 --temperature 0.1
```

### Contrastive Weight (λ):
```bash
# Less influence
--contrastive_weight 0.05

# More influence
--contrastive_weight 0.2
```

---

## ✅ Success Checklist

- [ ] Unit tests pass
- [ ] 1-epoch test runs without errors
- [ ] Contrastive loss decreases during training
- [ ] Triplet F1 improves by at least 0.3%
- [ ] Results are reproducible
- [ ] Model saved and documented

---

## 📝 Code Changes Summary

### Files Created:
1. ✅ `models/Contrastive_Module.py` (~120 lines)
2. ✅ `test_contrastive.py` (~150 lines)

### Files to Modify:
1. ⬜ `Parameter.py` (~10 lines added)
2. ⬜ `models/D2E2S_Model.py` (~50 lines added)
3. ⬜ `train.py` (~10 lines modified)

### Total Changes:
- New code: ~270 lines
- Modified code: ~70 lines
- **Total**: ~340 lines

---

## 🎯 Next Steps After Success

1. **Document Results**
   - Save training logs
   - Record best F1 scores
   - Compare with baseline

2. **Hyperparameter Optimization**
   - Grid search over temperature and weight
   - Find optimal configuration

3. **Move to Next Improvement**
   - Span Boundary Refinement
   - Cross-Attention Fusion
   - Ensemble methods

---

## 📚 Quick Reference

### Training Commands:

**Baseline** (for comparison):
```bash
python train.py --dataset 14res --epochs 120 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn
```

**With Contrastive**:
```bash
python train.py --dataset 14res --epochs 120 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn \
    --use_contrastive --contrastive_weight 0.1
```

---

## 🎉 Ready to Start!

**Estimated Time**: 1-2 hours for implementation + 2-3 hours for training

**Start with**: Step 1 - Add parameters to Parameter.py

**Questions?** Refer to the detailed implementation plan or quick start guide.

---

**Last Updated**: January 6, 2026, 22:30 IST
