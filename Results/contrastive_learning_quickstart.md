# 🎯 Contrastive Learning - Quick Start Guide

## What is Contrastive Learning?

**Simple Explanation**: Teach the model that correct entity-opinion pairs should be "close" in embedding space, while incorrect pairs should be "far apart".

### Example:
```
Sentence: "The pasta was delicious but service was slow"

Positive Pairs (should be close):
✅ (pasta, delicious)
✅ (service, slow)

Negative Pairs (should be far):
❌ (pasta, slow)
❌ (service, delicious)
❌ (pasta, pasta)
❌ (delicious, slow)
```

---

## Why Will This Work?

### Current Problem:
Your model classifies each entity-opinion pair independently without learning what makes a good pair vs a bad pair.

### Solution:
Contrastive learning explicitly teaches:
1. **Attraction**: Pull positive pairs together
2. **Repulsion**: Push negative pairs apart
3. **Discrimination**: Learn to distinguish good from bad pairs

---

## Implementation Overview

### 3 Main Components:

#### 1. **Contrastive Encoder** (NEW)
```
Entity Representation → Project → Normalize → 
                                              Compare Similarity
Opinion Representation → Project → Normalize →
```

#### 2. **Loss Function** (InfoNCE)
```
For each positive pair (entity, opinion):
  - Compute similarity with all opinions
  - Maximize similarity with correct opinion
  - Minimize similarity with wrong opinions
```

#### 3. **Integration** (Modify existing code)
```
Total Loss = Entity Loss + Sentiment Loss + λ * Contrastive Loss
                                            ↑
                                      (λ = 0.1)
```

---

## Files to Create/Modify

### ✅ NEW FILE: `models/Contrastive_Module.py`
- SimplifiedContrastiveLoss class
- ~80 lines of code

### ✏️ MODIFY: `models/D2E2S_Model.py`
- Add contrastive encoder to __init__
- Add _compute_contrastive_loss method
- Update _forward_train return
- ~50 lines added

### ✏️ MODIFY: `train.py`
- Update loss computation
- ~10 lines modified

### ✏️ MODIFY: `Parameter.py`
- Add --use_contrastive flag
- Add --contrastive_weight parameter
- ~10 lines added

---

## Step-by-Step Implementation

### Step 1: Create Contrastive Module (30 min)
```bash
# Create new file
touch models/Contrastive_Module.py
```

Copy the SimplifiedContrastiveLoss class from the implementation plan.

### Step 2: Modify D2E2S_Model (1 hour)
1. Add import at top
2. Add self.contrastive_encoder in __init__
3. Add _compute_contrastive_loss method
4. Update _forward_train to return contrastive_loss

### Step 3: Modify train.py (30 min)
1. Update model forward call to receive contrastive_loss
2. Add contrastive_loss to total loss

### Step 4: Add Parameters (15 min)
1. Add --use_contrastive flag
2. Add --contrastive_weight parameter

### Step 5: Test (1 hour)
```bash
# Quick test (1 epoch)
python train.py --dataset 14res --epochs 1 \
    --use_enhanced_semgcn --use_contrastive

# Full training
python train.py --dataset 14res --epochs 120 \
    --use_enhanced_semgcn --use_contrastive
```

---

## Expected Timeline

| Task | Time | Status |
|------|------|--------|
| Create Contrastive_Module.py | 30 min | ⬜ |
| Modify D2E2S_Model.py | 1 hour | ⬜ |
| Modify train.py | 30 min | ⬜ |
| Modify Parameter.py | 15 min | ⬜ |
| Unit testing | 30 min | ⬜ |
| Small-scale test (1 epoch) | 15 min | ⬜ |
| Full training (120 epochs) | 2-3 hours | ⬜ |
| Analysis | 30 min | ⬜ |
| **TOTAL** | **~6 hours** | ⬜ |

---

## Training Command

### Baseline (for comparison):
```bash
python train.py --dataset 14res --epochs 120 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn
```

### With Contrastive Learning:
```bash
python train.py --dataset 14res --epochs 120 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn \
    --use_contrastive --contrastive_weight 0.1
```

---

## Expected Results

### Before (Baseline):
```
Entity F1:   88.68%
Triplet F1:  77.14%
```

### After (With Contrastive):
```
Entity F1:   88.5-89.0%
Triplet F1:  77.6-78.0%  ← +0.5-0.8% improvement ✨
```

---

## Debugging Tips

### If loss is NaN:
- Check temperature > 0
- Verify no division by zero
- Use torch.logsumexp for stability

### If no improvement:
- Try different temperatures (0.05, 0.07, 0.1)
- Adjust contrastive_weight (0.05, 0.1, 0.2)
- Check if contrastive loss is decreasing

### If memory error:
- Reduce batch size
- Process triplets in chunks
- Use gradient checkpointing

---

## Key Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| temperature | 0.07 | 0.05-0.1 | Lower = harder negatives |
| contrastive_weight | 0.1 | 0.05-0.2 | Higher = more influence |
| projection_dim | 256 | 128-512 | Higher = more expressive |

---

## Validation Checklist

✅ Code compiles without errors  
✅ Training runs for 1 epoch  
✅ Contrastive loss decreases over epochs  
✅ Triplet F1 improves by 0.3%+  
✅ Results are reproducible  

---

## Next Steps After Success

1. **Document Results**
   - Save best model
   - Record F1 scores
   - Compare with baseline

2. **Hyperparameter Tuning**
   - Try different temperatures
   - Try different weights
   - Find optimal configuration

3. **Move to Next Improvement**
   - Span Boundary Refinement
   - Cross-Attention Fusion
   - Ensemble methods

---

## Quick Reference: Key Code Snippets

### Contrastive Loss Computation:
```python
# In _forward_train
if self.use_contrastive and self.training:
    contrastive_loss = self._compute_contrastive_loss(
        entity_spans_pool, entity_masks, sentiments, h
    )
```

### Total Loss:
```python
# In train.py
if args.use_contrastive:
    total_loss = batch_loss + args.contrastive_weight * contrastive_loss
else:
    total_loss = batch_loss
```

### Contrastive Encoder:
```python
# In __init__
self.contrastive_encoder = SimplifiedContrastiveLoss(
    hidden_dim=self._emb_dim,
    temperature=0.07
)
```

---

## Success Metrics

### Minimum Success:
- Triplet F1: 77.4% (+0.3%)
- Training stable, no errors

### Target Success:
- Triplet F1: 77.6-78.0% (+0.5-0.8%)
- Consistent across epochs

### Stretch Goal:
- Triplet F1: 78.0%+ (+0.9%+)
- Generalizes to other datasets

---

**Ready to implement?** Start with Step 1: Create `models/Contrastive_Module.py`

**Questions?** Refer to the detailed implementation plan document.

**Last Updated**: January 6, 2026, 22:30 IST
