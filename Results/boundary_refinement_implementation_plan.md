# 🎯 Span Boundary Refinement - Implementation Plan

**Goal**: Improve span boundary detection for better entity/opinion extraction  
**Expected Gain**: +0.4-0.6% Triplet F1 (77.14% → 77.5-77.7%)  
**Priority**: HIGH ⭐⭐⭐  
**Date**: January 7, 2026

---

## 📊 Problem Analysis

### Current Implementation Issues:

**In `_classify_entities` method** (D2E2S_Model.py, line ~350):
```python
# Current: Simple max/average pooling
if self.args.span_generator == "Max":
    entity_spans_pool = entity_spans_pool.max(dim=2)[0]
else:
    entity_spans_pool = entity_spans_pool.mean(dim=2)
```

**Problems**:
1. ❌ **No boundary awareness** - treats all tokens equally
2. ❌ **Imprecise boundaries** - "delicious pasta" vs "pasta"
3. ❌ **No start/end distinction** - doesn't focus on boundary tokens
4. ❌ **Context ignored** - doesn't use surrounding tokens

### Common Errors:
- "the delicious pasta" → extracts "the delicious pasta" instead of "pasta"
- "very good service" → extracts "very good service" instead of "service"
- "not bad" → extracts "not" instead of "not bad"

---

## 🏗️ Solution: Boundary-Aware Attention Module

### Architecture:

```
Input: Span tokens [batch, num_spans, span_len, hidden_dim]
                    ↓
        Boundary-Aware BiLSTM
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
Start Attention              End Attention
(focus on first tokens)      (focus on last tokens)
    ↓                               ↓
Start Representation         End Representation
    ↓                               ↓
    └───────────────┬───────────────┘
                    ↓
        Boundary-Refined Span
        (weighted combination)
```

### Key Features:
1. **Bidirectional LSTM** - captures context in both directions
2. **Separate Start/End Attention** - focuses on boundary tokens
3. **Weighted Combination** - learns optimal start/end balance
4. **Residual Connection** - preserves original span information

---

## 📝 Implementation Steps

### Step 1: Create BoundaryRefinement Module (NEW FILE)

**File**: `models/Boundary_Refinement.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryRefinement(nn.Module):
    """
    Boundary-aware attention for refining entity/opinion span representations.
    Focuses on start and end tokens to improve boundary detection.
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
        # lstm_out: [batch * num_spans, span_len, hidden_dim]
        
        # Reshape back
        lstm_out = lstm_out.view(batch_size, num_spans, span_len, hidden_dim)
        
        # Compute start attention (focus on beginning tokens)
        start_scores = self.start_attention(lstm_out).squeeze(-1)
        # [batch, num_spans, span_len]
        
        # Compute end attention (focus on ending tokens)
        end_scores = self.end_attention(lstm_out).squeeze(-1)
        # [batch, num_spans, span_len]
        
        # Apply mask (set padding to -inf)
        if span_masks is not None:
            mask_value = -1e9
            start_scores = start_scores.masked_fill(span_masks == 0, mask_value)
            end_scores = end_scores.masked_fill(span_masks == 0, mask_value)
        
        # Softmax to get attention weights
        start_weights = F.softmax(start_scores, dim=-1)  # [batch, num_spans, span_len]
        end_weights = F.softmax(end_scores, dim=-1)
        
        # Weighted sum to get start and end representations
        start_repr = torch.sum(start_weights.unsqueeze(-1) * lstm_out, dim=2)
        # [batch, num_spans, hidden_dim]
        end_repr = torch.sum(end_weights.unsqueeze(-1) * lstm_out, dim=2)
        # [batch, num_spans, hidden_dim]
        
        # Concatenate start and end
        boundary_repr = torch.cat([start_repr, end_repr], dim=-1)
        # [batch, num_spans, hidden_dim * 2]
        
        # Fuse to original dimension
        refined = self.fusion(boundary_repr)
        refined = self.dropout(refined)
        
        # Layer norm with residual (use mean pooling as residual)
        residual = lstm_out.mean(dim=2)  # [batch, num_spans, hidden_dim]
        refined = self.layer_norm(refined + residual)
        
        return refined


class SimplifiedBoundaryRefinement(nn.Module):
    """
    Simplified version: Just boundary-aware attention without LSTM.
    Faster and uses less memory.
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
            span_masks: [batch, num_spans, span_len]
        
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
```

---

### Step 2: Modify D2E2S_Model.py

#### 2.1 Add Import (line ~10)
```python
from models.Boundary_Refinement import SimplifiedBoundaryRefinement
```

#### 2.2 Add to __init__ (after line ~100)
```python
# Boundary refinement for better span extraction
self.boundary_refiner = SimplifiedBoundaryRefinement(
    hidden_dim=self._emb_dim,
    dropout=self._prop_drop
)
self.use_boundary_refinement = getattr(self.args, 'use_boundary_refinement', False)
```

#### 2.3 Modify _classify_entities (around line 350)

**Current code**:
```python
# m: [batch, num_entities, span_len, 1]
m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)

if self.args.span_generator == "Max":
    entity_spans_pool = entity_spans_pool.max(dim=2)[0]
else:
    entity_spans_pool = entity_spans_pool.mean(dim=2).squeeze(-2)
```

**New code**:
```python
# m: [batch, num_entities, span_len, 1]
m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)

# Apply boundary refinement if enabled
if self.use_boundary_refinement:
    # Create mask: 1 for valid tokens, 0 for padding
    span_mask = (entity_masks != 0).float()  # [batch, num_entities, span_len]
    
    # Apply boundary refinement
    entity_spans_pool = self.boundary_refiner(entity_spans_pool, span_mask)
else:
    # Original pooling
    if self.args.span_generator == "Max":
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]
    else:
        entity_spans_pool = entity_spans_pool.mean(dim=2).squeeze(-2)
```

---

### Step 3: Add Parameter

**File**: `Parameter.py` (after line ~50)

```python
parser.add_argument(
    "--use_boundary_refinement",
    action="store_true",
    default=False,
    help="Use boundary-aware attention for span refinement"
)
```

---

### Step 4: Testing

#### 4.1 Quick Test (1 epoch)
```bash
cd Codebase
python train.py --dataset 14res --epochs 1 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn \
    --use_boundary_refinement
```

#### 4.2 Full Training (120 epochs)
```bash
python train.py --dataset 14res --epochs 120 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn \
    --use_boundary_refinement
```

---

## 📊 Expected Results

### Baseline (Enhanced SemGCN):
- Entity F1: 88.68%
- Triplet F1: 77.14%

### With Boundary Refinement:
- Entity F1: 89.0-89.3% (+0.3-0.6%)
- Triplet F1: **77.5-77.7%** (+0.4-0.6%)

### Why It Will Work:
1. ✅ **Better boundary detection** - focuses on start/end tokens
2. ✅ **Contextual awareness** - uses bidirectional information
3. ✅ **Learned weighting** - model learns optimal attention
4. ✅ **Residual connection** - preserves original information
5. ✅ **Low risk** - additive improvement, doesn't break existing

---

## 🔧 Implementation Variants

### Option 1: SimplifiedBoundaryRefinement (Recommended)
- **Pros**: Fast, low memory, easy to debug
- **Cons**: Less expressive than LSTM version
- **Use when**: Want quick results, limited GPU memory

### Option 2: BoundaryRefinement (Full LSTM)
- **Pros**: More expressive, better context modeling
- **Cons**: Slower, more memory, more parameters
- **Use when**: Have GPU resources, want maximum performance

### Option 3: Hybrid
- Use SimplifiedBoundaryRefinement for entities
- Use full BoundaryRefinement for sentiments (fewer spans)

---

## 🎯 Success Criteria

### Minimum Success:
- Triplet F1 ≥ 77.4% (+0.3%)
- Entity F1 ≥ 88.9% (+0.2%)
- Training stable, no NaN

### Target Success:
- Triplet F1 ≥ 77.5% (+0.4%)
- Entity F1 ≥ 89.0% (+0.3%)
- Consistent improvement across epochs

### Stretch Goal:
- Triplet F1 ≥ 77.7% (+0.6%)
- Entity F1 ≥ 89.3% (+0.6%)

---

## 🔍 Debugging Checklist

### If no improvement:
1. Check if boundary refinement is being called
2. Verify span_mask is correct (1 for valid, 0 for padding)
3. Print attention weights to see if they're reasonable
4. Try full BoundaryRefinement instead of Simplified
5. Adjust dropout rate (try 0.05 or 0.2)

### If NaN loss:
1. Check for division by zero in attention
2. Verify mask is applied correctly
3. Add gradient clipping
4. Reduce learning rate

### If worse performance:
1. Try without residual connection
2. Reduce dropout
3. Use only start OR end attention (not both)
4. Revert to original pooling

---

## 📋 Code Changes Summary

### New Files:
1. `models/Boundary_Refinement.py` (~150 lines)

### Modified Files:
1. `models/D2E2S_Model.py` (+15 lines)
2. `Parameter.py` (+6 lines)

### Total Changes:
- New code: ~150 lines
- Modified code: ~21 lines
- **Total**: ~171 lines

---

## 🚀 Next Steps After Implementation

1. **Validate Results**
   - Compare with baseline (77.14%)
   - Check if improvement is consistent
   - Test on validation set

2. **Ablation Study**
   - Test with/without LSTM
   - Test start-only vs end-only vs both
   - Test different dropout rates

3. **Combine with Other Improvements**
   - Add Cross-Attention Fusion
   - Add Data Augmentation
   - Try Ensemble

4. **If Successful**
   - Document results
   - Update Kaggle notebook
   - Move to next improvement

---

## 💡 Key Insights

### Why This Should Work:

1. **Addresses Real Problem**: Boundary errors are common in span extraction
2. **Focused Improvement**: Targets specific weakness in current model
3. **Low Risk**: Additive module, doesn't break existing functionality
4. **Proven Approach**: Boundary-aware attention used successfully in NER tasks
5. **Complementary**: Works well with Enhanced SemGCN

### What Makes It Different from Contrastive Learning:

| Aspect | Contrastive Learning | Boundary Refinement |
|--------|---------------------|---------------------|
| **Target** | Entity-opinion pairing | Span boundary detection |
| **Approach** | Global similarity learning | Local boundary attention |
| **Risk** | High (can distort features) | Low (additive improvement) |
| **Complexity** | Medium-High | Low-Medium |
| **Expected Gain** | 0.5-0.8% (failed) | 0.4-0.6% (likely) |

---

**Implementation Priority**: HIGH ⭐⭐⭐  
**Risk Level**: LOW  
**Expected ROI**: HIGH  

**Start Date**: January 7, 2026  
**Target Completion**: January 8, 2026  

---

**Last Updated**: January 7, 2026, 11:06 IST
