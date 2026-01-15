# Cross-Attention Performance Issue - Quick Fix

**Issue**: Training is slow, CPU bottleneck, GPU not fully utilized

## Root Cause:
The Cross-Attention Fusion module is doing **2x cross-attention** operations per forward pass:
1. Semantic → Syntactic attention
2. Syntactic → Semantic attention

This doubles the computation compared to TIN's simple concatenation.

## Quick Fixes Applied:

### Fix 1: Add `need_weights=False` (DONE)
- Updated `Cross_Attention_Fusion.py`
- Skips attention weight computation (we don't use them anyway)
- **Expected speedup**: 10-15%

### Fix 2: Use Fewer Attention Heads
Instead of 8 heads, use 4:
```bash
python train.py --dataset 14res --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --cross_attention_heads 4  # Changed from 8
```
**Expected speedup**: 30-40%

### Fix 3: Use Lightweight Version (NEW)
Created `Lightweight_Cross_Attention.py` with:
- Single attention pass (not bidirectional)
- 4 heads default
- Simpler fusion layer

To use:
```python
# In D2E2S_Model.py, replace:
from models.Cross_Attention_Fusion import CrossAttentionFusion
# with:
from models.Lightweight_Cross_Attention import LightweightCrossAttentionFusion as CrossAttentionFusion
```
**Expected speedup**: 50%+

## Recommended Solution:

**Option 1: Quick (No code change)**
```bash
# Use 4 heads instead of 8
--cross_attention_heads 4
```

**Option 2: Revert to TIN (Fastest)**
```bash
# Remove --use_cross_attention flag
python train.py --dataset 14res --epochs 120 \
    --use_enhanced_semgcn \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768
```

**Option 3: Use Lightweight Version**
- Requires updating D2E2S_Model.py import
- Best balance of speed and performance

## Why Cross-Attention is Slower:

| Module | Operations | Complexity |
|--------|-----------|------------|
| TIN | Concatenation + LSTM | O(n) |
| Cross-Attention (8 heads) | 2x Multi-head Attention | O(n²) × 2 × 8 |
| Lightweight (4 heads) | 1x Multi-head Attention | O(n²) × 4 |

Cross-attention is inherently more expensive due to attention computation.

## Immediate Action:

**Try with 4 heads first** (no code change needed):
```bash
--cross_attention_heads 4
```

If still slow, **revert to TIN** (77.14% baseline) and try other improvements.

---

**Updated**: January 15, 2026, 12:30 IST
