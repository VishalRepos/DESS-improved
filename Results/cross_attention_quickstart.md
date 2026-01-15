# Cross-Attention Fusion - Quick Start Guide

## 📋 Overview

Cross-Attention Fusion replaces the simple TIN concatenation with multi-head cross-attention between semantic and syntactic GCN outputs. This allows features to query each other and learn which features are important for each context.

## 🎯 Why This Approach?

**Previous Failures:**
- ❌ Contrastive Learning: -1.04% (76.10%)
- ❌ Boundary Refinement: -5.68% (71.46%)

**Why Cross-Attention Will Work:**
1. **Dynamic Feature Integration**: Learns which features matter, not fixed concatenation
2. **Bidirectional Querying**: Semantic ↔ Syntactic mutual attention
3. **Multi-Head Attention**: Captures different feature relationships
4. **Lower Risk**: Replaces existing module, not adding complexity
5. **Proven Approach**: Cross-attention widely successful in NLP

**Expected Improvement:** +0.5-0.7% (Target: 77.6-77.8%)

---

## 🚀 Quick Start

### 1. Test the Module (Optional)

```bash
cd DESS
python test_cross_attention.py
```

This validates:
- ✓ Correct input/output shapes
- ✓ No NaN/Inf values
- ✓ Compatible with TIN interface
- ✓ Works with different batch sizes

### 2. Train with Cross-Attention Fusion

**Basic Command:**
```bash
python train.py --dataset 14res --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768
```

**With Custom Attention Heads:**
```bash
python train.py --dataset 14res --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --cross_attention_heads 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768
```

**Without Cross-Attention (Original TIN):**
```bash
python train.py --dataset 14res --epochs 120 \
    --use_enhanced_semgcn \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768
```

---

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_cross_attention` | False | Enable cross-attention fusion |
| `--cross_attention_heads` | 8 | Number of attention heads |

**Recommended Settings:**
- **Heads = 8**: Good balance (default)
- **Heads = 4**: Faster, less expressive
- **Heads = 12**: More expressive, slower

---

## 📊 Expected Results

### Baseline (Enhanced SemGCN Only)
- Entity F1: 88.68%
- Triplet F1: **77.14%**
- Best Epoch: 68

### Target (Enhanced SemGCN + Cross-Attention)
- Entity F1: 88.7-89.0%
- Triplet F1: **77.6-77.8%** (+0.5-0.7%)
- Expected: Stable convergence

---

## 🏗️ Architecture Details

### Original TIN Module
```
Semantic Features ──┐
                    ├─→ Concatenate ──→ LSTM ──→ Output
Syntactic Features ─┘
```

### Cross-Attention Fusion
```
Semantic Features ──┬─→ Query Syntactic ──┐
                    │                      ├─→ Fuse ──→ Output
Syntactic Features ─┴─→ Query Semantic ───┘
```

**Key Components:**
1. **Sem→Syn Attention**: Semantic queries syntactic features
2. **Syn→Sem Attention**: Syntactic queries semantic features
3. **Residual Connections**: Preserve original information
4. **Layer Normalization**: Stable training
5. **Fusion Layer**: Combine attended features

---

## 📁 Files Modified

1. **`models/Cross_Attention_Fusion.py`** (NEW)
   - CrossAttentionFusion module
   - Multi-head cross-attention implementation

2. **`Parameter.py`**
   - Added `--use_cross_attention` flag
   - Added `--cross_attention_heads` parameter

3. **`models/D2E2S_Model.py`**
   - Conditional fusion module selection
   - Uses CrossAttentionFusion or TIN based on flag

4. **`test_cross_attention.py`** (NEW)
   - Validation script for the module

5. **`CHANGELOG.md`**
   - Documentation of the enhancement

---

## 🔍 How It Works

### Step 1: Residual Connections
```python
h_syn = LayerNorm(h_feature + h_syn_feature)
h_sem = LayerNorm(h_feature + h_sem_feature)
```

### Step 2: Cross-Attention
```python
# Semantic queries syntactic
sem_attended = MultiHeadAttention(
    query=h_sem, key=h_syn, value=h_syn
)

# Syntactic queries semantic
syn_attended = MultiHeadAttention(
    query=h_syn, key=h_sem, value=h_sem
)
```

### Step 3: Fusion
```python
# Add residuals
sem_attended = h_sem + sem_attended
syn_attended = h_syn + syn_attended

# Concatenate and fuse
concat = cat([sem_attended, syn_attended])
output = FusionLayer(concat)
```

---

## 🎯 Success Criteria

**Minimum Success:**
- Triplet F1 ≥ 77.14% (match baseline)
- No training instability
- Reasonable convergence time

**Target Success:**
- Triplet F1 ≥ 77.5% (+0.36%)
- Entity F1 ≥ 88.7%
- Stable training (low variance)

**Excellent Success:**
- Triplet F1 ≥ 77.8% (+0.66%)
- Entity F1 ≥ 89.0%
- Faster convergence than baseline

---

## 🐛 Troubleshooting

### Issue: OOM (Out of Memory)
**Solution:** Reduce attention heads
```bash
--cross_attention_heads 4
```

### Issue: Slow Training
**Solution:** Reduce attention heads or use original TIN
```bash
--cross_attention_heads 4
# or remove --use_cross_attention
```

### Issue: Poor Results
**Solution:** Try different head counts
```bash
# Try 4, 8, or 12 heads
--cross_attention_heads 4
--cross_attention_heads 12
```

---

## 📈 Monitoring Training

**Key Metrics to Watch:**
1. **Triplet F1**: Should reach 77.5%+ by epoch 70-80
2. **Entity F1**: Should stay around 88.7%+
3. **Training Loss**: Should decrease smoothly
4. **Convergence**: Should be stable (low variance)

**Red Flags:**
- ⚠️ Triplet F1 < 77.0% by epoch 80
- ⚠️ High variance in F1 scores
- ⚠️ Training loss not decreasing
- ⚠️ NaN or Inf in losses

---

## 🔄 Comparison with Previous Approaches

| Approach | Triplet F1 | Change | Status |
|----------|------------|--------|--------|
| Enhanced SemGCN | 77.14% | Baseline | ✅ Current Best |
| + Contrastive | 76.10% | -1.04% | ❌ Failed |
| + Boundary Refine | 71.46% | -5.68% | ❌ Failed Badly |
| **+ Cross-Attention** | **77.6-77.8%** | **+0.5-0.7%** | 🎯 **Target** |

---

## 💡 Why This Should Work

### 1. **Addresses TIN Limitation**
- TIN just concatenates features
- Cross-attention learns which features matter
- Dynamic weighting based on context

### 2. **Proven Approach**
- Cross-attention is standard in transformers
- Successfully used in many NLP tasks
- Well-understood and stable

### 3. **Lower Risk**
- Replaces existing module (not adding complexity)
- Same interface as TIN
- Can easily revert if needed

### 4. **Complementary Features**
- Semantic: meaning-based relationships
- Syntactic: structure-based relationships
- Cross-attention finds best combination

---

## 📝 Next Steps After Training

### If Successful (≥77.5%)
1. ✅ Document results in Results/ folder
2. ✅ Update CHANGELOG with actual performance
3. ✅ Try on other datasets (14lap, 15res, 16res)
4. ✅ Consider ensemble with baseline

### If Marginal (77.2-77.4%)
1. 🔧 Try different attention head counts
2. 🔧 Experiment with dropout rates
3. 🔧 Combine with data augmentation

### If Failed (<77.0%)
1. ❌ Revert to Enhanced SemGCN only
2. ❌ Document failure in Results/
3. ❌ Try alternative approaches (data augmentation, ensemble)

---

## 🎓 Technical Details

**Module Parameters:**
- Attention heads: 8 (default)
- Dropout: 0.1
- Hidden dim: 768 (deberta-v3-base)
- Layer norm: Applied after residuals

**Computational Cost:**
- Slightly higher than TIN (due to attention)
- Acceptable for P100 GPU
- Can reduce heads if needed

**Memory Usage:**
- Similar to TIN
- Attention matrices are small (seq_len × seq_len)
- No significant memory increase

---

## ✅ Checklist Before Training

- [ ] Test module with `test_cross_attention.py`
- [ ] Verify Enhanced SemGCN is enabled (`--use_enhanced_semgcn`)
- [ ] Set correct model (`microsoft/deberta-v3-base`)
- [ ] Set correct dimensions (768, 384, 768)
- [ ] Enable cross-attention (`--use_cross_attention`)
- [ ] Set attention heads (default 8 is good)
- [ ] Set epochs to 120
- [ ] Monitor training logs

---

**Status**: ✅ Implementation Complete - Ready for Training  
**Expected Improvement**: +0.5-0.7% (77.6-77.8%)  
**Risk Level**: Low (can easily revert to TIN)  
**Next**: Train and evaluate on 14res dataset

**Last Updated**: January 15, 2026, 11:10 IST
