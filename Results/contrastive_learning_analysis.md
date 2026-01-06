# 📊 Contrastive Learning Results Analysis - Restaurant 2014 Dataset

**Date**: January 7, 2026, 00:45 IST  
**Configuration**: Enhanced Semantic GCN + Contrastive Learning  
**Dataset**: Restaurant 2014 (14res)  
**Total Epochs**: 120 (completed 117 evaluations)

---

## 🎯 FINAL RESULTS

### **Best Performance** (Epoch 89):
- **Entity F1**: 88.19%
- **Triplet F1**: 76.10%
- **Entity Precision**: 85.28%
- **Entity Recall**: 90.61%
- **Triplet Precision**: 74.45%
- **Triplet Recall**: 76.21%

### **Final Epoch** (Epoch 120):
- **Entity F1**: 87.77%
- **Triplet F1**: 74.72%

---

## 📈 Performance Comparison

| Configuration | Entity F1 | Triplet F1 | vs Baseline | Status |
|--------------|-----------|------------|-------------|--------|
| **Baseline (Original)** | 87.65% | 75.75% | --- | Previous |
| **+ Enhanced SemGCN** | 88.68% | 77.14% | +1.39% | Previous Best |
| **+ SemGCN + Contrastive** | 88.19% | **76.10%** | **+0.35%** | Current |

---

## ⚠️ UNEXPECTED RESULT: Performance Decreased

### Expected vs Actual:
- **Expected**: 77.6-78.0% Triplet F1 (+0.5-0.8% improvement)
- **Actual**: 76.10% Triplet F1 (-1.04% vs Enhanced SemGCN alone)

### Key Findings:
1. ❌ **Contrastive learning HURT performance** instead of helping
2. ❌ **Worse than Enhanced SemGCN alone** (77.14% → 76.10%)
3. ✅ **Still better than original baseline** (75.75% → 76.10%, +0.35%)
4. ⚠️ **Entity F1 also decreased** (88.68% → 88.19%, -0.49%)

---

## 📊 Top 10 Epochs by Triplet F1

| Rank | Epoch | Entity F1 | Triplet F1 | Notes |
|------|-------|-----------|------------|-------|
| 1 | 89 | 88.19% | **76.10%** | Best overall |
| 2 | 85 | 87.71% | 75.90% | |
| 3 | 109 | 88.29% | 75.83% | |
| 4 | 70 | 87.85% | 75.71% | |
| 5 | 75 | 87.77% | 75.68% | |
| 6 | 77 | 87.93% | 75.68% | |
| 7 | 87 | 88.09% | 75.59% | |
| 8 | 86 | 88.20% | 75.55% | |
| 9 | 90 | 88.09% | 75.53% | |
| 10 | 80 | 87.83% | 75.52% | |

---

## 📉 Training Statistics

- **Total Epochs Evaluated**: 117
- **Best Triplet F1**: 76.10% (Epoch 89)
- **Average Triplet F1**: 67.98%
- **Final Epoch F1**: 74.72%
- **Convergence**: Best epoch at 89, then slight decline

---

## 🔍 Detailed Analysis

### Why Did Contrastive Learning Fail?

#### 1. **Overfitting to Contrastive Task**
- Model may have focused too much on entity-opinion similarity
- Lost some discriminative power for sentiment classification
- Contrastive loss may have dominated the training signal

#### 2. **Feature Space Distortion**
- Pulling entity-opinion pairs together may have:
  - Reduced feature diversity
  - Made sentiment boundaries less clear
  - Interfered with existing well-learned representations

#### 3. **Hyperparameter Mismatch**
- **Contrastive weight (0.1)** may be too high
- **Temperature (0.07)** may be too aggressive
- Need to tune these parameters

#### 4. **Dataset Characteristics**
- Restaurant 2014 dataset may not benefit from contrastive learning
- Enhanced SemGCN already captures entity-opinion relationships well
- Adding contrastive loss is redundant

#### 5. **Implementation Issues**
- Possible bug in contrastive loss computation
- May be extracting wrong entity-opinion pairs
- Need to verify positive pair extraction logic

---

## 🎯 Performance by Sentiment Type (Best Epoch 89)

### Triplet Extraction:
| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| **POS** | 78.74% | 84.25% | 81.40% | 762 |
| **NEG** | 68.75% | 70.27% | 69.50% | 148 |
| **NEU** | 25.00% | 11.48% | 15.56% | 61 |
| **Micro Avg** | 74.45% | 76.21% | **76.10%** | 971 |

### Observations:
- ✅ **Positive sentiment**: Best performance (81.40% F1)
- ⚠️ **Negative sentiment**: Moderate (69.50% F1)
- ❌ **Neutral sentiment**: Poor (15.56% F1) - major weakness

---

## 📊 Training Progression

### Early Epochs (1-30):
- Rapid improvement from 0% to ~70% Triplet F1
- Model learning basic patterns

### Mid Epochs (31-70):
- Steady improvement to ~75% Triplet F1
- Approaching optimal performance

### Peak Performance (71-90):
- Best results around epoch 89 (76.10%)
- Stable performance in this range

### Late Epochs (91-120):
- Slight decline to ~74-75% Triplet F1
- Possible overfitting or instability

---

## 🔧 Recommendations

### Immediate Actions:

#### 1. **Reduce Contrastive Weight** ⭐⭐⭐
Try lower weights to reduce contrastive loss influence:
```bash
--contrastive_weight 0.05  # Half of current
--contrastive_weight 0.02  # Very low influence
```

#### 2. **Adjust Temperature** ⭐⭐
Try softer negatives:
```python
# In Contrastive_Module.py
SimplifiedContrastiveLoss(hidden_dim, temperature=0.1)  # Instead of 0.07
```

#### 3. **Verify Implementation** ⭐⭐⭐
Check if contrastive loss is computing correctly:
- Print number of positive pairs extracted
- Verify entity-opinion indices are correct
- Check if contrastive loss is decreasing

#### 4. **Try Without Contrastive Learning** ⭐⭐⭐
**Recommendation**: Use Enhanced SemGCN alone (77.14% F1)
```bash
# Remove --use_contrastive flag
python train.py --dataset 14res --epochs 120 \
    --use_enhanced_semgcn
```

---

## 💡 Alternative Approaches

### Option 1: Disable Contrastive Learning
**Best approach**: Stick with Enhanced SemGCN only (77.14% F1)

### Option 2: Tune Hyperparameters
Try different combinations:
- Weight: 0.01, 0.02, 0.05, 0.1
- Temperature: 0.05, 0.07, 0.1, 0.15

### Option 3: Modify Contrastive Loss
- Use only hard negatives (most similar incorrect pairs)
- Add margin to loss function
- Use triplet loss instead of InfoNCE

### Option 4: Apply Contrastive Learning Differently
- Only in early epochs (first 50 epochs)
- Gradually reduce weight over training
- Apply to specific layers only

---

## 📋 Comparison with Previous Results

### Enhanced SemGCN Only (Previous Best):
```
Best Epoch: 68
Entity F1: 88.68%
Triplet F1: 77.14%
```

### Enhanced SemGCN + Contrastive (Current):
```
Best Epoch: 89
Entity F1: 88.19% (-0.49%)
Triplet F1: 76.10% (-1.04%)
```

### Conclusion:
**Contrastive learning did NOT improve performance on this dataset.**

---

## 🎯 Next Steps

### Priority 1: Revert to Enhanced SemGCN Only ⭐⭐⭐
- Best proven configuration: 77.14% Triplet F1
- No need for contrastive learning on this dataset

### Priority 2: Try Other Improvements
Instead of contrastive learning, focus on:
1. **Span Boundary Refinement** - Better entity/opinion extraction
2. **Cross-Attention Fusion** - Better TIN module
3. **Data Augmentation** - More training examples
4. **Ensemble Methods** - Combine multiple models

### Priority 3: Test on Other Datasets
- Try contrastive learning on 14lap, 15res, 16res
- May work better on different data distributions

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| **Best Triplet F1** | 76.10% |
| **Best Entity F1** | 88.19% |
| **Best Epoch** | 89 |
| **Total Training Time** | ~120 epochs |
| **Improvement vs Baseline** | +0.35% |
| **Change vs Enhanced SemGCN** | -1.04% ❌ |

---

## ✅ Conclusions

1. **Contrastive learning DECREASED performance** on Restaurant 2014 dataset
2. **Enhanced SemGCN alone is better** (77.14% vs 76.10%)
3. **Contrastive weight (0.1) may be too high** - try lower values
4. **Dataset may not benefit** from entity-opinion contrastive learning
5. **Recommendation**: Use Enhanced SemGCN only, skip contrastive learning

---

## 🔬 Hypothesis for Failure

The most likely reason contrastive learning failed:

**Enhanced Semantic GCN already learns entity-opinion relationships effectively through:**
- Relative position encoding
- Global context aggregation
- Multi-scale features

**Adding contrastive learning:**
- Creates redundant learning signal
- Distorts well-learned feature space
- Adds unnecessary complexity
- May cause feature collapse (all pairs become too similar)

**Conclusion**: The model doesn't need explicit contrastive learning when it already has strong semantic relationship modeling.

---

**Status**: ❌ Contrastive learning did not improve performance  
**Recommendation**: Revert to Enhanced SemGCN only (77.14% F1)  
**Next**: Try other improvement strategies (Span Refinement, Cross-Attention, etc.)

**Last Updated**: January 7, 2026, 00:45 IST
