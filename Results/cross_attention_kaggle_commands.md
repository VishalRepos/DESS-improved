# Cross-Attention Fusion - Kaggle Training Commands

**Date**: January 15, 2026  
**Status**: Ready for Training on Kaggle P100

---

## 🚀 Quick Start (Copy-Paste Ready)

### Step 1: Test the Module
```python
# In Kaggle notebook cell
!cd /kaggle/working/DESS && python test_cross_attention.py
```

### Step 2: Train with Cross-Attention Fusion
```python
# In Kaggle notebook cell
!cd /kaggle/working/DESS/Codebase && \
python train.py \
    --dataset 14res \
    --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --cross_attention_heads 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 \
    --hidden_dim 384 \
    --emb_dim 768 \
    --batch_size 16 \
    --lr 5e-6 \
    --prop_drop 0.1 \
    --gcn_dropout 0.2
```

---

## 📋 Alternative Configurations

### Configuration 1: Default (Recommended)
```bash
python train.py \
    --dataset 14res \
    --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --cross_attention_heads 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 \
    --hidden_dim 384 \
    --emb_dim 768
```
**Expected**: 77.6-77.8% Triplet F1

### Configuration 2: Fewer Heads (Faster)
```bash
python train.py \
    --dataset 14res \
    --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --cross_attention_heads 4 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 \
    --hidden_dim 384 \
    --emb_dim 768
```
**Expected**: 77.4-77.6% Triplet F1 (slightly lower, but faster)

### Configuration 3: More Heads (More Expressive)
```bash
python train.py \
    --dataset 14res \
    --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --cross_attention_heads 12 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 \
    --hidden_dim 384 \
    --emb_dim 768
```
**Expected**: 77.7-77.9% Triplet F1 (potentially higher, but slower)

### Configuration 4: Baseline (No Cross-Attention)
```bash
python train.py \
    --dataset 14res \
    --epochs 120 \
    --use_enhanced_semgcn \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 \
    --hidden_dim 384 \
    --emb_dim 768
```
**Expected**: 77.14% Triplet F1 (baseline for comparison)

---

## 📊 Expected Timeline

### Training Progress:
- **Epochs 1-20**: Initial learning (F1 ~40-60%)
- **Epochs 20-50**: Rapid improvement (F1 ~60-75%)
- **Epochs 50-80**: Fine-tuning (F1 ~75-77%)
- **Epochs 80-120**: Convergence (F1 ~77-78%)

### Best Epoch:
- Expected around epoch 70-90
- Monitor for stability in last 30 epochs

### Total Time:
- ~3-4 hours on Kaggle P100
- ~120 epochs × 2 minutes/epoch

---

## 🔍 Monitoring During Training

### Key Metrics to Watch:

1. **Triplet F1** (Most Important)
   - Target: ≥77.5% by epoch 80
   - Excellent: ≥77.8%
   - Red flag: <77.0% by epoch 100

2. **Entity F1**
   - Target: ≥88.7%
   - Should be stable throughout
   - Red flag: <88.0%

3. **Training Loss**
   - Should decrease smoothly
   - Red flag: Oscillating or increasing

4. **Convergence**
   - Should stabilize in last 30 epochs
   - Red flag: High variance (>1% std dev)

---

## 📝 Kaggle Notebook Template

```python
# Cell 1: Setup
import os
os.chdir('/kaggle/working')

# Clone or upload your code
# !git clone <your-repo>
# or upload DESS folder

# Cell 2: Install Dependencies (if needed)
# !pip install torch-geometric
# !pip install transformers

# Cell 3: Test Cross-Attention Module
!cd /kaggle/working/DESS && python test_cross_attention.py

# Cell 4: Train with Cross-Attention
!cd /kaggle/working/DESS/Codebase && \
python train.py \
    --dataset 14res \
    --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --cross_attention_heads 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 \
    --hidden_dim 384 \
    --emb_dim 768 \
    --batch_size 16 \
    --lr 5e-6

# Cell 5: Check Results
!ls -lh /kaggle/working/DESS/Codebase/log/
!tail -100 /kaggle/working/DESS/Codebase/log/train_14res.log

# Cell 6: Find Best Epoch
!grep "Best" /kaggle/working/DESS/Codebase/log/train_14res.log
```

---

## 🎯 Success Criteria

### Minimum Success (Match Baseline):
- ✅ Triplet F1 ≥ 77.14%
- ✅ Entity F1 ≥ 88.68%
- ✅ No training instability
- ✅ Reasonable convergence

### Target Success:
- ✅ Triplet F1 ≥ 77.5% (+0.36%)
- ✅ Entity F1 ≥ 88.7%
- ✅ Stable training (std dev <0.5%)
- ✅ Convergence by epoch 80

### Excellent Success:
- ✅ Triplet F1 ≥ 77.8% (+0.66%)
- ✅ Entity F1 ≥ 89.0%
- ✅ Very stable (std dev <0.3%)
- ✅ Fast convergence (by epoch 70)

---

## 🐛 Troubleshooting

### Problem 1: OOM Error
```python
# Solution: Reduce batch size or attention heads
!cd /kaggle/working/DESS/Codebase && \
python train.py \
    --dataset 14res \
    --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --cross_attention_heads 4 \
    --batch_size 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 \
    --hidden_dim 384 \
    --emb_dim 768
```

### Problem 2: Poor Results (<77.0%)
```python
# Solution: Try different head counts
# Try 4 heads
--cross_attention_heads 4

# Try 12 heads
--cross_attention_heads 12

# Or revert to baseline (no cross-attention)
# Remove --use_cross_attention flag
```

### Problem 3: Training Instability
```python
# Solution: Increase dropout or reduce learning rate
!cd /kaggle/working/DESS/Codebase && \
python train.py \
    --dataset 14res \
    --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --cross_attention_heads 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 \
    --hidden_dim 384 \
    --emb_dim 768 \
    --prop_drop 0.15 \
    --lr 3e-6
```

---

## 📊 Results Template

After training, document results in this format:

```markdown
# Cross-Attention Fusion Results

**Date**: [Date]
**Configuration**: Enhanced SemGCN + Cross-Attention (8 heads)
**Dataset**: Restaurant 2014 (14res)

## Best Results (Epoch X):
- Entity F1: XX.XX%
- Triplet F1: XX.XX%

## Comparison:
| Configuration | Entity F1 | Triplet F1 | Change |
|--------------|-----------|------------|--------|
| Enhanced SemGCN | 88.68% | 77.14% | Baseline |
| + Cross-Attention | XX.XX% | XX.XX% | +X.XX% |

## Top 5 Epochs:
1. Epoch X: XX.XX%
2. Epoch X: XX.XX%
3. Epoch X: XX.XX%
4. Epoch X: XX.XX%
5. Epoch X: XX.XX%

## Analysis:
[Your analysis here]

## Conclusion:
[Success/Failure and next steps]
```

---

## ✅ Pre-Training Checklist

Before starting training, verify:

- [ ] Code uploaded to Kaggle
- [ ] test_cross_attention.py runs successfully
- [ ] Enhanced SemGCN flag is enabled
- [ ] Cross-attention flag is enabled
- [ ] Correct model name (deberta-v3-base)
- [ ] Correct dimensions (768, 384, 768)
- [ ] Correct dataset (14res)
- [ ] Epochs set to 120
- [ ] GPU accelerator enabled in Kaggle

---

## 🔄 After Training

### If Successful (≥77.5%):
1. Save the model checkpoint
2. Document results in Results/ folder
3. Update CHANGELOG.md with actual performance
4. Try on other datasets (14lap, 15res, 16res)
5. Consider ensemble methods

### If Marginal (77.2-77.4%):
1. Try different attention head counts
2. Experiment with dropout rates
3. Try longer training (150 epochs)
4. Combine with data augmentation

### If Failed (<77.0%):
1. Document failure analysis
2. Revert to Enhanced SemGCN only
3. Try alternative approaches
4. Consider data augmentation or ensemble

---

## 📈 Expected Output Format

Training will output logs like:
```
Epoch 1/120: Entity F1: 45.23%, Triplet F1: 32.45%
Epoch 2/120: Entity F1: 52.34%, Triplet F1: 38.67%
...
Epoch 68/120: Entity F1: 88.72%, Triplet F1: 77.56% ⭐
...
Epoch 120/120: Entity F1: 88.45%, Triplet F1: 77.23%

Best Epoch: 68
Best Entity F1: 88.72%
Best Triplet F1: 77.56%
```

---

## 🎯 Final Checklist

- [ ] Upload code to Kaggle
- [ ] Run test script
- [ ] Start training with cross-attention
- [ ] Monitor progress (check every 20 epochs)
- [ ] Document results
- [ ] Update CHANGELOG
- [ ] Decide next steps based on results

---

**Status**: ✅ Ready for Kaggle Training  
**Expected Time**: 3-4 hours  
**Expected Result**: 77.6-77.8% Triplet F1  
**Risk Level**: Low

**Good luck with training! 🚀**

**Last Updated**: January 15, 2026, 11:10 IST
