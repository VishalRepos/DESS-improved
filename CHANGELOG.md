# CHANGELOG - DESS Improvements

This document tracks all improvements and changes made to the DESS (D2E2S) model architecture.

---

## [Pending] - 2026-01-15 11:10:00 +0530
### Enhancement: Cross-Attention Fusion (Replace TIN Concatenation)

**Motivation**: Previous TIN module simply concatenates semantic and syntactic features. Cross-attention allows features to query each other, learning which features are important for each context. This should provide better integration of complementary information.

**Why it will work**:
- Semantic features can query syntactic features and vice versa
- Multi-head attention learns feature importance dynamically
- Better than simple concatenation for integrating complementary information
- Lower risk than boundary refinement (which failed with -5.68%)

**Changes**:
- **models/Cross_Attention_Fusion.py**: 
  - New module with multi-head cross-attention between semantic and syntactic GCN outputs
  - Semantic features query syntactic (sem→syn attention)
  - Syntactic features query semantic (syn→sem attention)
  - Residual connections and layer normalization for stability
  - Configurable number of attention heads (default: 8)

- **Parameter.py**:
  - Added `--use_cross_attention` flag to enable cross-attention fusion
  - Added `--cross_attention_heads` parameter (default: 8)

- **models/D2E2S_Model.py**:
  - Replaced hardcoded TIN with conditional `fusion_module`
  - Uses `CrossAttentionFusion` when `--use_cross_attention` is enabled
  - Falls back to original `TIN` when disabled
  - Updated both `_forward_train` and `_forward_eval` methods

- **test_cross_attention.py**:
  - Comprehensive test script to validate the module
  - Tests shape compatibility, NaN/Inf checks, different batch sizes
  - Compares with TIN module interface

**Expected Impact**:
- Better semantic-syntactic feature integration
- Learned attention weights for context-dependent feature importance
- Expected improvement: +0.5-0.7% in Triplet F1
- Target: 77.6-77.8% (from 77.14% Enhanced SemGCN baseline)

**Testing**:
```bash
# Test the module
python test_cross_attention.py

# Train with cross-attention fusion
python train.py --dataset 14res --epochs 120 \
    --use_enhanced_semgcn \
    --use_cross_attention \
    --cross_attention_heads 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768
```

**Comparison with Previous Approaches**:
- ✅ Enhanced SemGCN: 77.14% (baseline)
- ❌ Contrastive Learning: 76.10% (-1.04%)
- ❌ Boundary Refinement: 71.46% (-5.68%)
- 🎯 Cross-Attention Fusion: Expected 77.6-77.8% (+0.5-0.7%)

**Advantages over TIN**:
1. **Dynamic feature weighting**: Attention learns importance, not fixed concatenation
2. **Bidirectional querying**: Both semantic and syntactic can query each other
3. **Multi-head attention**: Captures different aspects of feature relationships
4. **Residual connections**: Preserves original information while adding refinement

---

## [Pending] - 2025-12-31 06:35:00 +0530
### Revert: Remove dropout and layer normalization enhancement

**Motivation**: Enhancement [ef6fbec] did not show significant improvement in results (75.75% Triplet F1). Reverting to baseline for new experiments.

**Changes**:
- **Parameter.py**: Removed `--attention_dropout` and `--hidden_dropout` parameters
- **models/D2E2S_Model.py**: 
  - Removed config dropout modifications
  - Removed `deberta_layer_norm` and `deberta_dropout` layers
  - Reverted forward passes to original implementation (no residual connections)
- **DESS_Kaggle_P100.ipynb**: Removed dropout parameters from training commands

**Impact**: Code reverted to commit 9a909f0 state (with AdamW fixes retained)

**Baseline Results** (for comparison):
- Best Epoch: 94
- Entity F1: 87.65%
- Triplet F1: 75.75%

---

## [17fb0ee] - 2025-12-31 06:31:00 +0530
### fix: Remove correct_bias parameter from torch.optim.AdamW

**Motivation**: `correct_bias` parameter doesn't exist in `torch.optim.AdamW` (only in transformers' old AdamW)

**Changes**:
- **train.py**: Removed `correct_bias=False` parameter from AdamW initialization

**Impact**: Fixed compatibility with PyTorch's native AdamW optimizer

---

## [f423db5] - 2025-12-30 23:13:49 +0530
### fix: Import AdamW from torch.optim instead of transformers

**Motivation**: AdamW was moved from transformers to torch.optim in newer versions

**Changes**:
- **train.py**: Changed `from transformers import AdamW` to `from torch.optim import AdamW`

**Impact**: Fixed import error with newer transformers versions

---

## [033a45b] - 2025-12-30 21:14:21 +0530
### docs: Update CHANGELOG with timestamps for all commits

**Changes**: Added full timestamps (date + time + timezone) to all changelog entries

---

## [e2cb33d] - 2025-12-30 21:12:16 +0530
### docs: Add CHANGELOG.md to track all improvements and changes

**Changes**: Created changelog document to track all code improvements with commit IDs and timestamps

---

## [ef6fbec] - 2025-12-30 21:09:42 +0530
### Enhancement: Add improved dropout and layer normalization to DeBERTa transformer

**Motivation**: Improve model generalization and training stability on Kaggle P100 GPU

**Changes**:
- **Parameter.py**:
  - Added `--attention_dropout` parameter (default: 0.1) for attention layer dropout
  - Added `--hidden_dropout` parameter (default: 0.1) for hidden layer dropout

- **models/D2E2S_Model.py**:
  - Updated DeBERTa config to use configurable dropout rates
  - Added `nn.LayerNorm` after DeBERTa output for better gradient flow
  - Implemented residual connections: `dropout(layernorm(x)) + x`
  - Applied enhancements to both `_forward_train` and `_forward_eval` methods

- **DESS_Kaggle_P100.ipynb**:
  - Updated training commands with new dropout parameters
  - Added experimental cell for testing higher dropout values (0.2)

**Expected Impact**:
- Better regularization to prevent overfitting
- Improved gradient flow through deeper layers
- More stable training with residual connections
- Configurable dropout for experimentation

**Results** (120 epochs on 14res dataset):
- **Best Epoch**: 94
- **Entity F1**: 87.65%
- **Triplet F1**: 75.75% ⭐
- Model shows excellent stability in epochs 90-120

**Testing**:
```bash
python train.py --dataset 14res --epochs 120 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --attention_dropout 0.1 --hidden_dropout 0.1
```

---

## [9a909f0] - 2025-12-30 20:18:43 +0530
### Fix: Remove all hardcoded deberta-v2-xxlarge references

**Motivation**: Enable flexible model selection to avoid OOM errors on P100 GPU

**Changes**:
- **train.py**:
  - Changed `AutoConfig.from_pretrained("microsoft/deberta-v2-xxlarge")` to use `args.pretrained_deberta_name`

- **models/D2E2S_Model.py**:
  - Changed `AutoModel.from_pretrained("microsoft/deberta-v2-xxlarge")` to use `args.pretrained_deberta_name`

**Impact**: Can now use smaller models like deberta-v3-base (768 dim) instead of xxlarge (1536 dim)

---

## [84c8104] - 2025-12-30 20:14:29 +0530
### Fix: Use configurable tokenizer instead of hardcoded deberta-v2-xxlarge

**Motivation**: Tokenizer must match the model being used

**Changes**:
- **train.py**:
  - Changed `AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")` to use `args.pretrained_deberta_name`

**Impact**: Tokenizer now correctly matches the selected model

---

## [4b5ea0e] - 2025-04-30 13:05:31 +0530
### Architecture.drawio

**Changes**: Added architecture diagram

---

## [6eb6529] - 2025-02-14 00:00:02 +0530
### Initial commit - Deberta V2 xxlarge

**Changes**: Initial codebase with DeBERTa-v2-xxlarge model

---

## Training Results Summary

### Best Model Performance (Epoch 94):
- **Dataset**: Restaurant 2014 (14res)
- **Entity Extraction F1**: 87.65%
- **Triplet Extraction F1**: 75.75%
- **Configuration**: deberta-v3-base with enhanced dropout (0.1) and layer normalization

### Top 5 Epochs:
1. Epoch 94: 75.75% (Triplet F1)
2. Epoch 117: 75.42%
3. Epoch 99: 75.37%
4. Epoch 107: 75.22%
5. Epoch 98: 75.09%

---

## Future Improvements (Planned)

### Next Steps:
1. Enhanced GCN layers with attention mechanisms
2. Improved entity-sentiment pair classification
3. Multi-task learning enhancements
4. Advanced data augmentation techniques
5. Experiment with different dropout rates (0.15, 0.2)
6. Test on other datasets (14lap, 15res, 16res)

---

## How to Use This Changelog

When making changes:
1. Make your code changes
2. Test locally or on Kaggle
3. Commit with descriptive message
4. Add entry to this changelog with:
   - Commit hash (short: 7 chars)
   - Date and time with timezone
   - Description of changes
   - Motivation
   - Expected impact
   - Testing commands (if applicable)
   - Results (if available)

**Format**: `## [commit_hash] - YYYY-MM-DD HH:MM:SS +TIMEZONE`

---

**Last Updated**: 2025-12-31 06:30:00 +0530
