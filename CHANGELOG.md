# CHANGELOG - DESS Improvements

This document tracks all improvements and changes made to the DESS (D2E2S) model architecture.

---

## [ef6fbec] - 2025-12-30
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

**Testing**:
```bash
python train.py --dataset 14res --epochs 120 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --attention_dropout 0.1 --hidden_dropout 0.1
```

---

## [9a909f0] - 2025-12-30
### Fix: Remove all hardcoded deberta-v2-xxlarge references

**Motivation**: Enable flexible model selection to avoid OOM errors on P100 GPU

**Changes**:
- **train.py**:
  - Changed `AutoConfig.from_pretrained("microsoft/deberta-v2-xxlarge")` to use `args.pretrained_deberta_name`

- **models/D2E2S_Model.py**:
  - Changed `AutoModel.from_pretrained("microsoft/deberta-v2-xxlarge")` to use `args.pretrained_deberta_name`

**Impact**: Can now use smaller models like deberta-v3-base (768 dim) instead of xxlarge (1536 dim)

---

## [84c8104] - 2025-12-30
### Fix: Use configurable tokenizer instead of hardcoded deberta-v2-xxlarge

**Motivation**: Tokenizer must match the model being used

**Changes**:
- **train.py**:
  - Changed `AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")` to use `args.pretrained_deberta_name`

**Impact**: Tokenizer now correctly matches the selected model

---

## [4b5ea0e] - 2025-12-30
### Architecture.drawio

**Changes**: Added architecture diagram

---

## [6eb6529] - 2025-12-30
### Initial commit - Deberta V2 xxlarge

**Changes**: Initial codebase with DeBERTa-v2-xxlarge model

---

## Future Improvements (Planned)

### Next Steps:
1. Enhanced GCN layers with attention mechanisms
2. Improved entity-sentiment pair classification
3. Multi-task learning enhancements
4. Advanced data augmentation techniques

---

## How to Use This Changelog

When making changes:
1. Make your code changes
2. Test locally or on Kaggle
3. Commit with descriptive message
4. Add entry to this changelog with:
   - Commit hash
   - Date
   - Description of changes
   - Motivation
   - Expected impact
   - Testing commands (if applicable)

---

**Last Updated**: 2025-12-30
