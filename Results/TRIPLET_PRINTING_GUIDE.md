# Triplet Extraction Visualization - Usage Guide

## Feature Added: Print Triplet Extraction During Training

This modification allows you to see how each sentence is classified into triplets during evaluation.

---

## How to Use

### Enable Triplet Printing:

Add the `--print_triplets` flag to your training command:

```bash
python train.py --dataset 14res --epochs 120 \
    --use_enhanced_semgcn \
    --print_triplets \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768
```

---

## Output Format

For each sentence during evaluation, you'll see:

```
================================================================================
SENTENCE 0:
Text: this version has been my least favorite version i've had for the following reasons listed bellow the pros.
--------------------------------------------------------------------------------
PREDICTED ENTITIES (2):
  1. [1:2] 'version' -> ASPECT (score: 0.923)
  2. [5:7] 'least favorite' -> OPINION (score: 0.856)

PREDICTED TRIPLETS (1):
  1. Aspect: 'version' [1:2] (ASPECT)
     Opinion: 'least favorite' [5:7] (OPINION)
     Sentiment: NEG (score: 0.789)
================================================================================
```

---

## What You'll See

### 1. Sentence Text
- The original input sentence (cleaned)

### 2. Predicted Entities
- Token span positions [start:end]
- Entity text extracted from tokens
- Entity type (ASPECT/OPINION)
- Confidence score

### 3. Predicted Triplets
- Aspect entity (text, position, type)
- Opinion entity (text, position, type)
- Sentiment polarity (POS/NEG/NEU)
- Confidence score

---

## When to Use

### During Development:
```bash
--print_triplets --epochs 1
```
- See how model extracts triplets
- Debug entity and sentiment classification
- Understand model predictions

### During Training:
```bash
# NOT recommended - too much output
# Only use for debugging specific epochs
```

### For Analysis:
```bash
--print_triplets --epochs 1 > triplet_analysis.txt
```
- Save output to file for analysis
- Compare predictions across epochs

---

## Example Output Interpretation

```
SENTENCE 0:
Text: the food was delicious but service was slow

PREDICTED ENTITIES (4):
  1. [1:2] 'food' -> ASPECT (score: 0.95)
  2. [3:4] 'delicious' -> OPINION (score: 0.92)
  3. [5:6] 'service' -> ASPECT (score: 0.94)
  4. [7:8] 'slow' -> OPINION (score: 0.88)

PREDICTED TRIPLETS (2):
  1. Aspect: 'food' [1:2] (ASPECT)
     Opinion: 'delicious' [3:4] (OPINION)
     Sentiment: POS (score: 0.91)
  
  2. Aspect: 'service' [5:6] (ASPECT)
     Opinion: 'slow' [7:8] (OPINION)
     Sentiment: NEG (score: 0.85)
```

**Interpretation:**
- Model correctly identified 2 aspects: "food" and "service"
- Model correctly identified 2 opinions: "delicious" and "slow"
- Model correctly paired them into 2 triplets
- Model correctly classified sentiments: POS for food, NEG for service

---

## Files Modified

1. **`trainer/evaluator.py`**
   - Added `print_triplets` parameter to `__init__`
   - Added `_print_triplet_extraction()` method
   - Modified `eval_batch()` to call print function

2. **`train.py`**
   - Pass `print_triplets` parameter to Evaluator

3. **`Parameter.py`**
   - Added `--print_triplets` argument

---

## Performance Impact

- **Minimal** - Only prints during evaluation (not training)
- **Output volume** - Can be large for full datasets
- **Recommendation**: Use only for debugging or analysis

---

## Tips

### 1. Limit Output
```bash
# Only print first epoch
--print_triplets --epochs 1
```

### 2. Save to File
```bash
python train.py --print_triplets ... > output.txt 2>&1
```

### 3. Filter Specific Sentences
Modify the code to only print specific sentence indices:
```python
if sample_idx < 5:  # Only first 5 sentences
    self._print_triplet_extraction(...)
```

---

## Troubleshooting

### Too Much Output
- Remove `--print_triplets` flag
- Or redirect to file: `> output.txt`

### No Output
- Check if evaluation is running
- Ensure `--print_triplets` flag is present
- Check if sentences have predictions

### Garbled Text
- This is normal for special tokens
- Text is cleaned with `_prettify()` method

---

**Last Updated**: January 16, 2026, 22:55 IST
