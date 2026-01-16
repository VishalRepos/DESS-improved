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

For each sentence during evaluation, you'll see a single-line JSON:

```json
{"ID": "sentence_1_0", "Text": "this version has been my least favorite version", "Quadruplet": [{"Aspect": "version", "Opinion": "least favorite", "Sentiment": "NEG"}]}
```

### Multiple Triplets:
```json
{"ID": "sentence_1_12", "Text": "The service is great, the desserts are excellent and the coffee is so very good", "Quadruplet": [{"Aspect": "service", "Opinion": "great", "Sentiment": "POS"}, {"Aspect": "desserts", "Opinion": "excellent", "Sentiment": "POS"}, {"Aspect": "coffee", "Opinion": "good", "Sentiment": "POS"}]}
```

### No Triplets Found:
```json
{"ID": "sentence_1_13", "Text": "They are served on Focacchia bread", "Quadruplet": [{"Aspect": "NULL", "Opinion": "NULL", "Sentiment": "NULL"}]}
```

---

## What You'll See

### JSON Format Output

Each sentence is printed as a single-line JSON object with:

1. **ID**: Unique identifier (format: `sentence_{epoch}_{index}`)
2. **Text**: The original sentence (cleaned)
3. **Quadruplet**: Array of triplets, each containing:
   - **Aspect**: Aspect term text (or "NULL" if none)
   - **Opinion**: Opinion term text (or "NULL" if none)
   - **Sentiment**: Sentiment polarity (POS/NEG/NEU or "NULL")

### Field Descriptions:

- **Aspect**: The target entity being discussed (e.g., "food", "service")
- **Opinion**: The opinion expression about the aspect (e.g., "delicious", "slow")
- **Sentiment**: The polarity of the opinion (POS=positive, NEG=negative, NEU=neutral)

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

### Example 1: Multiple Triplets
```json
{"ID": "sentence_1_0", "Text": "the food was delicious but service was slow", "Quadruplet": [{"Aspect": "food", "Opinion": "delicious", "Sentiment": "POS"}, {"Aspect": "service", "Opinion": "slow", "Sentiment": "NEG"}]}
```

**Interpretation:**
- Sentence has 2 triplets
- Triplet 1: "food" (aspect) + "delicious" (opinion) = POS sentiment
- Triplet 2: "service" (aspect) + "slow" (opinion) = NEG sentiment

### Example 2: Single Triplet
```json
{"ID": "sentence_1_1", "Text": "the pizza was amazing", "Quadruplet": [{"Aspect": "pizza", "Opinion": "amazing", "Sentiment": "POS"}]}
```

### Example 3: No Triplets
```json
{"ID": "sentence_1_2", "Text": "we went there yesterday", "Quadruplet": [{"Aspect": "NULL", "Opinion": "NULL", "Sentiment": "NULL"}]}
```

**Interpretation:**
- No aspect-opinion pairs found
- NULL values indicate no triplet extraction

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
python train.py --print_triplets ... > triplets.jsonl 2>&1
```

Then extract only JSON lines:
```bash
grep '^{' triplets.jsonl > clean_triplets.jsonl
```

Each line is a valid JSON object that can be parsed.

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
