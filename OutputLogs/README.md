# Triplet Extraction from Training Logs

## Quick Start

### 1. Save training output to file

```bash
python train.py --print_triplets --dataset 14res --epochs 1 \
    --use_enhanced_semgcn > OutputLogs/output.txt 2>&1
```

### 2. Extract triplets to JSONL

```bash
python extract_triplets.py
```

This will:
- Read from: `OutputLogs/output.txt`
- Write to: `OutputTriplets/triplets.jsonl`

---

## Directory Structure

```
DESS/
├── extract_triplets.py          # Extraction script
├── OutputLogs/                  # Place your output.txt here
│   └── output.txt              # Training logs with JSON triplets
└── OutputTriplets/              # Extracted JSONL files saved here
    └── triplets.jsonl          # Clean JSON lines (one per sentence)
```

---

## Usage

### Step 1: Generate output.txt

Run training with `--print_triplets` flag and redirect output:

```bash
# In Kaggle
!python train.py --print_triplets --dataset 14res --epochs 1 \
    --use_enhanced_semgcn > /kaggle/working/OutputLogs/output.txt 2>&1

# Locally
python train.py --print_triplets --dataset 14res --epochs 1 \
    --use_enhanced_semgcn > OutputLogs/output.txt 2>&1
```

### Step 2: Extract triplets

```bash
python extract_triplets.py
```

**Output:**
```
📖 Reading from: OutputLogs/output.txt
💾 Writing to: OutputTriplets/triplets.jsonl

✅ Extraction complete!
📊 Total triplets extracted: 150
📁 Output saved to: OutputTriplets/triplets.jsonl

📋 Preview (first 3 lines):
  1. ID: sentence_1_0, Triplets: 2
  2. ID: sentence_1_1, Triplets: 1
  3. ID: sentence_1_2, Triplets: 0
```

---

## Output Format

Each line in `triplets.jsonl` is a valid JSON object:

```json
{"ID": "sentence_1_0", "Text": "the food was delicious", "Quadruplet": [{"Aspect": "food", "Opinion": "delicious", "Sentiment": "POS"}]}
{"ID": "sentence_1_1", "Text": "service was slow", "Quadruplet": [{"Aspect": "service", "Opinion": "slow", "Sentiment": "NEG"}]}
{"ID": "sentence_1_2", "Text": "we went yesterday", "Quadruplet": [{"Aspect": "NULL", "Opinion": "NULL", "Sentiment": "NULL"}]}
```

---

## Processing JSONL File

### Python

```python
import json

with open('OutputTriplets/triplets.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        print(f"Text: {data['Text']}")
        print(f"Triplets: {data['Quadruplet']}")
```

### Pandas

```python
import pandas as pd
import json

# Read JSONL
data = []
with open('OutputTriplets/triplets.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
print(df.head())
```

### Command Line

```bash
# Count total sentences
wc -l OutputTriplets/triplets.jsonl

# View first 10
head -10 OutputTriplets/triplets.jsonl

# Pretty print first entry
head -1 OutputTriplets/triplets.jsonl | python -m json.tool
```

---

## Troubleshooting

### Issue: output.txt is empty
**Solution:** Make sure you used `--print_triplets` flag during training

### Issue: No JSON lines found
**Solution:** Check if output.txt contains lines starting with `{"ID":`

### Issue: OutputLogs folder not found
**Solution:** Create it manually: `mkdir -p OutputLogs`

---

## Notes

- The script automatically creates `OutputTriplets/` folder if it doesn't exist
- Invalid JSON lines are skipped automatically
- Each line in the JSONL file is independently parseable
- File encoding is UTF-8 to support all characters

---

**Last Updated**: January 17, 2026, 07:47 IST
