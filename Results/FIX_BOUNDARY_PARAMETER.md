# 🔧 Fix: unrecognized arguments --use_boundary_refinement

**Error**: `train.py: error: unrecognized arguments: --use_boundary_refinement`

**Cause**: You're using old code that doesn't have the boundary refinement parameter.

---

## ✅ Solution: Pull Latest Code

### In Kaggle, Cell 1 (Clone Repository):

**Update to**:
```python
# Remove old directory if exists
!rm -rf DESS-improved

# Clone fresh
!git clone https://github.com/VishalRepos/DESS-improved.git
%cd DESS-improved/Codebase

# Verify latest commit
!echo "\n=== Latest Commits ==="
!git log --oneline -3

# Should show:
# fd5a419 feat: Update Kaggle notebook with Boundary Refinement
# 0290ee2 feat: Add Span Boundary Refinement module
```

---

## 🔍 Verify Parameter Exists

After cloning, run this to verify:

```python
!grep -n "use_boundary_refinement" Parameter.py
```

**Expected output**:
```
64:        "--use_boundary_refinement",
67:        help="Use boundary-aware attention for span refinement"
```

---

## 🚀 Then Run Training

```bash
python train.py --dataset 14res --epochs 1 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --deberta_feature_dim 768 --hidden_dim 384 --emb_dim 768 \
    --use_enhanced_semgcn \
    --use_boundary_refinement
```

---

## Alternative: Manual Fix (If Pull Doesn't Work)

If you can't pull, add this to `Parameter.py` manually (after line 60):

```python
parser.add_argument(
    "--use_boundary_refinement",
    action="store_true",
    default=False,
    help="Use boundary-aware attention for span refinement"
)
```

---

**Quick Fix**: Just re-run Cell 1 in Kaggle to clone fresh code!
