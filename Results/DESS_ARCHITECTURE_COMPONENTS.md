# DESS (D2E2S) Model Architecture Components

**Date**: January 16, 2026  
**Model Version**: 1.1

---

## 📋 COMPLETE ARCHITECTURE COMPONENT LIST

### 1. ENCODER LAYER
**Component**: DeBERTa Transformer
- **Module**: `self.deberta` (AutoModel)
- **Type**: Pre-trained transformer (microsoft/deberta-v3-base or deberta-v2-xxlarge)
- **Input**: Token IDs
- **Output**: Contextualized embeddings (768 or 1536 dim)
- **Purpose**: Extract contextual word representations

---

### 2. SEQUENCE MODELING LAYER
**Component**: Bidirectional LSTM
- **Module**: `self.lstm`
- **Type**: nn.LSTM (2 layers, bidirectional)
- **Input**: DeBERTa embeddings
- **Output**: LSTM hidden states
- **Dropout**: `self.lstm_dropout`
- **Purpose**: Capture sequential dependencies

---

### 3. ATTENTION LAYER
**Component**: Self-Attention Module
- **Module**: `self.attention_layer` (SelfAttention)
- **Type**: Multi-head self-attention
- **Heads**: Configurable (default: 1)
- **Purpose**: Capture long-range dependencies

---

### 4. GRAPH CONVOLUTION LAYERS

#### 4a. Syntactic GCN
**Base Version**:
- **Module**: `self.Syn_gcn` (GCN)
- **Type**: Standard Graph Convolutional Network
- **Input**: Dependency tree adjacency matrix
- **Purpose**: Capture syntactic structure

**Enhanced Version** (Optional):
- **Module**: `self.Syn_gcn` (EnhancedSynGCN)
- **Components**:
  - GATv2 layers (Graph Attention Network v2)
  - GraphSAGE layers (Sample and Aggregate)
  - Hybrid fusion mechanism
- **Heads**: 4 (for GATv2)
- **Layers**: 2
- **Purpose**: Advanced syntactic feature extraction

#### 4b. Semantic GCN
**Base Version**:
- **Module**: `self.Sem_gcn` (SemGCN)
- **Type**: Attention-based semantic graph
- **Purpose**: Capture semantic relationships

**Enhanced Version** (Optional):
- **Module**: `self.Sem_gcn` (EnhancedSemGCN)
- **Components**:
  - Multi-head attention with relative position encoding
  - Global context aggregation
  - Multi-scale feature extraction
- **Max relative position**: 128
- **Purpose**: Advanced semantic feature extraction

---

### 5. FEATURE FUSION LAYER

#### 5a. TIN Module (Default)
**Component**: Token Interaction Network
- **Module**: `self.fusion_module` (TIN)
- **Components**:
  - 4 residual layers (Linear + ReLU + Linear + LayerNorm)
  - Bidirectional LSTM (2 layers)
  - Feature fusion MLP
- **Input**: Syntactic + Semantic features
- **Output**: Fused representations
- **Purpose**: Merge syntactic and semantic information

#### 5b. Cross-Attention Fusion (Optional)
**Component**: Multi-head Cross-Attention
- **Module**: `self.fusion_module` (CrossAttentionFusion)
- **Components**:
  - Semantic→Syntactic attention
  - Syntactic→Semantic attention
  - Layer normalization
  - Fusion MLP
- **Heads**: Configurable (4 or 8)
- **Purpose**: Learn dynamic feature importance

---

### 6. CLASSIFICATION LAYERS

#### 6a. Entity Classifier
**Component**: Entity Type Prediction
- **Module**: `self.entity_classifier`
- **Type**: nn.Linear
- **Input**: Contextualized span representation + size embedding
- **Input dim**: hidden_size × 2 + size_embedding
- **Output**: Entity type logits
- **Purpose**: Classify entity spans

#### 6b. Sentiment Classifier
**Component**: Triplet Sentiment Prediction
- **Module**: `self.senti_classifier`
- **Type**: nn.Linear
- **Input**: Entity pair representation + size embeddings
- **Input dim**: hidden_size × 3 + size_embedding × 2
- **Output**: Sentiment type logits (POS/NEG/NEU)
- **Purpose**: Classify sentiment between entity pairs

---

### 7. EMBEDDING LAYERS

#### 7a. Size Embeddings
**Component**: Span Size Encoding
- **Module**: `self.size_embeddings`
- **Type**: nn.Embedding(100, 25)
- **Purpose**: Encode span length information

---

### 8. REGULARIZATION LAYERS

#### 8a. Dropout Layers
- **Module**: `self.dropout` (prop_drop)
- **Module**: `self.dropout1` (0.5)
- **Module**: `self.dropout2` (0.0)
- **Module**: `self.lstm_dropout` (drop_rate)
- **Module**: `self.gcn_drop` (gcn_dropout)
- **Purpose**: Prevent overfitting

---

### 9. OPTIONAL ENHANCEMENT MODULES

#### 9a. Contrastive Learning Module (Optional)
**Component**: Entity-Opinion Pairing
- **Module**: `self.contrastive_encoder` (SimplifiedContrastiveLoss)
- **Temperature**: 0.07
- **Weight**: 0.1
- **Purpose**: Learn better entity-opinion associations
- **Status**: ❌ Failed (-1.04%)

#### 9b. Boundary Refinement Module (Optional)
**Component**: Span Boundary Attention
- **Module**: `self.boundary_refiner` (SimplifiedBoundaryRefinement)
- **Purpose**: Refine span boundaries
- **Status**: ❌ Failed (-5.68%)

---

## 🔄 DATA FLOW

```
Input Text
    ↓
[1] DeBERTa Encoder
    ↓
[2] Bidirectional LSTM
    ↓
[3] Self-Attention
    ↓
    ├─→ [4a] Syntactic GCN (dependency tree)
    │       ↓
    └─→ [4b] Semantic GCN (attention graph)
            ↓
        [5] Feature Fusion (TIN or Cross-Attention)
            ↓
        ├─→ [6a] Entity Classifier → Entity Predictions
        │
        └─→ [6b] Sentiment Classifier → Triplet Predictions
```

---

## 📊 COMPONENT SUMMARY

### Core Components (Always Active):
1. ✅ DeBERTa Encoder
2. ✅ Bidirectional LSTM
3. ✅ Self-Attention
4. ✅ Syntactic GCN (base or enhanced)
5. ✅ Semantic GCN (base or enhanced)
6. ✅ Feature Fusion (TIN or Cross-Attention)
7. ✅ Entity Classifier
8. ✅ Sentiment Classifier
9. ✅ Size Embeddings
10. ✅ Dropout Layers

### Optional Components (Configurable):
11. ⚙️ Enhanced Syntactic GCN (`--use_enhanced_syngcn`)
12. ⚙️ Enhanced Semantic GCN (`--use_enhanced_semgcn`)
13. ⚙️ Cross-Attention Fusion (`--use_cross_attention`)
14. ⚙️ Contrastive Learning (`--use_contrastive`) ❌ Not recommended
15. ⚙️ Boundary Refinement (`--use_boundary_refinement`) ❌ Not recommended

---

## 🎯 BEST CONFIGURATION

**Proven Best Performance** (77.14% Triplet F1):
```bash
--use_enhanced_semgcn
# Uses: DeBERTa + LSTM + Self-Attention + Enhanced SemGCN + TIN
```

**Components Active**:
1. DeBERTa Encoder (deberta-v3-base)
2. Bidirectional LSTM (2 layers)
3. Self-Attention (1 head)
4. Syntactic GCN (base)
5. **Enhanced Semantic GCN** ⭐
   - Multi-head attention with relative position
   - Global context aggregation
   - Multi-scale features
6. TIN Fusion
7. Entity + Sentiment Classifiers

---

## 📈 PARAMETER COUNTS (Approximate)

| Component | Parameters |
|-----------|-----------|
| DeBERTa (v3-base) | ~184M |
| LSTM | ~9M |
| Self-Attention | ~2M |
| Syntactic GCN | ~1M |
| Semantic GCN (Enhanced) | ~3M |
| TIN Fusion | ~4M |
| Classifiers | ~2M |
| **Total** | **~205M** |

---

## 🔧 CONFIGURABLE PARAMETERS

### Model Architecture:
- `--pretrained_deberta_name`: DeBERTa model variant
- `--deberta_feature_dim`: DeBERTa output dimension (768/1536)
- `--hidden_dim`: LSTM hidden dimension (384/768)
- `--emb_dim`: Embedding dimension (768/1536)
- `--lstm_layers`: Number of LSTM layers (2)
- `--attention_heads`: Self-attention heads (1)
- `--num_layers`: GCN layers (2)

### Regularization:
- `--prop_drop`: General dropout (0.1)
- `--drop_out_rate`: LSTM dropout (0.5)
- `--gcn_dropout`: GCN dropout (0.2)

### Training:
- `--batch_size`: Batch size (16)
- `--lr`: Learning rate (5e-6)
- `--epochs`: Training epochs (120)
- `--max_span_size`: Maximum entity span (8)

---

**Last Updated**: January 16, 2026, 22:45 IST
