/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py:441: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
Parse dataset 'train': 100%|██████████████| 1264/1264 [00:00<00:00, 1819.48it/s]
Parse dataset 'test': 100%|█████████████████| 480/480 [00:00<00:00, 1938.29it/s]
    14res    8
Using Original Syntactic GCN
Using Enhanced Semantic GCN with relative position, global context, and multi-scale features
Some weights of D2E2SModel were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['TIN.residual_layer2.2.bias', 'Sem_gcn.W.1.bias', 'TIN.residual_layer3.2.weight', 'TIN.GatedGCN.conv3.rnn.bias_ih', 'TIN.residual_layer3.0.weight', 'TIN.residual_layer2.3.weight', 'TIN.lstm.bias_ih_l1', 'attention_layer.linear_q.weight', 'attention_layer.v.weight', 'TIN.lstm.bias_ih_l0_reverse', 'Sem_gcn.global_context.gate.weight', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'TIN.lstm.weight_ih_l0', 'Sem_gcn.global_context.fc.weight', 'lstm.bias_ih_l0_reverse', 'Sem_gcn.global_context.fc.bias', 'lstm.bias_hh_l1_reverse', 'attention_layer.w_query.weight', 'TIN.feature_fusion.3.weight', 'TIN.residual_layer1.2.weight', 'Sem_gcn.W.1.weight', 'TIN.residual_layer4.3.bias', 'lstm.weight_ih_l0_reverse', 'contrastive_encoder.opinion_proj.bias', 'contrastive_encoder.opinion_proj.weight', 'boundary_refiner.fusion.bias', 'attention_layer.linear_q.bias', 'TIN.residual_layer4.0.bias', 'TIN.lstm.bias_hh_l0', 'TIN.residual_layer3.2.bias', 'lstm.weight_ih_l1_reverse', 'TIN.lstm.weight_ih_l0_reverse', 'TIN.residual_layer1.0.bias', 'boundary_refiner.end_attention.bias', 'TIN.lstm.weight_hh_l1', 'TIN.GatedGCN.conv3.weight', 'contrastive_encoder.entity_proj.bias', 'entity_classifier.weight', 'TIN.residual_layer4.3.weight', 'TIN.residual_layer2.3.bias', 'lstm.weight_hh_l0_reverse', 'TIN.residual_layer2.2.weight', 'entity_classifier.bias', 'TIN.residual_layer1.3.bias', 'TIN.GatedGCN.conv1.bias', 'TIN.lstm.bias_ih_l1_reverse', 'senti_classifier.weight', 'TIN.residual_layer1.2.bias', 'fc.weight', 'Sem_gcn.multi_scale.fusion.weight', 'TIN.feature_fusion.2.weight', 'TIN.GatedGCN.conv3.rnn.weight_ih', 'lstm.weight_hh_l0', 'TIN.lstm.bias_ih_l0', 'fc.bias', 'contrastive_encoder.entity_proj.weight', 'TIN.feature_fusion.0.weight', 'TIN.lstm.weight_hh_l1_reverse', 'TIN.residual_layer4.2.weight', 'TIN.residual_layer3.3.weight', 'lstm.bias_hh_l1', 'attention_layer.w_query.bias', 'lstm.weight_hh_l1_reverse', 'TIN.GatedGCN.conv2.lin.weight', 'Sem_gcn.multi_scale.fusion.bias', 'Syn_gcn.W.1.bias', 'TIN.lstm.bias_hh_l0_reverse', 'TIN.lstm.weight_hh_l0', 'boundary_refiner.start_attention.bias', 'TIN.GatedGCN.conv3.rnn.weight_hh', 'Sem_gcn.attn.linears.1.bias', 'TIN.GatedGCN.conv1.lin.weight', 'TIN.lstm.weight_hh_l0_reverse', 'Sem_gcn.attn.relative_position_k.weight', 'size_embeddings.weight', 'lstm.bias_ih_l1', 'TIN.residual_layer4.2.bias', 'TIN.residual_layer1.3.weight', 'TIN.lstm.weight_ih_l1', 'Sem_gcn.W.0.weight', 'TIN.residual_layer2.0.weight', 'Syn_gcn.W.0.bias', 'TIN.lstm.bias_hh_l1_reverse', 'TIN.lstm.bias_hh_l1', 'Sem_gcn.attn.linears.0.weight', 'lstm.weight_hh_l1', 'attention_layer.w_value.weight', 'TIN.residual_layer4.0.weight', 'Sem_gcn.global_context.gate.bias', 'boundary_refiner.fusion.weight', 'lstm.bias_hh_l0_reverse', 'TIN.residual_layer2.0.bias', 'TIN.GatedGCN.conv2.bias', 'lstm.bias_ih_l1_reverse', 'boundary_refiner.end_attention.weight', 'TIN.feature_fusion.3.bias', 'lstm.weight_ih_l0', 'boundary_refiner.start_attention.weight', 'TIN.residual_layer1.0.weight', 'TIN.feature_fusion.0.bias', 'TIN.residual_layer3.3.bias', 'TIN.feature_fusion.2.bias', 'Sem_gcn.attn.linears.1.weight', 'TIN.residual_layer3.0.bias', 'TIN.lstm.weight_ih_l1_reverse', 'TIN.GatedGCN.conv3.rnn.bias_hh', 'senti_classifier.bias', 'lstm.weight_ih_l1', 'attention_layer.w_value.bias', 'Sem_gcn.multi_scale.scale_weights', 'Sem_gcn.attn.relative_position_v.weight', 'Syn_gcn.W.0.weight', 'Syn_gcn.W.1.weight', 'Sem_gcn.W.0.bias', 'Sem_gcn.attn.linears.0.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Train epoch 0:   0%|                                     | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 0: 100%|████████████████████████████| 79/79 [00:29<00:00,  2.69it/s]
Evaluate epoch 1:   0%|                                  | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 1: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.79it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 1:   0%|                                     | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 1: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 2:   0%|                                  | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 2: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.78it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 2:   0%|                                     | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 2: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 3:   0%|                                  | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 3: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.65it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 3:   0%|                                     | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 3: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.82it/s]
Evaluate epoch 4:   0%|                                  | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 4: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.75it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 4:   0%|                                     | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 4: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.88it/s]
Evaluate epoch 5:   0%|                                  | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 5: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.74it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 5:   0%|                                     | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 5: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.90it/s]
Evaluate epoch 6:   0%|                                  | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 6: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 6:   0%|                                     | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 6: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 7:   0%|                                  | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 7: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.67it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 7:   0%|                                     | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 7: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 8:   0%|                                  | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 8: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.43it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 8:   0%|                                     | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 8: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.81it/s]
Evaluate epoch 9:   0%|                                  | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 9: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.69it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 9:   0%|                                     | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 9: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 10:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 10: 100%|████████████████████████| 30/30 [00:03<00:00,  9.57it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 10:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 10: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 11:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 11: 100%|████████████████████████| 30/30 [00:03<00:00,  9.60it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 11:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 11: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 12:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 12: 100%|████████████████████████| 30/30 [00:03<00:00,  9.67it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 12:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 12: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 13:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 13: 100%|████████████████████████| 30/30 [00:03<00:00,  9.46it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 13:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 13: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 14:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 14: 100%|████████████████████████| 30/30 [00:03<00:00,  9.65it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 14:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 14: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 15:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 15: 100%|████████████████████████| 30/30 [00:03<00:00,  9.58it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 15:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 15: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.89it/s]
Evaluate epoch 16:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 16: 100%|████████████████████████| 30/30 [00:03<00:00,  9.90it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 16:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 16: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.88it/s]
Evaluate epoch 17:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 17: 100%|████████████████████████| 30/30 [00:03<00:00,  9.67it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00         61.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 17:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 17:  27%|███████▏                   | 21/79 [00:07<00:19,  3.00it/s]^C
Train epoch 17:  27%|███████▏                   | 21/79 [00:07<00:20,  2.87it/s]
Traceback (most recent call last):
  File "/kaggle/working/DESS-improved/Codebase/DESS-improved/Codebase/DESS-improved/Codebase/train.py", line 383, in <module>
    trainer._train(
  File "/kaggle/working/DESS-improved/Codebase/DESS-improved/Codebase/DESS-improved/Codebase/train.py", line 130, in _train
    self.train_epoch(
  File "/kaggle/working/DESS-improved/Codebase/DESS-improved/Codebase/DESS-improved/Codebase/train.py", line 180, in train_epoch
    epoch_loss = compute_loss.compute(
                 ^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/DESS-improved/Codebase/DESS-improved/Codebase/DESS-improved/Codebase/trainer/loss.py", line 38, in compute
    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/utils/clip_grad.py", line 36, in _no_grad_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/utils/clip_grad.py", line 222, in clip_grad_norm_
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/utils/clip_grad.py", line 36, in _no_grad_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/utils/clip_grad.py", line 147, in _clip_grads_with_norm_
    grads = [p.grad for p in parameters if p.grad is not None]
             ^^^^^^
KeyboardInterrupt