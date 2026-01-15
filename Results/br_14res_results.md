/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py:441: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
Parse dataset 'train': 100%|██████████████| 1264/1264 [00:00<00:00, 1720.18it/s]
Parse dataset 'test': 100%|█████████████████| 480/480 [00:00<00:00, 1875.92it/s]
    14res    8
Using Original Syntactic GCN
Using Enhanced Semantic GCN with relative position, global context, and multi-scale features
Some weights of D2E2SModel were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['Sem_gcn.attn.linears.1.bias', 'TIN.residual_layer4.3.bias', 'Sem_gcn.W.1.weight', 'TIN.residual_layer3.2.bias', 'Sem_gcn.global_context.gate.bias', 'lstm.bias_ih_l1', 'lstm.bias_ih_l0_reverse', 'TIN.feature_fusion.3.bias', 'senti_classifier.weight', 'fc.bias', 'TIN.lstm.bias_hh_l0', 'Syn_gcn.W.1.weight', 'TIN.residual_layer4.3.weight', 'TIN.residual_layer1.0.weight', 'TIN.lstm.weight_hh_l0_reverse', 'TIN.residual_layer2.3.weight', 'TIN.feature_fusion.2.weight', 'TIN.lstm.bias_hh_l1', 'TIN.lstm.bias_ih_l1', 'TIN.residual_layer2.2.bias', 'lstm.weight_hh_l1', 'boundary_refiner.start_attention.bias', 'lstm.weight_hh_l0_reverse', 'lstm.bias_hh_l0', 'attention_layer.w_value.bias', 'TIN.residual_layer1.3.weight', 'TIN.GatedGCN.conv1.bias', 'lstm.weight_ih_l0_reverse', 'TIN.feature_fusion.0.weight', 'Sem_gcn.W.1.bias', 'boundary_refiner.end_attention.bias', 'TIN.GatedGCN.conv3.rnn.weight_hh', 'lstm.weight_hh_l1_reverse', 'TIN.GatedGCN.conv3.weight', 'Sem_gcn.attn.relative_position_k.weight', 'senti_classifier.bias', 'TIN.residual_layer1.2.bias', 'TIN.GatedGCN.conv2.bias', 'TIN.residual_layer2.0.bias', 'TIN.lstm.weight_ih_l1_reverse', 'TIN.residual_layer4.2.weight', 'TIN.residual_layer1.0.bias', 'TIN.GatedGCN.conv2.lin.weight', 'size_embeddings.weight', 'attention_layer.linear_q.weight', 'TIN.lstm.bias_hh_l1_reverse', 'Syn_gcn.W.0.bias', 'Sem_gcn.attn.relative_position_v.weight', 'fc.weight', 'lstm.weight_hh_l0', 'Sem_gcn.W.0.weight', 'entity_classifier.bias', 'TIN.GatedGCN.conv3.rnn.bias_hh', 'attention_layer.v.weight', 'TIN.GatedGCN.conv3.rnn.weight_ih', 'TIN.residual_layer3.0.bias', 'TIN.residual_layer2.0.weight', 'TIN.lstm.weight_hh_l1', 'lstm.weight_ih_l0', 'attention_layer.linear_q.bias', 'Sem_gcn.global_context.gate.weight', 'TIN.GatedGCN.conv3.rnn.bias_ih', 'attention_layer.w_query.bias', 'Sem_gcn.attn.linears.1.weight', 'Sem_gcn.global_context.fc.weight', 'TIN.residual_layer4.0.weight', 'boundary_refiner.fusion.weight', 'lstm.bias_hh_l1_reverse', 'TIN.lstm.weight_hh_l0', 'TIN.lstm.weight_ih_l1', 'TIN.feature_fusion.3.weight', 'TIN.feature_fusion.2.bias', 'lstm.weight_ih_l1', 'Sem_gcn.global_context.fc.bias', 'TIN.residual_layer2.3.bias', 'TIN.residual_layer3.3.bias', 'entity_classifier.weight', 'Syn_gcn.W.0.weight', 'Sem_gcn.multi_scale.fusion.weight', 'TIN.residual_layer2.2.weight', 'TIN.residual_layer4.0.bias', 'TIN.lstm.weight_ih_l0', 'contrastive_encoder.opinion_proj.bias', 'attention_layer.w_query.weight', 'TIN.lstm.weight_hh_l1_reverse', 'lstm.bias_ih_l1_reverse', 'attention_layer.w_value.weight', 'lstm.bias_ih_l0', 'TIN.lstm.bias_ih_l0', 'TIN.lstm.bias_ih_l0_reverse', 'Sem_gcn.multi_scale.fusion.bias', 'TIN.residual_layer1.2.weight', 'Sem_gcn.multi_scale.scale_weights', 'TIN.residual_layer4.2.bias', 'TIN.feature_fusion.0.bias', 'boundary_refiner.fusion.bias', 'Sem_gcn.attn.linears.0.weight', 'TIN.residual_layer1.3.bias', 'Sem_gcn.W.0.bias', 'TIN.residual_layer3.0.weight', 'contrastive_encoder.entity_proj.weight', 'Sem_gcn.attn.linears.0.bias', 'contrastive_encoder.entity_proj.bias', 'TIN.residual_layer3.3.weight', 'lstm.weight_ih_l1_reverse', 'TIN.lstm.bias_hh_l0_reverse', 'TIN.GatedGCN.conv1.lin.weight', 'TIN.lstm.bias_ih_l1_reverse', 'lstm.bias_hh_l0_reverse', 'TIN.lstm.weight_ih_l0_reverse', 'contrastive_encoder.opinion_proj.weight', 'Syn_gcn.W.1.bias', 'boundary_refiner.start_attention.weight', 'TIN.residual_layer3.2.weight', 'boundary_refiner.end_attention.weight', 'lstm.bias_hh_l1']
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
Train epoch 0: 100%|████████████████████████████| 79/79 [00:29<00:00,  2.70it/s]
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
Evaluate epoch 1: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.70it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        834.0
                   t         0.00         0.00         0.00        828.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

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
Train epoch 1: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
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
Evaluate epoch 2: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.96it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        834.0
                   t         0.00         0.00         0.00        828.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

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
Train epoch 2: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.90it/s]
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
Evaluate epoch 3: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.80it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        834.0
                   t         0.00         0.00         0.00        828.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

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
Train epoch 3: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.81it/s]
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
                   o         0.00         0.00         0.00        834.0
                   t         0.00         0.00         0.00        828.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

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
Train epoch 4: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.90it/s]
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
Evaluate epoch 5: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.73it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        834.0
                   t         0.00         0.00         0.00        828.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

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
Train epoch 5: 100%|████████████████████████████| 79/79 [00:26<00:00,  2.93it/s]
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
Evaluate epoch 6: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.76it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        834.0
                   t         0.00         0.00         0.00        828.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        762.0
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0

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
Train epoch 6: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.89it/s]
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
Evaluate epoch 7: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.37it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        54.35         6.00        10.80          834
                   t        53.03        21.14        30.22          828

               micro        53.32        13.54        21.59         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        22.90         3.94         6.72          762
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61

               micro        22.90         3.09         5.44          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        21.37         3.67         6.27          762
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61

               micro        21.37         2.88         5.08          971

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
Train epoch 7: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.88it/s]
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
Evaluate epoch 8: 100%|█████████████████████████| 30/30 [00:03<00:00,  8.49it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        66.86        42.09        51.66          834
                   t        61.00        60.63        60.81          828

               micro        63.28        51.32        56.68         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        33.08        34.12        33.59          762
                 NEG        50.00         8.78        14.94          148
                 NEU         0.00         0.00         0.00           61

               micro        33.62        28.12        30.62          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        32.32        33.33        32.82          762
                 NEG        50.00         8.78        14.94          148
                 NEU         0.00         0.00         0.00           61

               micro        32.88        27.50        29.95          971

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
Train epoch 8: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
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
Evaluate epoch 9: 100%|█████████████████████████| 30/30 [00:03<00:00,  8.51it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        67.67        72.78        70.13          834
                   t        65.82        65.82        65.82          828

               micro        66.78        69.31        68.02         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        42.41        53.54        47.33          762
                 NEG        57.14        21.62        31.37          148
                 NEU         0.00         0.00         0.00           61

               micro        43.22        45.31        44.24          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        42.10        53.15        46.98          762
                 NEG        57.14        21.62        31.37          148
                 NEU         0.00         0.00         0.00           61

               micro        42.93        45.01        43.94          971

2026-01-07 06:10:32.917910: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1767766233.110919     979 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1767766233.167468     979 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1767766233.664139     979 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1767766233.664180     979 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1767766233.664184     979 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1767766233.664188     979 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
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
Train epoch 9: 100%|████████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
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
Evaluate epoch 10: 100%|████████████████████████| 30/30 [00:03<00:00,  8.21it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        72.02        79.62        75.63          834
                   t        69.60        66.91        68.23          828

               micro        70.90        73.29        72.07         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        41.14        59.45        48.63          762
                 NEG        52.00        35.14        41.94          148
                 NEU         0.00         0.00         0.00           61

               micro        42.05        52.01        46.50          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        41.14        59.45        48.63          762
                 NEG        52.00        35.14        41.94          148
                 NEU         0.00         0.00         0.00           61

               micro        42.05        52.01        46.50          971

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
Train epoch 10: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
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
Evaluate epoch 11: 100%|████████████████████████| 30/30 [00:03<00:00,  8.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        76.64        81.06        78.79          834
                   t        71.82        68.96        70.36          828

               micro        74.36        75.03        74.69         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        49.67        59.97        54.34          762
                 NEG        50.41        41.22        45.35          148
                 NEU         0.00         0.00         0.00           61

               micro        49.76        53.35        51.49          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        49.67        59.97        54.34          762
                 NEG        50.41        41.22        45.35          148
                 NEU         0.00         0.00         0.00           61

               micro        49.76        53.35        51.49          971

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
Train epoch 11: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
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
Evaluate epoch 12: 100%|████████████████████████| 30/30 [00:03<00:00,  8.03it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        74.19        84.77        79.13          834
                   t        75.60        68.48        71.86          828

               micro        74.81        76.65        75.72         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        49.32        61.42        54.70          762
                 NEG        47.19        56.76        51.53          148
                 NEU         0.00         0.00         0.00           61

               micro        48.98        56.85        52.62          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        49.32        61.42        54.70          762
                 NEG        47.19        56.76        51.53          148
                 NEU         0.00         0.00         0.00           61

               micro        48.98        56.85        52.62          971

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
Train epoch 12: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
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
Evaluate epoch 13: 100%|████████████████████████| 30/30 [00:03<00:00,  8.26it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.16        81.89        81.02          834
                   t        79.51        67.03        72.74          828

               micro        79.87        74.49        77.09         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        65.17        56.96        60.78          762
                 NEG        47.67        55.41        51.25          148
                 NEU         0.00         0.00         0.00           61

               micro        61.58        53.14        57.05          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        65.17        56.96        60.78          762
                 NEG        47.67        55.41        51.25          148
                 NEU         0.00         0.00         0.00           61

               micro        61.58        53.14        57.05          971

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
Train epoch 13: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
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
Evaluate epoch 14: 100%|████████████████████████| 30/30 [00:03<00:00,  8.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.29        83.57        81.38          834
                   t        77.32        69.57        73.24          828

               micro        78.39        76.59        77.48         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        59.34        61.68        60.49          762
                 NEG        46.28        58.78        51.79          148
                 NEU         0.00         0.00         0.00           61

               micro        56.84        57.36        57.10          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        59.34        61.68        60.49          762
                 NEG        45.74        58.11        51.19          148
                 NEU         0.00         0.00         0.00           61

               micro        56.73        57.26        57.00          971

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
Train epoch 14: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
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
Evaluate epoch 15: 100%|████████████████████████| 30/30 [00:03<00:00,  8.26it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.81        82.49        82.15          834
                   t        79.28        69.32        73.97          828

               micro        80.64        75.93        78.22         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        64.11        60.24        62.11          762
                 NEG        47.16        56.08        51.23          148
                 NEU         0.00         0.00         0.00           61

               micro        60.76        55.82        58.19          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        64.11        60.24        62.11          762
                 NEG        47.16        56.08        51.23          148
                 NEU         0.00         0.00         0.00           61

               micro        60.76        55.82        58.19          971

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
Train epoch 15: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.88it/s]
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
Evaluate epoch 16: 100%|████████████████████████| 30/30 [00:03<00:00,  8.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.49        85.49        82.38          834
                   t        74.22        74.40        74.31          828

               micro        76.95        79.96        78.43         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        51.94        68.64        59.13          762
                 NEG        51.22        56.76        53.85          148
                 NEU         0.00         0.00         0.00           61

               micro        51.75        62.51        56.62          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        51.94        68.64        59.13          762
                 NEG        51.22        56.76        53.85          148
                 NEU         0.00         0.00         0.00           61

               micro        51.75        62.51        56.62          971

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
Train epoch 16: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
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
Evaluate epoch 17: 100%|████████████████████████| 30/30 [00:03<00:00,  8.10it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        76.24        88.49        81.91          834
                   t        72.27        75.85        74.01          828

               micro        74.36        82.19        78.08         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        52.46        69.95        59.96          762
                 NEG        48.69        62.84        54.87          148
                 NEU        33.33         1.64         3.12           61

               micro        51.82        64.57        57.50          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        52.46        69.95        59.96          762
                 NEG        48.69        62.84        54.87          148
                 NEU        33.33         1.64         3.12           61

               micro        51.82        64.57        57.50          971

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
Train epoch 17: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.88it/s]
Evaluate epoch 18:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 18: 100%|████████████████████████| 30/30 [00:03<00:00,  8.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.69        86.57        82.99          834
                   t        77.49        73.19        75.28          828

               micro        78.67        79.90        79.28         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        59.91        67.06        63.28          762
                 NEG        50.28        61.49        55.32          148
                 NEU        25.00         3.28         5.80           61

               micro        57.97        62.20        60.01          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        59.91        67.06        63.28          762
                 NEG        50.28        61.49        55.32          148
                 NEU        25.00         3.28         5.80           61

               micro        57.97        62.20        60.01          971

Train epoch 18:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 18: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 19:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 19: 100%|████████████████████████| 30/30 [00:03<00:00,  8.06it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.33        86.69        83.39          834
                   t        77.81        73.67        75.68          828

               micro        79.16        80.20        79.68         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        59.45        68.50        63.66          762
                 NEG        54.79        54.05        54.42          148
                 NEU        27.27         4.92         8.33           61

               micro        58.45        62.31        60.32          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        59.45        68.50        63.66          762
                 NEG        54.79        54.05        54.42          148
                 NEU        27.27         4.92         8.33           61

               micro        58.45        62.31        60.32          971

Train epoch 19:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 19: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 20:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 20: 100%|████████████████████████| 30/30 [00:03<00:00,  8.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.22        87.05        83.50          834
                   t        72.90        78.62        75.65          828

               micro        76.59        82.85        79.60         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        57.23        72.18        63.84          762
                 NEG        57.53        56.76        57.14          148
                 NEU        16.13         8.20        10.87           61

               micro        56.15        65.81        60.60          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        57.23        72.18        63.84          762
                 NEG        57.53        56.76        57.14          148
                 NEU        16.13         8.20        10.87           61

               micro        56.15        65.81        60.60          971

Train epoch 20:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 20: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 21:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 21: 100%|████████████████████████| 30/30 [00:03<00:00,  8.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.50        84.29        84.39          834
                   t        88.05        65.82        75.33          828

               micro        86.01        75.09        80.18         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        75.04        59.19        66.18          762
                 NEG        56.93        52.70        54.74          148
                 NEU        40.00         6.56        11.27           61

               micro        71.26        54.89        62.01          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        75.04        59.19        66.18          762
                 NEG        56.93        52.70        54.74          148
                 NEU        40.00         6.56        11.27           61

               micro        71.26        54.89        62.01          971

Train epoch 21:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 21: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.89it/s]
Evaluate epoch 22:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 22: 100%|████████████████████████| 30/30 [00:03<00:00,  8.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        77.87        88.61        82.89          834
                   t        72.75        80.92        76.62          828

               micro        75.35        84.78        79.78         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        57.58        73.75        64.67          762
                 NEG        51.43        60.81        55.73          148
                 NEU        19.23         8.20        11.49           61

               micro        55.82        67.66        61.17          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        57.58        73.75        64.67          762
                 NEG        51.43        60.81        55.73          148
                 NEU        19.23         8.20        11.49           61

               micro        55.82        67.66        61.17          971

Train epoch 22:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 22: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 23:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 23: 100%|████████████████████████| 30/30 [00:03<00:00,  8.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.61        88.49        83.82          834
                   t        78.34        75.12        76.70          828

               micro        79.02        81.83        80.40         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        61.57        69.16        65.14          762
                 NEG        51.18        58.78        54.72          148
                 NEU        23.53         6.56        10.26           61

               micro        59.25        63.65        61.37          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        61.57        69.16        65.14          762
                 NEG        51.18        58.78        54.72          148
                 NEU        23.53         6.56        10.26           61

               micro        59.25        63.65        61.37          971

Train epoch 23:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 23: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 24:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 24: 100%|████████████████████████| 30/30 [00:06<00:00,  4.94it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.00        87.41        84.62          834
                   t        75.66        79.59        77.58          828

               micro        78.86        83.51        81.12         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        62.18        72.70        67.03          762
                 NEG        52.87        62.16        57.14          148
                 NEU        25.81        13.11        17.39           61

               micro        59.67        67.35        63.28          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        62.18        72.70        67.03          762
                 NEG        52.30        61.49        56.52          148
                 NEU        25.81        13.11        17.39           61

               micro        59.58        67.25        63.18          971

Train epoch 24:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 24: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 25:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 25: 100%|████████████████████████| 30/30 [00:03<00:00,  8.19it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.76        85.97        84.85          834
                   t        82.24        72.71        77.18          828

               micro        83.06        79.36        81.17         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        71.29        66.14        68.62          762
                 NEG        55.78        55.41        55.59          148
                 NEU        23.08         9.84        13.79           61

               micro        67.27        60.97        63.97          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        71.29        66.14        68.62          762
                 NEG        55.78        55.41        55.59          148
                 NEU        23.08         9.84        13.79           61

               micro        67.27        60.97        63.97          971

Train epoch 25:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 25: 100%|███████████████████████████| 79/79 [00:28<00:00,  2.81it/s]
Evaluate epoch 26:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 26: 100%|████████████████████████| 30/30 [00:03<00:00,  8.12it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.50        88.37        83.70          834
                   t        76.86        78.62        77.73          828

               micro        78.24        83.51        80.79         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        57.80        73.88        64.86          762
                 NEG        53.45        62.84        57.76          148
                 NEU        17.65        14.75        16.07           61

               micro        55.46        68.49        61.29          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        57.80        73.88        64.86          762
                 NEG        53.45        62.84        57.76          148
                 NEU        17.65        14.75        16.07           61

               micro        55.46        68.49        61.29          971

Train epoch 26:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 26: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 27:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 27: 100%|████████████████████████| 30/30 [00:03<00:00,  8.14it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.78        87.17        84.39          834
                   t        80.10        75.85        77.92          828

               micro        80.99        81.53        81.26         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        68.52        69.69        69.10          762
                 NEG        60.74        55.41        57.95          148
                 NEU        22.50        14.75        17.82           61

               micro        65.47        64.06        64.76          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        68.52        69.69        69.10          762
                 NEG        60.74        55.41        57.95          148
                 NEU        22.50        14.75        17.82           61

               micro        65.47        64.06        64.76          971

Train epoch 27:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 27: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 28:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 28: 100%|████████████████████████| 30/30 [00:03<00:00,  8.12it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.05        87.17        84.00          834
                   t        74.59        81.52        77.90          828

               micro        77.80        84.36        80.95         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        61.31        73.62        66.91          762
                 NEG        49.47        62.84        55.36          148
                 NEU        17.02        13.11        14.81           61

               micro        57.57        68.18        62.42          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        61.31        73.62        66.91          762
                 NEG        48.94        62.16        54.76          148
                 NEU        17.02        13.11        14.81           61

               micro        57.48        68.07        62.33          971

Train epoch 28:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 28: 100%|███████████████████████████| 79/79 [00:28<00:00,  2.82it/s]
Evaluate epoch 29:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 29: 100%|████████████████████████| 30/30 [00:03<00:00,  8.04it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.63        88.37        84.32          834
                   t        74.22        83.09        78.40          828

               micro        77.40        85.74        81.36         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        60.35        76.90        67.63          762
                 NEG        52.51        63.51        57.49          148
                 NEU        19.05        13.11        15.53           61

               micro        57.72        70.85        63.62          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        60.35        76.90        67.63          762
                 NEG        52.51        63.51        57.49          148
                 NEU        19.05        13.11        15.53           61

               micro        57.72        70.85        63.62          971

Train epoch 29:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 29: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 30:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 30: 100%|████████████████████████| 30/30 [00:03<00:00,  8.00it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.49        88.13        84.68          834
                   t        74.95        82.73        78.65          828

               micro        78.19        85.44        81.66         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        58.76        76.12        66.32          762
                 NEG        51.12        61.49        55.83          148
                 NEU        22.22        13.11        16.49           61

               micro        56.54        69.93        62.52          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        58.76        76.12        66.32          762
                 NEG        51.12        61.49        55.83          148
                 NEU        22.22        13.11        16.49           61

               micro        56.54        69.93        62.52          971

Train epoch 30:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 30: 100%|███████████████████████████| 79/79 [00:28<00:00,  2.82it/s]
Evaluate epoch 31:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 31: 100%|████████████████████████| 30/30 [00:03<00:00,  8.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.99        84.65        85.32          834
                   t        81.75        75.72        78.62          828

               micro        83.94        80.20        82.03         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.85        68.90        69.86          762
                 NEG        63.78        54.73        58.91          148
                 NEU        35.29         9.84        15.38           61

               micro        69.15        63.03        65.95          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.85        68.90        69.86          762
                 NEG        63.78        54.73        58.91          148
                 NEU        35.29         9.84        15.38           61

               micro        69.15        63.03        65.95          971

Train epoch 31:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 31: 100%|███████████████████████████| 79/79 [00:28<00:00,  2.81it/s]
Evaluate epoch 32:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 32: 100%|████████████████████████| 30/30 [00:03<00:00,  8.06it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        78.59        88.01        83.03          834
                   t        74.43        83.33        78.63          828

               micro        76.52        85.68        80.84         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        55.63        77.17        64.65          762
                 NEG        50.55        62.16        55.76          148
                 NEU        16.98        14.75        15.79           61

               micro        53.33        70.96        60.89          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        55.63        77.17        64.65          762
                 NEG        50.55        62.16        55.76          148
                 NEU        16.98        14.75        15.79           61

               micro        53.33        70.96        60.89          971

Train epoch 32:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 32: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 33:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 33: 100%|████████████████████████| 30/30 [00:03<00:00,  8.21it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.44        88.25        84.16          834
                   t        85.80        71.50        78.00          828

               micro        82.74        79.90        81.30         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        71.51        66.54        68.93          762
                 NEG        55.41        58.78        57.05          148
                 NEU        26.92        11.48        16.09           61

               micro        67.38        61.89        64.52          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        71.51        66.54        68.93          762
                 NEG        54.78        58.11        56.39          148
                 NEU        26.92        11.48        16.09           61

               micro        67.26        61.79        64.41          971

Train epoch 33:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 33: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 34:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 34: 100%|████████████████████████| 30/30 [00:03<00:00,  8.18it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.13        88.49        84.10          834
                   t        75.14        83.94        79.29          828

               micro        77.63        86.22        81.70         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        64.12        77.17        70.04          762
                 NEG        52.38        59.46        55.70          148
                 NEU        19.35         9.84        13.04           61

               micro        61.11        70.24        65.36          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        64.12        77.17        70.04          762
                 NEG        52.38        59.46        55.70          148
                 NEU        19.35         9.84        13.04           61

               micro        61.11        70.24        65.36          971

Train epoch 34:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 34: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 35:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 35: 100%|████████████████████████| 30/30 [00:03<00:00,  8.19it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        74.83        90.53        81.93          834
                   t        79.33        79.71        79.52          828

               micro        76.86        85.14        80.79         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        63.16        74.02        68.16          762
                 NEG        48.73        64.86        55.65          148
                 NEU        22.22         9.84        13.64           61

               micro        59.62        68.59        63.79          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        63.16        74.02        68.16          762
                 NEG        48.22        64.19        55.07          148
                 NEU        22.22         9.84        13.64           61

               micro        59.53        68.49        63.70          971

Train epoch 35:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 35: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 36:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 36: 100%|████████████████████████| 30/30 [00:03<00:00,  8.16it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.33        88.25        84.65          834
                   t        81.41        78.26        79.80          828

               micro        81.36        83.27        82.31         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.81        72.70        70.17          762
                 NEG        54.55        56.76        55.63          148
                 NEU        24.24        13.11        17.02           61

               micro        64.34        66.53        65.42          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.81        72.70        70.17          762
                 NEG        53.90        56.08        54.97          148
                 NEU        24.24        13.11        17.02           61

               micro        64.24        66.43        65.32          971

Train epoch 36:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 36: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 37:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 37: 100%|████████████████████████| 30/30 [00:03<00:00,  8.26it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.98        88.85        84.73          834
                   t        81.64        76.81        79.15          828

               micro        81.29        82.85        82.06         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        66.59        71.65        69.03          762
                 NEG        54.37        58.78        56.49          148
                 NEU        29.03        14.75        19.57           61

               micro        63.50        66.12        64.78          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        66.59        71.65        69.03          762
                 NEG        54.37        58.78        56.49          148
                 NEU        29.03        14.75        19.57           61

               micro        63.50        66.12        64.78          971

Train epoch 37:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 37: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 38:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 38: 100%|████████████████████████| 30/30 [00:03<00:00,  8.15it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        78.93        89.81        84.02          834
                   t        80.55        78.50        79.51          828

               micro        79.67        84.18        81.86         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        63.79        73.75        68.41          762
                 NEG        54.60        60.14        57.23          148
                 NEU        16.67         9.84        12.37           61

               micro        60.83        67.66        64.07          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        63.79        73.75        68.41          762
                 NEG        54.60        60.14        57.23          148
                 NEU        16.67         9.84        12.37           61

               micro        60.83        67.66        64.07          971

Train epoch 38:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 38: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 39:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 39: 100%|████████████████████████| 30/30 [00:03<00:00,  8.12it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.35        87.05        85.16          834
                   t        77.75        81.04        79.36          828

               micro        80.57        84.06        82.27         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.07        73.23        70.01          762
                 NEG        56.33        60.14        58.17          148
                 NEU        26.09         9.84        14.29           61

               micro        64.46        67.25        65.83          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.07        73.23        70.01          762
                 NEG        55.70        59.46        57.52          148
                 NEU        26.09         9.84        14.29           61

               micro        64.36        67.15        65.73          971

Train epoch 39:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 39: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 40:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 40: 100%|████████████████████████| 30/30 [00:03<00:00,  8.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.43        87.53        85.43          834
                   t        79.95        79.95        79.95          828

               micro        81.74        83.75        82.73         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.23        73.23        70.10          762
                 NEG        57.43        57.43        57.43          148
                 NEU        26.67        13.11        17.58           61

               micro        64.58        67.04        65.79          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.23        73.23        70.10          762
                 NEG        57.43        57.43        57.43          148
                 NEU        26.67        13.11        17.58           61

               micro        64.58        67.04        65.79          971

Train epoch 40:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 40: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 41:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 41: 100%|████████████████████████| 30/30 [00:03<00:00,  8.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.85        87.17        85.48          834
                   t        80.12        79.83        79.98          828

               micro        82.03        83.51        82.77         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.10        74.15        70.45          762
                 NEG        57.33        58.11        57.72          148
                 NEU        22.58        11.48        15.22           61

               micro        64.32        67.77        66.00          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.10        74.15        70.45          762
                 NEG        57.33        58.11        57.72          148
                 NEU        22.58        11.48        15.22           61

               micro        64.32        67.77        66.00          971

Train epoch 41:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 41: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 42:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 42: 100%|████████████████████████| 30/30 [00:03<00:00,  8.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.62        86.45        85.53          834
                   t        81.23        77.90        79.53          828

               micro        82.99        82.19        82.59         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.66        71.39        69.48          762
                 NEG        60.56        58.11        59.31          148
                 NEU        29.17        11.48        16.47           61

               micro        65.67        65.60        65.64          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.66        71.39        69.48          762
                 NEG        60.56        58.11        59.31          148
                 NEU        29.17        11.48        16.47           61

               micro        65.67        65.60        65.64          971

Train epoch 42:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 42: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 43:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 43: 100%|████████████████████████| 30/30 [00:03<00:00,  8.34it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.49        86.21        85.34          834
                   t        82.31        77.54        79.85          828

               micro        83.45        81.89        82.66         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        71.62        70.87        71.24          762
                 NEG        62.79        54.73        58.48          148
                 NEU        25.00        14.75        18.56           61

               micro        68.55        64.88        66.67          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        71.62        70.87        71.24          762
                 NEG        62.79        54.73        58.48          148
                 NEU        25.00        14.75        18.56           61

               micro        68.55        64.88        66.67          971

Train epoch 43:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 43: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 44:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 44: 100%|████████████████████████| 30/30 [00:03<00:00,  8.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.44        86.57        85.49          834
                   t        78.80        82.61        80.66          828

               micro        81.60        84.60        83.07         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.56        75.72        71.41          762
                 NEG        63.16        56.76        59.79          148
                 NEU        27.78         8.20        12.66           61

               micro        66.27        68.59        67.41          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.56        75.72        71.41          762
                 NEG        63.16        56.76        59.79          148
                 NEU        27.78         8.20        12.66           61

               micro        66.27        68.59        67.41          971

Train epoch 44:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 44: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 45:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 45: 100%|████████████████████████| 30/30 [00:03<00:00,  8.13it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.95        87.17        85.53          834
                   t        82.73        76.93        79.72          828

               micro        83.37        82.07        82.72         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        68.55        71.52        70.01          762
                 NEG        64.46        52.70        57.99          148
                 NEU        11.54         4.92         6.90           61

               micro        66.45        64.47        65.45          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        68.55        71.52        70.01          762
                 NEG        64.46        52.70        57.99          148
                 NEU        11.54         4.92         6.90           61

               micro        66.45        64.47        65.45          971

Train epoch 45:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 45: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 46:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 46: 100%|████████████████████████| 30/30 [00:03<00:00,  8.00it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.97        87.77        84.77          834
                   t        75.57        84.78        79.91          828

               micro        78.70        86.28        82.32         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        61.18        79.00        68.96          762
                 NEG        55.15        61.49        58.15          148
                 NEU        17.14         9.84        12.50           61

               micro        59.04        71.99        64.87          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        61.18        79.00        68.96          762
                 NEG        55.15        61.49        58.15          148
                 NEU        17.14         9.84        12.50           61

               micro        59.04        71.99        64.87          971

Train epoch 46:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 46: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 47:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 47: 100%|████████████████████████| 30/30 [00:03<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.41        87.65        84.95          834
                   t        74.87        84.90        79.57          828

               micro        78.53        86.28        82.22         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        63.26        77.95        69.84          762
                 NEG        56.17        61.49        58.71          148
                 NEU        16.22         9.84        12.24           61

               micro        60.72        71.16        65.53          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        63.26        77.95        69.84          762
                 NEG        56.17        61.49        58.71          148
                 NEU        16.22         9.84        12.24           61

               micro        60.72        71.16        65.53          971

Train epoch 47:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 47: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 48:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 48: 100%|████████████████████████| 30/30 [00:03<00:00,  8.08it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.31        87.41        85.31          834
                   t        74.34        85.39        79.48          828

               micro        78.64        86.40        82.34         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        59.94        79.13        68.21          762
                 NEG        58.87        56.08        57.44          148
                 NEU        17.95        11.48        14.00           61

               micro        58.43        71.37        64.26          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        59.94        79.13        68.21          762
                 NEG        58.87        56.08        57.44          148
                 NEU        17.95        11.48        14.00           61

               micro        58.43        71.37        64.26          971

Train epoch 48:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 48: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.90it/s]
Evaluate epoch 49:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 49: 100%|████████████████████████| 30/30 [00:03<00:00,  7.96it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.15        88.61        84.17          834
                   t        77.89        82.13        79.95          828

               micro        79.05        85.38        82.09         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.26        74.93        70.89          762
                 NEG        57.67        63.51        60.45          148
                 NEU        18.42        11.48        14.14           61

               micro        64.00        69.21        66.50          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.26        74.93        70.89          762
                 NEG        57.06        62.84        59.81          148
                 NEU        18.42        11.48        14.14           61

               micro        63.90        69.10        66.40          971

Train epoch 49:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 49: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.82it/s]
Evaluate epoch 50:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 50: 100%|████████████████████████| 30/30 [00:03<00:00,  8.14it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.22        89.45        84.58          834
                   t        80.83        79.95        80.39          828

               micro        80.50        84.72        82.56         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.18        74.41        70.61          762
                 NEG        56.88        61.49        59.09          148
                 NEU        24.14        11.48        15.56           61

               micro        64.38        68.49        66.37          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.18        74.41        70.61          762
                 NEG        56.25        60.81        58.44          148
                 NEU        24.14        11.48        15.56           61

               micro        64.28        68.38        66.27          971

Train epoch 50:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 50: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.89it/s]
Evaluate epoch 51:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 51: 100%|████████████████████████| 30/30 [00:03<00:00,  8.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.29        86.21        85.24          834
                   t        76.37        84.30        80.14          828

               micro        80.19        85.26        82.65         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        65.80        76.51        70.75          762
                 NEG        59.03        57.43        58.22          148
                 NEU        24.00         9.84        13.95           61

               micro        63.89        69.41        66.54          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        65.80        76.51        70.75          762
                 NEG        59.03        57.43        58.22          148
                 NEU        24.00         9.84        13.95           61

               micro        63.89        69.41        66.54          971

Train epoch 51:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 51: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.88it/s]
Evaluate epoch 52:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 52: 100%|████████████████████████| 30/30 [00:03<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.51        86.33        85.41          834
                   t        77.00        83.70        80.21          828

               micro        80.65        85.02        82.78         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.01        76.51        71.45          762
                 NEG        61.15        57.43        59.23          148
                 NEU        25.71        14.75        18.75           61

               micro        64.85        69.72        67.20          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.01        76.51        71.45          762
                 NEG        61.15        57.43        59.23          148
                 NEU        25.71        14.75        18.75           61

               micro        64.85        69.72        67.20          971

Train epoch 52:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 52: 100%|███████████████████████████| 79/79 [00:28<00:00,  2.79it/s]
Evaluate epoch 53:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 53: 100%|████████████████████████| 30/30 [00:03<00:00,  8.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.07        88.37        85.10          834
                   t        76.00        84.54        80.05          828

               micro        79.00        86.46        82.56         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        64.94        77.30        70.58          762
                 NEG        55.03        62.84        58.68          148
                 NEU        15.25        14.75        15.00           61

               micro        60.88        71.16        65.62          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        64.94        77.30        70.58          762
                 NEG        54.44        62.16        58.04          148
                 NEU        15.25        14.75        15.00           61

               micro        60.79        71.06        65.53          971

Train epoch 53:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 53: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 54:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 54: 100%|████████████████████████| 30/30 [00:03<00:00,  8.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.28        88.49        84.73          834
                   t        75.19        85.27        79.91          828

               micro        78.18        86.88        82.30         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        63.45        78.61        70.22          762
                 NEG        55.15        61.49        58.15          148
                 NEU        21.43         9.84        13.48           61

               micro        61.21        71.68        66.03          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        63.45        78.61        70.22          762
                 NEG        55.15        61.49        58.15          148
                 NEU        21.43         9.84        13.48           61

               micro        61.21        71.68        66.03          971

Train epoch 54:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 54: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 55:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 55: 100%|████████████████████████| 30/30 [00:03<00:00,  8.11it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.46        85.25        85.35          834
                   t        84.01        78.02        80.90          828

               micro        84.76        81.65        83.17         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.68        70.87        70.77          762
                 NEG        60.74        55.41        57.95          148
                 NEU        23.68        14.75        18.18           61

               micro        67.34        64.98        66.14          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.68        70.87        70.77          762
                 NEG        60.74        55.41        57.95          148
                 NEU        23.68        14.75        18.18           61

               micro        67.34        64.98        66.14          971

Train epoch 55:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 55: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 56:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 56: 100%|████████████████████████| 30/30 [00:03<00:00,  8.32it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.35        86.45        84.87          834
                   t        83.91        76.81        80.20          828

               micro        83.61        81.65        82.62         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        72.06        70.08        71.06          762
                 NEG        60.31        53.38        56.63          148
                 NEU        24.14        11.48        15.56           61

               micro        68.81        63.85        66.24          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        72.06        70.08        71.06          762
                 NEG        60.31        53.38        56.63          148
                 NEU        24.14        11.48        15.56           61

               micro        68.81        63.85        66.24          971

Train epoch 56:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 56: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 57:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 57: 100%|████████████████████████| 30/30 [00:03<00:00,  8.09it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.74        89.69        84.42          834
                   t        76.10        85.39        80.48          828

               micro        77.93        87.55        82.46         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        65.39        78.61        71.39          762
                 NEG        56.47        64.86        60.38          148
                 NEU        15.15         8.20        10.64           61

               micro        62.56        72.09        66.99          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        65.39        78.61        71.39          762
                 NEG        56.47        64.86        60.38          148
                 NEU        15.15         8.20        10.64           61

               micro        62.56        72.09        66.99          971

Train epoch 57:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 57: 100%|███████████████████████████| 79/79 [00:28<00:00,  2.82it/s]
Evaluate epoch 58:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 58: 100%|████████████████████████| 30/30 [00:03<00:00,  8.13it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.85        88.13        84.87          834
                   t        75.56        85.51        80.23          828

               micro        78.64        86.82        82.53         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        64.01        77.95        70.30          762
                 NEG        57.41        62.84        60.00          148
                 NEU        22.22         9.84        13.64           61

               micro        62.04        71.37        66.38          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        64.01        77.95        70.30          762
                 NEG        57.41        62.84        60.00          148
                 NEU        22.22         9.84        13.64           61

               micro        62.04        71.37        66.38          971

Train epoch 58:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 58: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 59:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 59: 100%|████████████████████████| 30/30 [00:03<00:00,  8.10it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.20        88.61        85.29          834
                   t        77.68        84.06        80.74          828

               micro        79.94        86.34        83.02         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        65.03        78.08        70.96          762
                 NEG        62.14        58.78        60.42          148
                 NEU        18.75         9.84        12.90           61

               micro        63.29        70.85        66.86          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        65.03        78.08        70.96          762
                 NEG        62.14        58.78        60.42          148
                 NEU        18.75         9.84        12.90           61

               micro        63.29        70.85        66.86          971

Train epoch 59:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 59: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 60:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 60: 100%|████████████████████████| 30/30 [00:03<00:00,  8.39it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.26        87.65        85.40          834
                   t        82.68        81.28        81.97          828

               micro        82.98        84.48        83.72         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.47        74.54        72.45          762
                 NEG        63.57        55.41        59.21          148
                 NEU        24.32        14.75        18.37           61

               micro        67.80        67.87        67.83          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.47        74.54        72.45          762
                 NEG        63.57        55.41        59.21          148
                 NEU        24.32        14.75        18.37           61

               micro        67.80        67.87        67.83          971

Train epoch 60:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 60: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.89it/s]
Evaluate epoch 61:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 61: 100%|████████████████████████| 30/30 [00:03<00:00,  8.09it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.84        88.01        85.35          834
                   t        77.54        84.66        80.95          828

               micro        80.17        86.34        83.14         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        66.82        77.69        71.84          762
                 NEG        61.59        57.43        59.44          148
                 NEU        18.75        14.75        16.51           61

               micro        63.99        70.65        67.16          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        66.82        77.69        71.84          762
                 NEG        61.59        57.43        59.44          148
                 NEU        18.75        14.75        16.51           61

               micro        63.99        70.65        67.16          971

Train epoch 61:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 61: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.89it/s]
Evaluate epoch 62:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 62: 100%|████████████████████████| 30/30 [00:03<00:00,  8.11it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.20        87.89        85.48          834
                   t        74.71        86.35        80.11          828

               micro        78.78        87.12        82.74         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        63.66        79.53        70.71          762
                 NEG        59.21        60.81        60.00          148
                 NEU        18.75         9.84        12.90           61

               micro        61.80        72.30        66.64          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        63.66        79.53        70.71          762
                 NEG        59.21        60.81        60.00          148
                 NEU        18.75         9.84        12.90           61

               micro        61.80        72.30        66.64          971

Train epoch 62:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 62: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 63:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 63: 100%|████████████████████████| 30/30 [00:03<00:00,  8.21it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.16        88.37        85.15          834
                   t        77.70        84.18        80.81          828

               micro        79.93        86.28        82.99         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        66.97        77.17        71.71          762
                 NEG        59.21        60.81        60.00          148
                 NEU        16.28        11.48        13.46           61

               micro        63.84        70.55        67.03          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        66.97        77.17        71.71          762
                 NEG        59.21        60.81        60.00          148
                 NEU        16.28        11.48        13.46           61

               micro        63.84        70.55        67.03          971

Train epoch 63:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 63: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 64:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 64: 100%|████████████████████████| 30/30 [00:03<00:00,  8.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.82        89.57        85.52          834
                   t        78.91        84.06        81.40          828

               micro        80.39        86.82        83.48         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        65.70        77.69        71.20          762
                 NEG        62.99        65.54        64.24          148
                 NEU        19.51        13.11        15.69           61

               micro        63.59        71.78        67.44          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        65.70        77.69        71.20          762
                 NEG        62.99        65.54        64.24          148
                 NEU        19.51        13.11        15.69           61

               micro        63.59        71.78        67.44          971

Train epoch 64:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 64: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.88it/s]
Evaluate epoch 65:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 65: 100%|████████████████████████| 30/30 [00:03<00:00,  8.09it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.48        89.21        85.71          834
                   t        78.40        84.18        81.19          828

               micro        80.46        86.70        83.46         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        66.93        77.30        71.74          762
                 NEG        59.52        67.57        63.29          148
                 NEU        24.14        11.48        15.56           61

               micro        64.62        71.68        67.97          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        66.93        77.30        71.74          762
                 NEG        59.52        67.57        63.29          148
                 NEU        24.14        11.48        15.56           61

               micro        64.62        71.68        67.97          971

Train epoch 65:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 65: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 66:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 66: 100%|████████████████████████| 30/30 [00:03<00:00,  8.21it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.98        85.49        85.24          834
                   t        77.95        84.54        81.11          828

               micro        81.35        85.02        83.14         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        66.63        76.77        71.34          762
                 NEG        61.87        58.11        59.93          148
                 NEU        26.67        13.11        17.58           61

               micro        64.85        69.93        67.29          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        66.63        76.77        71.34          762
                 NEG        61.87        58.11        59.93          148
                 NEU        26.67        13.11        17.58           61

               micro        64.85        69.93        67.29          971

Train epoch 66:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 66: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 67:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 67: 100%|████████████████████████| 30/30 [00:03<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.69        88.61        86.08          834
                   t        81.16        82.73        81.94          828

               micro        82.46        85.68        84.04         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.22        76.12        73.05          762
                 NEG        62.75        64.86        63.79          148
                 NEU        18.75        14.75        16.51           61

               micro        66.70        70.55        68.57          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.22        76.12        73.05          762
                 NEG        62.75        64.86        63.79          148
                 NEU        18.75        14.75        16.51           61

               micro        66.70        70.55        68.57          971

Train epoch 67:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 67: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 68:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 68: 100%|████████████████████████| 30/30 [00:03<00:00,  8.02it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.90        88.37        85.55          834
                   t        79.03        84.66        81.75          828

               micro        80.97        86.52        83.65         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.12        77.69        72.02          762
                 NEG        61.59        62.84        62.21          148
                 NEU        19.51        13.11        15.69           61

               micro        64.53        71.37        67.78          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.12        77.69        72.02          762
                 NEG        61.59        62.84        62.21          148
                 NEU        19.51        13.11        15.69           61

               micro        64.53        71.37        67.78          971

Train epoch 68:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 68: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 69:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 69: 100%|████████████████████████| 30/30 [00:03<00:00,  8.20it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.15        88.73        85.85          834
                   t        79.45        84.06        81.69          828

               micro        81.31        86.40        83.78         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        68.45        77.17        72.55          762
                 NEG        60.13        64.19        62.09          148
                 NEU        20.00        11.48        14.58           61

               micro        65.59        71.06        68.22          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        68.45        77.17        72.55          762
                 NEG        60.13        64.19        62.09          148
                 NEU        20.00        11.48        14.58           61

               micro        65.59        71.06        68.22          971

Train epoch 69:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 69: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 70:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 70: 100%|████████████████████████| 30/30 [00:03<00:00,  8.18it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.26        88.25        85.68          834
                   t        81.44        83.21        82.32          828

               micro        82.37        85.74        84.02         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        69.52        76.64        72.91          762
                 NEG        64.38        63.51        63.95          148
                 NEU        21.43         9.84        13.48           61

               micro        67.46        70.44        68.92          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        69.52        76.64        72.91          762
                 NEG        63.70        62.84        63.27          148
                 NEU        21.43         9.84        13.48           61

               micro        67.36        70.34        68.82          971

Train epoch 70:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 70: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 71:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 71: 100%|████████████████████████| 30/30 [00:03<00:00,  8.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.82        89.45        84.92          834
                   t        80.92        82.49        81.70          828

               micro        80.87        85.98        83.35         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        68.52        75.98        72.06          762
                 NEG        52.88        68.24        59.59          148
                 NEU        18.92        11.48        14.29           61

               micro        64.03        70.75        67.22          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        68.52        75.98        72.06          762
                 NEG        52.36        67.57        59.00          148
                 NEU        18.92        11.48        14.29           61

               micro        63.93        70.65        67.12          971

Train epoch 71:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 71: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 72:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 72: 100%|████████████████████████| 30/30 [00:03<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.47        87.77        85.56          834
                   t        75.67        85.27        80.18          828

               micro        79.45        86.52        82.83         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        63.54        79.13        70.49          762
                 NEG        64.34        62.16        63.23          148
                 NEU        14.71         8.20        10.53           61

               micro        62.17        72.09        66.76          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        63.54        79.13        70.49          762
                 NEG        64.34        62.16        63.23          148
                 NEU        14.71         8.20        10.53           61

               micro        62.17        72.09        66.76          971

Train epoch 72:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 72: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 73:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 73: 100%|████████████████████████| 30/30 [00:03<00:00,  8.32it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.07        88.61        86.28          834
                   t        84.61        80.31        82.40          828

               micro        84.32        84.48        84.40         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        72.66        74.28        73.46          762
                 NEG        63.70        62.84        63.27          148
                 NEU        22.22         9.84        13.64           61

               micro        69.85        68.49        69.16          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        72.66        74.28        73.46          762
                 NEG        63.70        62.84        63.27          148
                 NEU        22.22         9.84        13.64           61

               micro        69.85        68.49        69.16          971

Train epoch 73:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 73: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.82it/s]
Evaluate epoch 74:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 74: 100%|████████████████████████| 30/30 [00:03<00:00,  8.24it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.62        88.73        86.10          834
                   t        83.33        80.92        82.11          828

               micro        83.48        84.84        84.15         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        72.26        74.54        73.39          762
                 NEG        65.96        62.84        64.36          148
                 NEU        20.69         9.84        13.33           61

               micro        69.77        68.69        69.23          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        72.26        74.54        73.39          762
                 NEG        65.96        62.84        64.36          148
                 NEU        20.69         9.84        13.33           61

               micro        69.77        68.69        69.23          971

Train epoch 74:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 74: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 75:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 75: 100%|████████████████████████| 30/30 [00:03<00:00,  7.97it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.54        88.25        85.83          834
                   t        79.37        84.54        81.87          828

               micro        81.45        86.40        83.85         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.85        77.56        72.38          762
                 NEG        65.07        64.19        64.63          148
                 NEU        20.00         9.84        13.19           61

               micro        66.09        71.27        68.58          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.85        77.56        72.38          762
                 NEG        65.07        64.19        64.63          148
                 NEU        20.00         9.84        13.19           61

               micro        66.09        71.27        68.58          971

Train epoch 75:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 75: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 76:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 76: 100%|████████████████████████| 30/30 [00:03<00:00,  8.13it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.94        89.21        85.96          834
                   t        81.81        83.09        82.44          828

               micro        82.39        86.16        84.24         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        69.65        76.51        72.92          762
                 NEG        62.99        65.54        64.24          148
                 NEU        18.18         9.84        12.77           61

               micro        66.99        70.65        68.77          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        69.65        76.51        72.92          762
                 NEG        62.34        64.86        63.58          148
                 NEU        18.18         9.84        12.77           61

               micro        66.89        70.55        68.67          971

Train epoch 76:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 76: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 77:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 77: 100%|████████████████████████| 30/30 [00:05<00:00,  5.18it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.41        88.01        85.65          834
                   t        82.19        82.49        82.34          828

               micro        82.82        85.26        84.02         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        72.80        74.80        73.79          762
                 NEG        62.75        64.86        63.79          148
                 NEU        16.67         8.20        10.99           61

               micro        69.46        69.10        69.28          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        72.80        74.80        73.79          762
                 NEG        62.75        64.86        63.79          148
                 NEU        16.67         8.20        10.99           61

               micro        69.46        69.10        69.28          971

Train epoch 77:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 77: 100%|███████████████████████████| 79/79 [00:28<00:00,  2.81it/s]
Evaluate epoch 78:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 78: 100%|████████████████████████| 30/30 [00:03<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.65        89.09        85.75          834
                   t        75.19        85.63        80.07          828

               micro        78.83        87.36        82.88         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        64.20        78.61        70.68          762
                 NEG        61.78        65.54        63.61          148
                 NEU        14.63         9.84        11.76           61

               micro        62.07        72.30        66.79          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        64.20        78.61        70.68          762
                 NEG        61.78        65.54        63.61          148
                 NEU        14.63         9.84        11.76           61

               micro        62.07        72.30        66.79          971

Train epoch 78:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 78: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 79:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 79: 100%|████████████████████████| 30/30 [00:03<00:00,  8.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.31        88.61        85.88          834
                   t        82.68        81.88        82.28          828

               micro        83.01        85.26        84.12         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        71.68        75.07        73.33          762
                 NEG        60.87        66.22        63.43          148
                 NEU        23.53        13.11        16.84           61

               micro        68.28        69.82        69.04          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        71.68        75.07        73.33          762
                 NEG        60.25        65.54        62.78          148
                 NEU        23.53        13.11        16.84           61

               micro        68.18        69.72        68.94          971

Train epoch 79:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 79: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 80:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 80: 100%|████████████████████████| 30/30 [00:03<00:00,  8.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.50        89.33        85.78          834
                   t        81.06        83.21        82.12          828

               micro        81.80        86.28        83.98         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        69.07        76.77        72.72          762
                 NEG        60.98        67.57        64.10          148
                 NEU        23.08         9.84        13.79           61

               micro        66.63        71.16        68.82          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        69.07        76.77        72.72          762
                 NEG        60.37        66.89        63.46          148
                 NEU        23.08         9.84        13.79           61

               micro        66.54        71.06        68.73          971

Train epoch 80:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 80: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 81:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 81: 100%|████████████████████████| 30/30 [00:03<00:00,  8.00it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.03        90.17        85.36          834
                   t        75.77        86.11        80.61          828

               micro        78.38        88.15        82.98         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        65.06        79.66        71.62          762
                 NEG        58.14        67.57        62.50          148
                 NEU        13.95         9.84        11.54           61

               micro        62.11        73.43        67.30          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        65.06        79.66        71.62          762
                 NEG        58.14        67.57        62.50          148
                 NEU        13.95         9.84        11.54           61

               micro        62.11        73.43        67.30          971

Train epoch 81:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 81: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 82:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 82: 100%|████████████████████████| 30/30 [00:03<00:00,  8.02it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.48        88.49        85.91          834
                   t        82.16        83.45        82.80          828

               micro        82.84        85.98        84.38         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        69.85        75.72        72.67          762
                 NEG        64.67        65.54        65.10          148
                 NEU        18.52         8.20        11.36           61

               micro        67.70        69.93        68.79          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        69.85        75.72        72.67          762
                 NEG        64.67        65.54        65.10          148
                 NEU        18.52         8.20        11.36           61

               micro        67.70        69.93        68.79          971

Train epoch 82:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 82: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 83:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 83: 100%|████████████████████████| 30/30 [00:03<00:00,  7.80it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.80        90.05        86.27          834
                   t        78.20        84.90        81.41          828

               micro        80.51        87.48        83.85         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        66.63        78.87        72.24          762
                 NEG        63.23        66.22        64.69          148
                 NEU        18.92        11.48        14.29           61

               micro        64.53        72.71        68.38          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        66.63        78.87        72.24          762
                 NEG        63.23        66.22        64.69          148
                 NEU        18.92        11.48        14.29           61

               micro        64.53        72.71        68.38          971

Train epoch 83:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 83: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 84:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 84: 100%|████████████████████████| 30/30 [00:03<00:00,  7.95it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.39        90.89        86.43          834
                   t        80.21        83.70        81.91          828

               micro        81.33        87.30        84.21         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        69.21        78.48        73.55          762
                 NEG        62.80        69.59        66.03          148
                 NEU        21.21        11.48        14.89           61

               micro        66.73        72.91        69.69          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        69.21        78.48        73.55          762
                 NEG        62.20        68.92        65.38          148
                 NEU        21.21        11.48        14.89           61

               micro        66.64        72.81        69.59          971

Train epoch 84:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 84: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 85:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 85: 100%|████████████████████████| 30/30 [00:03<00:00,  8.04it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.18        91.25        86.48          834
                   t        81.40        84.06        82.71          828

               micro        81.81        87.67        84.64         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        68.85        78.61        73.41          762
                 NEG        62.50        67.57        64.94          148
                 NEU        20.00        11.48        14.58           61

               micro        66.29        72.71        69.35          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        68.85        78.61        73.41          762
                 NEG        62.50        67.57        64.94          148
                 NEU        20.00        11.48        14.58           61

               micro        66.29        72.71        69.35          971

Train epoch 85:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 85: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 86:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 86: 100%|████████████████████████| 30/30 [00:03<00:00,  7.91it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.78        89.93        86.21          834
                   t        81.29        83.94        82.59          828

               micro        82.06        86.94        84.43         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        68.87        77.82        73.07          762
                 NEG        63.52        68.24        65.80          148
                 NEU        20.59        11.48        14.74           61

               micro        66.51        72.19        69.23          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        68.87        77.82        73.07          762
                 NEG        63.52        68.24        65.80          148
                 NEU        20.59        11.48        14.74           61

               micro        66.51        72.19        69.23          971

Train epoch 86:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 86: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 87:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 87: 100%|████████████████████████| 30/30 [00:03<00:00,  8.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.26        89.45        86.24          834
                   t        80.51        83.82        82.13          828

               micro        81.91        86.64        84.21         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        69.20        77.56        73.14          762
                 NEG        64.00        64.86        64.43          148
                 NEU        21.21        11.48        14.89           61

               micro        66.92        71.47        69.12          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        69.20        77.56        73.14          762
                 NEG        64.00        64.86        64.43          148
                 NEU        21.21        11.48        14.89           61

               micro        66.92        71.47        69.12          971

Train epoch 87:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 87: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 88:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 88: 100%|████████████████████████| 30/30 [00:03<00:00,  8.09it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.41        91.01        86.50          834
                   t        80.86        84.18        82.49          828

               micro        81.66        87.61        84.53         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        71.65        78.61        74.97          762
                 NEG        64.56        68.92        66.67          148
                 NEU        17.65         9.84        12.63           61

               micro        68.77        72.81        70.74          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        71.65        78.61        74.97          762
                 NEG        64.56        68.92        66.67          148
                 NEU        17.65         9.84        12.63           61

               micro        68.77        72.81        70.74          971

Train epoch 88:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 88: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 89:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 89: 100%|████████████████████████| 30/30 [00:03<00:00,  8.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.00        88.73        86.30          834
                   t        83.31        82.61        82.96          828

               micro        83.67        85.68        84.66         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        74.07        75.72        74.89          762
                 NEG        65.28        63.51        64.38          148
                 NEU        22.22        13.11        16.49           61

               micro        70.80        69.93        70.36          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        74.07        75.72        74.89          762
                 NEG        65.28        63.51        64.38          148
                 NEU        22.22        13.11        16.49           61

               micro        70.80        69.93        70.36          971

Train epoch 89:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 89: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 90:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 90: 100%|████████████████████████| 30/30 [00:03<00:00,  8.34it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.33        89.69        86.93          834
                   t        83.99        82.97        83.48          828

               micro        84.16        86.34        85.24         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        73.03        76.77        74.86          762
                 NEG        60.84        68.24        64.33          148
                 NEU        24.24        13.11        17.02           61

               micro        69.40        71.47        70.42          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        73.03        76.77        74.86          762
                 NEG        60.84        68.24        64.33          148
                 NEU        24.24        13.11        17.02           61

               micro        69.40        71.47        70.42          971

Train epoch 90:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 90: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 91:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 91: 100%|████████████████████████| 30/30 [00:03<00:00,  8.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.55        90.17        86.19          834
                   t        83.78        82.97        83.37          828

               micro        83.13        86.58        84.82         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.71        77.95        74.16          762
                 NEG        64.29        66.89        65.56          148
                 NEU        22.58        11.48        15.22           61

               micro        68.29        72.09        70.14          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.71        77.95        74.16          762
                 NEG        64.29        66.89        65.56          148
                 NEU        22.58        11.48        15.22           61

               micro        68.29        72.09        70.14          971

Train epoch 91:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 91: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 92:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 92: 100%|████████████████████████| 30/30 [00:03<00:00,  8.21it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.21        90.89        86.33          834
                   t        81.69        84.06        82.86          828

               micro        81.96        87.48        84.63         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.29        78.87        74.34          762
                 NEG        62.26        66.89        64.50          148
                 NEU        20.00        11.48        14.58           61

               micro        67.40        72.81        70.00          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.29        78.87        74.34          762
                 NEG        62.26        66.89        64.50          148
                 NEU        20.00        11.48        14.58           61

               micro        67.40        72.81        70.00          971

Train epoch 92:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 92: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 93:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 93: 100%|████████████████████████| 30/30 [00:03<00:00,  8.37it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.06        89.45        87.20          834
                   t        85.44        80.07        82.67          828

               micro        85.24        84.78        85.01         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        75.33        74.54        74.93          762
                 NEG        65.33        66.22        65.77          148
                 NEU        24.14        11.48        15.56           61

               micro        72.13        69.31        70.69          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        75.33        74.54        74.93          762
                 NEG        65.33        66.22        65.77          148
                 NEU        24.14        11.48        15.56           61

               micro        72.13        69.31        70.69          971

Train epoch 93:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 93: 100%|███████████████████████████| 79/79 [00:28<00:00,  2.80it/s]
Evaluate epoch 94:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 94: 100%|████████████████████████| 30/30 [00:03<00:00,  8.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.72        90.05        86.77          834
                   t        84.61        80.31        82.40          828

               micro        84.14        85.20        84.66         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        75.33        74.54        74.93          762
                 NEG        66.89        68.24        67.56          148
                 NEU        23.33        11.48        15.38           61

               micro        72.30        69.62        70.93          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        75.33        74.54        74.93          762
                 NEG        66.89        68.24        67.56          148
                 NEU        23.33        11.48        15.38           61

               micro        72.30        69.62        70.93          971

Train epoch 94:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 94: 100%|███████████████████████████| 79/79 [00:28<00:00,  2.82it/s]
Evaluate epoch 95:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 95: 100%|████████████████████████| 30/30 [00:06<00:00,  4.90it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.77        91.01        86.69          834
                   t        83.43        82.73        83.08          828

               micro        83.08        86.88        84.94         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        71.78        77.43        74.49          762
                 NEG        63.58        69.59        66.45          148
                 NEU        18.75         9.84        12.90           61

               micro        68.80        71.99        70.36          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        71.78        77.43        74.49          762
                 NEG        62.96        68.92        65.81          148
                 NEU        18.75         9.84        12.90           61

               micro        68.70        71.88        70.26          971

Train epoch 95:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 95: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 96:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 96: 100%|████████████████████████| 30/30 [00:03<00:00,  8.04it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.02        90.17        86.99          834
                   t        79.19        84.54        81.78          828

               micro        81.62        87.36        84.39         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        68.57        78.74        73.30          762
                 NEG        62.73        68.24        65.37          148
                 NEU        20.59        11.48        14.74           61

               micro        66.17        72.91        69.38          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        68.57        78.74        73.30          762
                 NEG        62.73        68.24        65.37          148
                 NEU        20.59        11.48        14.74           61

               micro        66.17        72.91        69.38          971

Train epoch 96:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 96: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 97:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 97: 100%|████████████████████████| 30/30 [00:03<00:00,  8.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.44        90.05        86.62          834
                   t        84.49        82.25        83.35          828

               micro        83.94        86.16        85.04         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        74.24        76.77        75.48          762
                 NEG        65.54        65.54        65.54          148
                 NEU        20.59        11.48        14.74           61

               micro        71.03        70.96        70.99          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        74.24        76.77        75.48          762
                 NEG        65.54        65.54        65.54          148
                 NEU        20.59        11.48        14.74           61

               micro        71.03        70.96        70.99          971

Train epoch 97:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 97: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 98:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 98: 100%|████████████████████████| 30/30 [00:03<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.78        90.41        86.97          834
                   t        81.26        83.82        82.52          828

               micro        82.55        87.12        84.78         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.65        78.35        74.30          762
                 NEG        63.35        68.92        66.02          148
                 NEU        23.33        11.48        15.38           61

               micro        68.15        72.71        70.35          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.65        78.35        74.30          762
                 NEG        63.35        68.92        66.02          148
                 NEU        23.33        11.48        15.38           61

               micro        68.15        72.71        70.35          971

Train epoch 98:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 98: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 99:   0%|                                 | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 99: 100%|████████████████████████| 30/30 [00:03<00:00,  8.16it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.24        90.53        86.73          834
                   t        78.65        85.87        82.10          828

               micro        80.95        88.21        84.42         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        68.65        80.45        74.08          762
                 NEG        64.52        67.57        66.01          148
                 NEU        22.86        13.11        16.67           61

               micro        66.57        74.25        70.20          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        68.65        80.45        74.08          762
                 NEG        64.52        67.57        66.01          148
                 NEU        22.86        13.11        16.67           61

               micro        66.57        74.25        70.20          971

Train epoch 99:   0%|                                    | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 99: 100%|███████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 100:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 100: 100%|███████████████████████| 30/30 [00:03<00:00,  8.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.29        89.45        86.79          834
                   t        83.72        83.21        83.46          828

               micro        84.02        86.34        85.16         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        73.04        77.17        75.05          762
                 NEG        66.44        66.89        66.67          148
                 NEU        19.44        11.48        14.43           61

               micro        70.10        71.47        70.78          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        73.04        77.17        75.05          762
                 NEG        66.44        66.89        66.67          148
                 NEU        19.44        11.48        14.43           61

               micro        70.10        71.47        70.78          971

Train epoch 100:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 100: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 101:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 101: 100%|███████████████████████| 30/30 [00:03<00:00,  8.33it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.44        90.05        86.62          834
                   t        81.98        84.06        83.01          828

               micro        82.73        87.06        84.84         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.78        78.22        74.31          762
                 NEG        64.97        68.92        66.89          148
                 NEU        21.88        11.48        15.05           61

               micro        68.38        72.61        70.43          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.78        78.22        74.31          762
                 NEG        64.97        68.92        66.89          148
                 NEU        21.88        11.48        15.05           61

               micro        68.38        72.61        70.43          971

Train epoch 101:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 101: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 102:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 102: 100%|███████████████████████| 30/30 [00:03<00:00,  7.97it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.32        91.01        86.99          834
                   t        82.89        83.70        83.29          828

               micro        83.11        87.36        85.19         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.34        78.74        74.30          762
                 NEG        64.74        68.24        66.45          148
                 NEU        23.53        13.11        16.84           61

               micro        67.98        73.02        70.41          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.34        78.74        74.30          762
                 NEG        64.74        68.24        66.45          148
                 NEU        23.53        13.11        16.84           61

               micro        67.98        73.02        70.41          971

Train epoch 102:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 102: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 103:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 103: 100%|███████████████████████| 30/30 [00:03<00:00,  8.20it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.70        90.53        86.98          834
                   t        80.76        84.66        82.67          828

               micro        82.26        87.61        84.85         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.74        79.00        74.64          762
                 NEG        64.52        67.57        66.01          148
                 NEU        20.00        11.48        14.58           61

               micro        68.11        73.02        70.48          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.74        79.00        74.64          762
                 NEG        64.52        67.57        66.01          148
                 NEU        20.00        11.48        14.58           61

               micro        68.11        73.02        70.48          971

Train epoch 103:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 103: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 104:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 104: 100%|███████████████████████| 30/30 [00:03<00:00,  8.08it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.37        91.85        86.85          834
                   t        80.05        84.78        82.35          828

               micro        81.24        88.33        84.64         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        67.93        80.05        73.49          762
                 NEG        62.20        68.92        65.38          148
                 NEU        20.59        11.48        14.74           61

               micro        65.60        74.05        69.57          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        67.93        80.05        73.49          762
                 NEG        62.20        68.92        65.38          148
                 NEU        20.59        11.48        14.74           61

               micro        65.60        74.05        69.57          971

Train epoch 104:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 104: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 105:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 105: 100%|███████████████████████| 30/30 [00:03<00:00,  8.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.93        90.17        86.94          834
                   t        81.95        84.42        83.16          828

               micro        82.96        87.30        85.08         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.51        78.74        74.40          762
                 NEG        65.79        67.57        66.67          148
                 NEU        23.33        11.48        15.38           61

               micro        68.44        72.81        70.56          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.51        78.74        74.40          762
                 NEG        65.79        67.57        66.67          148
                 NEU        23.33        11.48        15.38           61

               micro        68.44        72.81        70.56          971

Train epoch 105:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 105: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 106:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 106: 100%|███████████████████████| 30/30 [00:03<00:00,  8.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.04        91.61        87.12          834
                   t        79.61        84.90        82.17          828

               micro        81.36        88.27        84.68         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        68.00        80.05        73.54          762
                 NEG        66.45        68.24        67.33          148
                 NEU        20.00        11.48        14.58           61

               micro        66.24        73.94        69.88          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        68.00        80.05        73.54          762
                 NEG        66.45        68.24        67.33          148
                 NEU        20.00        11.48        14.58           61

               micro        66.24        73.94        69.88          971

Train epoch 106:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 106: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 107:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 107: 100%|███████████████████████| 30/30 [00:03<00:00,  8.14it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.70        91.13        87.26          834
                   t        80.41        84.78        82.54          828

               micro        82.09        87.97        84.93         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.42        79.66        74.75          762
                 NEG        65.38        68.92        67.11          148
                 NEU        24.24        13.11        17.02           61

               micro        68.22        73.84        70.92          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.42        79.66        74.75          762
                 NEG        65.38        68.92        67.11          148
                 NEU        24.24        13.11        17.02           61

               micro        68.22        73.84        70.92          971

Train epoch 107:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 107: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 108:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 108: 100%|███████████████████████| 30/30 [00:03<00:00,  8.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.00        90.65        87.20          834
                   t        81.95        84.42        83.16          828

               micro        83.00        87.55        85.21         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.96        78.87        74.70          762
                 NEG        65.81        68.92        67.33          148
                 NEU        21.88        11.48        15.05           61

               micro        68.67        73.12        70.82          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.96        78.87        74.70          762
                 NEG        65.81        68.92        67.33          148
                 NEU        21.88        11.48        15.05           61

               micro        68.67        73.12        70.82          971

Train epoch 108:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 108: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 109:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 109: 100%|███████████████████████| 30/30 [00:03<00:00,  8.12it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.48        91.49        87.30          834
                   t        83.03        83.94        83.48          828

               micro        83.27        87.73        85.44         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        71.55        78.87        75.03          762
                 NEG        64.97        68.92        66.89          148
                 NEU        21.88        11.48        15.05           61

               micro        69.00        73.12        71.00          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        71.55        78.87        75.03          762
                 NEG        64.97        68.92        66.89          148
                 NEU        21.88        11.48        15.05           61

               micro        69.00        73.12        71.00          971

Train epoch 109:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 109: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 110:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 110: 100%|███████████████████████| 30/30 [00:03<00:00,  8.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.06        91.73        87.18          834
                   t        79.64        85.02        82.24          828

               micro        81.39        88.39        84.74         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        68.69        80.05        73.94          762
                 NEG        64.97        68.92        66.89          148
                 NEU        22.86        13.11        16.67           61

               micro        66.67        74.15        70.21          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        68.69        80.05        73.94          762
                 NEG        64.97        68.92        66.89          148
                 NEU        22.86        13.11        16.67           61

               micro        66.67        74.15        70.21          971

Train epoch 110:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 110: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 111:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 111: 100%|███████████████████████| 30/30 [00:03<00:00,  8.12it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.79        91.13        87.31          834
                   t        81.40        84.54        82.94          828

               micro        82.63        87.85        85.16         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.53        79.13        74.58          762
                 NEG        67.11        68.92        68.00          148
                 NEU        21.62        13.11        16.33           61

               micro        68.30        73.43        70.77          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.53        79.13        74.58          762
                 NEG        67.11        68.92        68.00          148
                 NEU        21.62        13.11        16.33           61

               micro        68.30        73.43        70.77          971

Train epoch 111:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 111: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 112:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 112: 100%|███████████████████████| 30/30 [00:03<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.89        90.53        87.08          834
                   t        82.05        83.94        82.99          828

               micro        83.00        87.24        85.07         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        70.52        78.48        74.29          762
                 NEG        66.45        68.24        67.33          148
                 NEU        23.53        13.11        16.84           61

               micro        68.38        72.81        70.52          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        70.52        78.48        74.29          762
                 NEG        66.45        68.24        67.33          148
                 NEU        23.53        13.11        16.84           61

               micro        68.38        72.81        70.52          971

Train epoch 112:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 112: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.87it/s]
Evaluate epoch 113:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 113: 100%|███████████████████████| 30/30 [00:03<00:00,  7.85it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.04        90.89        87.33          834
                   t        84.31        83.09        83.70          828

               micro        84.17        87.00        85.56         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        72.53        77.95        75.14          762
                 NEG        67.32        69.59        68.44          148
                 NEU        24.24        13.11        17.02           61

               micro        70.15        72.61        71.36          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        72.53        77.95        75.14          762
                 NEG        67.32        69.59        68.44          148
                 NEU        24.24        13.11        17.02           61

               micro        70.15        72.61        71.36          971

Train epoch 113:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 113: 100%|██████████████████████████| 79/79 [00:28<00:00,  2.82it/s]
Evaluate epoch 114:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 114: 100%|███████████████████████| 30/30 [00:05<00:00,  5.84it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.53        91.25        87.22          834
                   t        81.88        84.06        82.96          828

               micro        82.74        87.67        85.13         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        69.81        78.61        73.95          762
                 NEG        65.81        68.92        67.33          148
                 NEU        21.05        13.11        16.16           61

               micro        67.46        73.02        70.13          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        69.81        78.61        73.95          762
                 NEG        65.81        68.92        67.33          148
                 NEU        21.05        13.11        16.16           61

               micro        67.46        73.02        70.13          971

Train epoch 114:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 114: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.88it/s]
Evaluate epoch 115:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 115: 100%|███████████████████████| 30/30 [00:03<00:00,  8.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.92        91.37        87.49          834
                   t        84.10        83.70        83.90          828

               micro        84.01        87.55        85.74         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        72.17        78.61        75.25          762
                 NEG        68.00        68.92        68.46          148
                 NEU        21.05        13.11        16.16           61

               micro        69.65        73.02        71.29          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        72.17        78.61        75.25          762
                 NEG        68.00        68.92        68.46          148
                 NEU        21.05        13.11        16.16           61

               micro        69.65        73.02        71.29          971

Train epoch 115:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 115: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.83it/s]
Evaluate epoch 116:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 116: 100%|███████████████████████| 30/30 [00:03<00:00,  8.13it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.81        91.25        87.37          834
                   t        82.70        84.30        83.49          828

               micro        83.28        87.79        85.47         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        71.92        79.00        75.30          762
                 NEG        67.11        68.92        68.00          148
                 NEU        24.24        13.11        17.02           61

               micro        69.67        73.33        71.45          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        71.92        79.00        75.30          762
                 NEG        67.11        68.92        68.00          148
                 NEU        24.24        13.11        17.02           61

               micro        69.67        73.33        71.45          971

Train epoch 116:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 116: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.82it/s]
Evaluate epoch 117:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 117: 100%|███████████████████████| 30/30 [00:03<00:00,  8.20it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.72        91.25        87.32          834
                   t        83.25        84.06        83.65          828

               micro        83.50        87.67        85.53         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        72.12        78.74        75.28          762
                 NEG        68.21        69.59        68.90          148
                 NEU        22.22        13.11        16.49           61

               micro        69.77        73.22        71.46          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        72.12        78.74        75.28          762
                 NEG        68.21        69.59        68.90          148
                 NEU        22.22        13.11        16.49           61

               micro        69.77        73.22        71.46          971

Train epoch 117:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 117: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.84it/s]
Evaluate epoch 118:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 118: 100%|███████████████████████| 30/30 [00:03<00:00,  7.87it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.55        91.37        87.29          834
                   t        83.03        83.94        83.48          828

               micro        83.30        87.67        85.43         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        71.94        78.74        75.19          762
                 NEG        67.76        69.59        68.67          148
                 NEU        22.22        13.11        16.49           61

               micro        69.57        73.22        71.35          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        71.94        78.74        75.19          762
                 NEG        67.76        69.59        68.67          148
                 NEU        22.22        13.11        16.49           61

               micro        69.57        73.22        71.35          971

Train epoch 118:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 118: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.86it/s]
Evaluate epoch 119:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 119: 100%|███████████████████████| 30/30 [00:03<00:00,  8.24it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.53        91.25        87.22          834
                   t        83.11        83.82        83.46          828

               micro        83.33        87.55        85.39         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        72.17        78.61        75.25          762
                 NEG        67.76        69.59        68.67          148
                 NEU        20.00        11.48        14.58           61

               micro        69.71        73.02        71.33          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        72.17        78.61        75.25          762
                 NEG        67.76        69.59        68.67          148
                 NEU        20.00        11.48        14.58           61

               micro        69.71        73.02        71.33          971

Train epoch 119:   0%|                                   | 0/79 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Train epoch 119: 100%|██████████████████████████| 79/79 [00:27<00:00,  2.85it/s]
Evaluate epoch 120:   0%|                                | 0/30 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
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
Evaluate epoch 120: 100%|███████████████████████| 30/30 [00:03<00:00,  8.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.63        91.25        87.27          834
                   t        83.11        83.82        83.46          828

               micro        83.38        87.55        85.41         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        72.17        78.61        75.25          762
                 NEG        68.21        69.59        68.90          148
                 NEU        18.92        11.48        14.29           61

               micro        69.65        73.02        71.29          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        72.17        78.61        75.25          762
                 NEG        68.21        69.59        68.90          148
                 NEU        18.92        11.48        14.29           61

               micro        69.65        73.02        71.29          971