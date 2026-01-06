Parse dataset 'train': 100%|██████████████| 1264/1264 [00:00<00:00, 1621.50it/s]
Parse dataset 'test': 100%|█████████████████| 480/480 [00:00<00:00, 1765.29it/s]
    14res    8
Using Original Syntactic GCN
Using Enhanced Semantic GCN with relative position, global context, and multi-scale features
Some weights of D2E2SModel were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['fc.weight', 'attention_layer.v.weight', 'TIN.lstm.weight_ih_l0_reverse', 'senti_classifier.bias', 'contrastive_encoder.opinion_proj.bias', 'TIN.GatedGCN.conv1.bias', 'Syn_gcn.W.1.weight', 'Sem_gcn.attn.relative_position_k.weight', 'TIN.residual_layer4.0.bias', 'fc.bias', 'lstm.weight_ih_l1_reverse', 'lstm.weight_hh_l0_reverse', 'lstm.bias_ih_l0_reverse', 'TIN.GatedGCN.conv3.rnn.bias_ih', 'lstm.bias_hh_l0', 'Sem_gcn.attn.linears.1.weight', 'TIN.residual_layer1.0.bias', 'TIN.GatedGCN.conv3.weight', 'Sem_gcn.attn.linears.0.bias', 'TIN.feature_fusion.2.weight', 'TIN.residual_layer1.2.weight', 'lstm.weight_ih_l0', 'TIN.residual_layer3.0.bias', 'TIN.residual_layer3.2.weight', 'senti_classifier.weight', 'Sem_gcn.global_context.fc.bias', 'Sem_gcn.multi_scale.fusion.bias', 'TIN.GatedGCN.conv1.lin.weight', 'Sem_gcn.attn.linears.0.weight', 'lstm.weight_ih_l1', 'lstm.bias_hh_l0_reverse', 'entity_classifier.bias', 'lstm.bias_hh_l1', 'TIN.residual_layer3.0.weight', 'Sem_gcn.W.0.weight', 'TIN.residual_layer1.0.weight', 'TIN.lstm.weight_hh_l0_reverse', 'Sem_gcn.global_context.gate.bias', 'entity_classifier.weight', 'Sem_gcn.attn.linears.1.bias', 'TIN.lstm.bias_ih_l0', 'TIN.feature_fusion.2.bias', 'TIN.lstm.bias_hh_l0', 'lstm.weight_hh_l0', 'lstm.weight_ih_l0_reverse', 'Sem_gcn.multi_scale.scale_weights', 'TIN.feature_fusion.3.bias', 'TIN.lstm.weight_hh_l1', 'TIN.lstm.bias_ih_l1', 'TIN.feature_fusion.0.bias', 'TIN.residual_layer2.2.weight', 'TIN.residual_layer4.2.weight', 'TIN.GatedGCN.conv2.bias', 'TIN.lstm.bias_hh_l1_reverse', 'Sem_gcn.W.0.bias', 'TIN.residual_layer2.0.weight', 'contrastive_encoder.entity_proj.weight', 'Sem_gcn.attn.relative_position_v.weight', 'TIN.residual_layer4.3.weight', 'lstm.bias_ih_l1', 'attention_layer.w_query.bias', 'TIN.residual_layer1.3.bias', 'TIN.lstm.weight_hh_l1_reverse', 'attention_layer.linear_q.bias', 'TIN.lstm.weight_hh_l0', 'Syn_gcn.W.0.bias', 'Sem_gcn.W.1.bias', 'TIN.GatedGCN.conv3.rnn.weight_ih', 'TIN.residual_layer2.0.bias', 'Sem_gcn.multi_scale.fusion.weight', 'size_embeddings.weight', 'Sem_gcn.global_context.fc.weight', 'contrastive_encoder.entity_proj.bias', 'lstm.bias_ih_l1_reverse', 'TIN.residual_layer3.2.bias', 'TIN.lstm.bias_hh_l1', 'attention_layer.linear_q.weight', 'TIN.residual_layer2.3.weight', 'TIN.GatedGCN.conv2.lin.weight', 'TIN.residual_layer4.3.bias', 'TIN.lstm.bias_ih_l1_reverse', 'lstm.bias_ih_l0', 'TIN.GatedGCN.conv3.rnn.bias_hh', 'TIN.residual_layer2.3.bias', 'TIN.lstm.bias_hh_l0_reverse', 'TIN.lstm.weight_ih_l1_reverse', 'Sem_gcn.global_context.gate.weight', 'Syn_gcn.W.0.weight', 'TIN.residual_layer3.3.weight', 'Syn_gcn.W.1.bias', 'TIN.lstm.weight_ih_l1', 'lstm.weight_hh_l1', 'TIN.residual_layer4.2.bias', 'TIN.residual_layer3.3.bias', 'Sem_gcn.W.1.weight', 'attention_layer.w_value.bias', 'TIN.lstm.bias_ih_l0_reverse', 'lstm.bias_hh_l1_reverse', 'TIN.residual_layer4.0.weight', 'attention_layer.w_query.weight', 'attention_layer.w_value.weight', 'contrastive_encoder.opinion_proj.weight', 'TIN.residual_layer2.2.bias', 'TIN.GatedGCN.conv3.rnn.weight_hh', 'TIN.feature_fusion.0.weight', 'TIN.lstm.weight_ih_l0', 'TIN.feature_fusion.3.weight', 'lstm.weight_hh_l1_reverse', 'TIN.residual_layer1.2.bias', 'TIN.residual_layer1.3.weight']
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
Train epoch 0: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.74it/s]
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
Evaluate epoch 1: 100%|█████████████████████████| 30/30 [00:10<00:00,  2.99it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.27         0.60         0.37          834
                   t        16.67         0.12         0.24          828

               micro         0.32         0.36         0.34         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 INV         0.00         0.00         0.00          0.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 INV         0.00         0.00         0.00          0.0
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0
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
Train epoch 1: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.77it/s]
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
Evaluate epoch 2: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.74it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        834.0
                   t         0.00         0.00         0.00        828.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0
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
Train epoch 2: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.78it/s]
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
Evaluate epoch 3: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.25it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        834.0
                   t         0.00         0.00         0.00        828.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0
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
Train epoch 3: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.78it/s]
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
Evaluate epoch 4: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        834.0
                   t         0.00         0.00         0.00        828.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0
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
Train epoch 4: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.78it/s]
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
Evaluate epoch 5: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        834.0
                   t         0.00         0.00         0.00        828.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0
                 NEU         0.00         0.00         0.00         61.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 POS         0.00         0.00         0.00        762.0
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
Train epoch 5: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.74it/s]
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
Evaluate epoch 6: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.39it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.33         1.80         3.52          834
                   t        71.54        10.63        18.51          828

               micro        73.05         6.20        11.43         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 POS        18.18         0.79         1.51          762
                 NEU         0.00         0.00         0.00           61

               micro        18.18         0.62         1.20          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 POS        18.18         0.79         1.51          762
                 NEU         0.00         0.00         0.00           61

               micro        18.18         0.62         1.20          971

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
Train epoch 6: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.74it/s]
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
Evaluate epoch 7: 100%|█████████████████████████| 30/30 [00:03<00:00,  9.08it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        73.84        24.70        37.02          834
                   t        65.75        40.34        50.00          828

               micro        68.61        32.49        44.10         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        80.00         2.70         5.23          148
                 POS        40.37        17.32        24.24          762
                 NEU         0.00         0.00         0.00           61

               micro        40.96        14.01        20.87          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        80.00         2.70         5.23          148
                 POS        39.45        16.93        23.69          762
                 NEU         0.00         0.00         0.00           61

               micro        40.06        13.70        20.41          971

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
Train epoch 7: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.77it/s]
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
Evaluate epoch 8: 100%|█████████████████████████| 30/30 [00:03<00:00,  8.41it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.55        54.56        64.72          834
                   t        69.86        58.21        63.50          828

               micro        74.25        56.38        64.09         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        53.57        20.27        29.41          148
                 POS        54.06        41.08        46.68          762
                 NEU         0.00         0.00         0.00           61

               micro        54.02        35.32        42.71          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        53.57        20.27        29.41          148
                 POS        53.54        40.68        46.23          762
                 NEU         0.00         0.00         0.00           61

               micro        53.54        35.02        42.34          971

2026-01-06 17:55:46.270258: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1767722146.471814     559 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1767722146.528420     559 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
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
Train epoch 8: 100%|████████████████████████████| 79/79 [00:29<00:00,  2.64it/s]
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
Evaluate epoch 9: 100%|█████████████████████████| 30/30 [00:03<00:00,  8.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.93        69.06        74.95          834
                   t        75.95        57.97        65.75          828

               micro        79.10        63.54        70.47         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        67.31        23.65        35.00          148
                 POS        63.94        48.16        54.94          762
                 NEU         0.00         0.00         0.00           61

               micro        64.22        41.40        50.34          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        67.31        23.65        35.00          148
                 POS        63.76        48.03        54.79          762
                 NEU         0.00         0.00         0.00           61

               micro        64.06        41.30        50.22          971

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
Train epoch 9: 100%|████████████████████████████| 79/79 [00:28<00:00,  2.75it/s]
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
Evaluate epoch 10: 100%|████████████████████████| 30/30 [00:03<00:00,  8.11it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        78.31        78.78        78.54          834
                   t        71.48        69.32        70.39          828

               micro        74.97        74.07        74.52         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        44.44        45.95        45.18          148
                 POS        58.64        57.87        58.26          762
                 NEU         0.00         0.00         0.00           61

               micro        56.24        52.42        54.26          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        44.44        45.95        45.18          148
                 POS        58.64        57.87        58.26          762
                 NEU         0.00         0.00         0.00           61

               micro        56.24        52.42        54.26          971

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
Train epoch 10: 100%|███████████████████████████| 79/79 [00:28<00:00,  2.75it/s]
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
Evaluate epoch 11: 100%|████████████████████████| 30/30 [00:03<00:00,  8.04it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        77.28        82.37        79.74          834
                   t        72.51        71.98        72.24          828

               micro        74.99        77.20        76.07         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        45.73        50.68        48.08          148
                 POS        58.98        60.76        59.86          762
                 NEU         0.00         0.00         0.00           61

               micro        56.69        55.41        56.04          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        45.73        50.68        48.08          148
                 POS        58.98        60.76        59.86          762
                 NEU         0.00         0.00         0.00           61

               micro        56.69        55.41        56.04          971

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
Train epoch 11: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.72it/s]
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
Evaluate epoch 12: 100%|████████████████████████| 30/30 [00:03<00:00,  8.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        76.38        84.53        80.25          834
                   t        76.10        71.14        73.53          828

               micro        76.25        77.86        77.05         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        42.71        57.43        48.99          148
                 POS        63.35        61.02        62.17          762
                 NEU         0.00         0.00         0.00           61

               micro        58.95        56.64        57.77          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        42.21        56.76        48.41          148
                 POS        63.35        61.02        62.17          762
                 NEU         0.00         0.00         0.00           61

               micro        58.84        56.54        57.67          971

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
Train epoch 12: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.72it/s]
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
Evaluate epoch 13: 100%|████████████████████████| 30/30 [00:03<00:00,  8.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.64        81.06        81.35          834
                   t        80.23        68.12        73.68          828

               micro        80.99        74.61        77.67         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        56.49        50.00        53.05          148
                 POS        61.17        60.37        60.77          762
                 NEU         0.00         0.00         0.00           61

               micro        60.48        54.99        57.61          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        56.49        50.00        53.05          148
                 POS        61.17        60.37        60.77          762
                 NEU         0.00         0.00         0.00           61

               micro        60.48        54.99        57.61          971

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
Train epoch 13: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.68it/s]
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
Evaluate epoch 14: 100%|████████████████████████| 30/30 [00:04<00:00,  7.46it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        77.63        84.89        81.10          834
                   t        70.11        79.59        74.55          828

               micro        73.81        82.25        77.80         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        52.56        55.41        53.95          148
                 POS        56.60        68.11        61.82          762
                 NEU         0.00         0.00         0.00           61

               micro        56.01        61.89        58.81          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        52.56        55.41        53.95          148
                 POS        56.60        68.11        61.82          762
                 NEU         0.00         0.00         0.00           61

               micro        56.01        61.89        58.81          971

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
Train epoch 14: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.61it/s]
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
Evaluate epoch 15: 100%|████████████████████████| 30/30 [00:04<00:00,  7.47it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        78.84        84.89        81.76          834
                   t        78.83        74.64        76.67          828

               micro        78.83        79.78        79.31         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        51.23        56.08        53.55          148
                 POS        65.87        65.35        65.61          762
                 NEU         0.00         0.00         0.00           61

               micro        63.29        59.84        61.51          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        51.23        56.08        53.55          148
                 POS        65.87        65.35        65.61          762
                 NEU         0.00         0.00         0.00           61

               micro        63.29        59.84        61.51          971

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
Train epoch 15: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.60it/s]
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
Evaluate epoch 16: 100%|████████████████████████| 30/30 [00:04<00:00,  7.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.88        84.17        82.49          834
                   t        84.40        71.86        77.63          828

               micro        82.45        78.04        80.19         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        53.90        56.08        54.97          148
                 POS        70.95        63.78        67.17          762
                 NEU         0.00         0.00         0.00           61

               micro        67.74        58.60        62.84          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        53.90        56.08        54.97          148
                 POS        70.95        63.78        67.17          762
                 NEU         0.00         0.00         0.00           61

               micro        67.74        58.60        62.84          971

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
Train epoch 16: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.58it/s]
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
Evaluate epoch 17: 100%|████████████████████████| 30/30 [00:04<00:00,  7.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.02        86.69        82.68          834
                   t        82.25        77.78        79.95          828

               micro        80.51        82.25        81.37         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        54.17        61.49        57.59          148
                 POS        71.31        67.85        69.54          762
                 NEU        40.00         3.28         6.06           61

               micro        67.93        62.82        65.28          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        54.17        61.49        57.59          148
                 POS        71.31        67.85        69.54          762
                 NEU        40.00         3.28         6.06           61

               micro        67.93        62.82        65.28          971

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
Train epoch 17: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.62it/s]
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
Evaluate epoch 18: 100%|████████████████████████| 30/30 [00:04<00:00,  7.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.37        84.05        83.20          834
                   t        85.58        76.69        80.89          828

               micro        83.87        80.39        82.09         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        61.15        57.43        59.23          148
                 POS        76.97        65.35        70.69          762
                 NEU        28.57         3.28         5.88           61

               micro        73.77        60.25        66.33          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        61.15        57.43        59.23          148
                 POS        76.97        65.35        70.69          762
                 NEU        28.57         3.28         5.88           61

               micro        73.77        60.25        66.33          971

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
Train epoch 18: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.59it/s]
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
Evaluate epoch 19: 100%|████████████████████████| 30/30 [00:04<00:00,  7.19it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.09        82.37        83.22          834
                   t        84.79        80.80        82.75          828

               micro        84.43        81.59        82.99         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        63.93        52.70        57.78          148
                 POS        74.20        70.21        72.15          762
                 NEU        50.00         1.64         3.17           61

               micro        72.66        63.23        67.62          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        63.93        52.70        57.78          148
                 POS        74.20        70.21        72.15          762
                 NEU        50.00         1.64         3.17           61

               micro        72.66        63.23        67.62          971

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
Train epoch 19: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.65it/s]
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
Evaluate epoch 20: 100%|████████████████████████| 30/30 [00:04<00:00,  7.30it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.82        87.29        83.39          834
                   t        79.66        85.63        82.54          828

               micro        79.74        86.46        82.97         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        49.23        64.86        55.98          148
                 POS        64.49        76.51        69.99          762
                 NEU        18.18         3.28         5.56           61

               micro        61.35        70.13        65.45          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        49.23        64.86        55.98          148
                 POS        64.49        76.51        69.99          762
                 NEU        18.18         3.28         5.56           61

               micro        61.35        70.13        65.45          971

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
Train epoch 20: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.57it/s]
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
Evaluate epoch 21: 100%|████████████████████████| 30/30 [00:04<00:00,  7.41it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.05        88.25        83.40          834
                   t        85.46        81.64        83.51          828

               micro        82.00        84.96        83.45         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        58.60        62.16        60.33          148
                 POS        72.91        74.54        73.72          762
                 NEU        22.22         3.28         5.71           61

               micro        70.05        68.18        69.10          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        58.60        62.16        60.33          148
                 POS        72.91        74.54        73.72          762
                 NEU        22.22         3.28         5.71           61

               micro        70.05        68.18        69.10          971

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
Train epoch 21: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.61it/s]
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
Evaluate epoch 22: 100%|████████████████████████| 30/30 [00:04<00:00,  7.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.45        85.61        84.00          834
                   t        85.87        83.70        84.77          828

               micro        84.10        84.66        84.38         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        59.62        62.84        61.18          148
                 POS        68.82        75.33        71.93          762
                 NEU        28.57         6.56        10.67           61

               micro        66.83        69.10        67.95          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        59.62        62.84        61.18          148
                 POS        68.82        75.33        71.93          762
                 NEU        28.57         6.56        10.67           61

               micro        66.83        69.10        67.95          971

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
Train epoch 22: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.59it/s]
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
Evaluate epoch 23: 100%|████████████████████████| 30/30 [00:03<00:00,  7.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.74        87.29        83.34          834
                   t        85.75        83.57        84.65          828

               micro        82.56        85.44        83.97         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        58.02        63.51        60.65          148
                 POS        71.25        76.12        73.60          762
                 NEU        31.25         8.20        12.99           61

               micro        68.45        69.93        69.18          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        58.02        63.51        60.65          148
                 POS        71.25        76.12        73.60          762
                 NEU        31.25         8.20        12.99           61

               micro        68.45        69.93        69.18          971

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
Train epoch 23: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.65it/s]
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
Evaluate epoch 24: 100%|████████████████████████| 30/30 [00:03<00:00,  7.66it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.12        87.05        83.98          834
                   t        82.97        87.08        84.97          828

               micro        82.03        87.06        84.47         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        50.00        65.54        56.73          148
                 POS        71.03        77.56        74.15          762
                 NEU        27.78         8.20        12.66           61

               micro        66.38        71.37        68.78          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        50.00        65.54        56.73          148
                 POS        71.03        77.56        74.15          762
                 NEU        27.78         8.20        12.66           61

               micro        66.38        71.37        68.78          971

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
Train epoch 24: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.61it/s]
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
Evaluate epoch 25: 100%|████████████████████████| 30/30 [00:04<00:00,  7.33it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        77.61        91.01        83.77          834
                   t        80.69        90.82        85.45          828

               micro        79.11        90.91        84.60         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        52.82        69.59        60.06          148
                 POS        64.72        82.81        72.65          762
                 NEU        20.00         8.20        11.63           61

               micro        61.84        76.11        68.24          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        52.82        69.59        60.06          148
                 POS        64.72        82.81        72.65          762
                 NEU        20.00         8.20        11.63           61

               micro        61.84        76.11        68.24          971

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
Train epoch 25: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.62it/s]
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
Evaluate epoch 26: 100%|████████████████████████| 30/30 [00:03<00:00,  7.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.46        88.37        84.23          834
                   t        81.92        90.82        86.14          828

               micro        81.19        89.59        85.18         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        64.44        58.78        61.48          148
                 POS        69.27        81.36        74.83          762
                 NEU        14.29         9.84        11.65           61

               micro        66.51        73.43        69.80          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        64.44        58.78        61.48          148
                 POS        69.27        81.36        74.83          762
                 NEU        14.29         9.84        11.65           61

               micro        66.51        73.43        69.80          971

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
Train epoch 26: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.62it/s]
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
Evaluate epoch 27: 100%|████████████████████████| 30/30 [00:04<00:00,  7.46it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.49        87.53        84.93          834
                   t        83.91        88.16        85.98          828

               micro        83.19        87.85        85.46         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        64.67        65.54        65.10          148
                 POS        68.66        79.92        73.86          762
                 NEU        24.00         9.84        13.95           61

               micro        67.04        73.33        70.04          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        64.67        65.54        65.10          148
                 POS        68.66        79.92        73.86          762
                 NEU        24.00         9.84        13.95           61

               micro        67.04        73.33        70.04          971

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
Train epoch 27: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.66it/s]
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
Evaluate epoch 28: 100%|████████████████████████| 30/30 [00:04<00:00,  7.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        78.57        89.69        83.76          834
                   t        83.43        88.16        85.73          828

               micro        80.90        88.93        84.72         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        56.18        67.57        61.35          148
                 POS        69.19        80.45        74.39          762
                 NEU        20.51        13.11        16.00           61

               micro        65.37        74.25        69.53          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        56.18        67.57        61.35          148
                 POS        69.19        80.45        74.39          762
                 NEU        20.51        13.11        16.00           61

               micro        65.37        74.25        69.53          971

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
Train epoch 28: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 29: 100%|████████████████████████| 30/30 [00:04<00:00,  7.42it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.66        85.37        85.01          834
                   t        86.18        85.87        86.03          828

               micro        85.41        85.62        85.52         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        57.76        62.84        60.19          148
                 POS        75.49        76.38        75.93          762
                 NEU        23.08         4.92         8.11           61

               micro        71.75        69.82        70.77          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        57.76        62.84        60.19          148
                 POS        75.49        76.38        75.93          762
                 NEU        23.08         4.92         8.11           61

               micro        71.75        69.82        70.77          971

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
Train epoch 29: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.60it/s]
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
Evaluate epoch 30: 100%|████████████████████████| 30/30 [00:04<00:00,  7.14it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.39        89.93        84.89          834
                   t        83.86        90.34        86.98          828

               micro        82.08        90.13        85.92         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        56.80        64.86        60.57          148
                 POS        68.59        82.55        74.93          762
                 NEU        15.00         4.92         7.41           61

               micro        65.82        74.97        70.10          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        56.80        64.86        60.57          148
                 POS        68.59        82.55        74.93          762
                 NEU        15.00         4.92         7.41           61

               micro        65.82        74.97        70.10          971

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
Train epoch 30: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.58it/s]
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
Evaluate epoch 31: 100%|████████████████████████| 30/30 [00:04<00:00,  7.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.85        87.17        85.48          834
                   t        85.66        87.32        86.48          828

               micro        84.75        87.24        85.98         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        66.67        60.81        63.60          148
                 POS        74.07        78.35        76.15          762
                 NEU        22.73         8.20        12.05           61

               micro        71.86        71.27        71.56          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        66.67        60.81        63.60          148
                 POS        74.07        78.35        76.15          762
                 NEU        22.73         8.20        12.05           61

               micro        71.86        71.27        71.56          971

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
Train epoch 31: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.61it/s]
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
Evaluate epoch 32: 100%|████████████████████████| 30/30 [00:04<00:00,  7.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.97        88.85        85.27          834
                   t        83.52        91.18        87.18          828

               micro        82.74        90.01        86.22         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        62.11        67.57        64.72          148
                 POS        72.83        81.63        76.98          762
                 NEU        19.44        11.48        14.43           61

               micro        69.36        75.08        72.11          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        62.11        67.57        64.72          148
                 POS        72.83        81.63        76.98          762
                 NEU        19.44        11.48        14.43           61

               micro        69.36        75.08        72.11          971

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
Train epoch 32: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 33: 100%|████████████████████████| 30/30 [00:04<00:00,  7.21it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        79.62        90.89        84.88          834
                   t        83.63        91.30        87.30          828

               micro        81.57        91.10        86.07         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        60.98        67.57        64.10          148
                 POS        71.98        82.94        77.07          762
                 NEU        21.43         9.84        13.48           61

               micro        68.97        76.00        72.32          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        60.98        67.57        64.10          148
                 POS        71.98        82.94        77.07          762
                 NEU        21.43         9.84        13.48           61

               micro        68.97        76.00        72.32          971

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
Train epoch 33: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.59it/s]
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
Evaluate epoch 34: 100%|████████████████████████| 30/30 [00:04<00:00,  7.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.94        88.01        85.40          834
                   t        84.73        90.46        87.50          828

               micro        83.83        89.23        86.45         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        64.14        62.84        63.48          148
                 POS        68.73        83.07        75.22          762
                 NEU        25.93        11.48        15.91           61

               micro        67.06        75.49        71.03          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        64.14        62.84        63.48          148
                 POS        68.73        83.07        75.22          762
                 NEU        25.93        11.48        15.91           61

               micro        67.06        75.49        71.03          971

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
Train epoch 34: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.57it/s]
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
Evaluate epoch 35: 100%|████████████████████████| 30/30 [00:06<00:00,  4.70it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.79        87.41        85.56          834
                   t        86.60        88.16        87.37          828

               micro        85.17        87.79        86.46         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        64.24        65.54        64.88          148
                 POS        75.40        80.45        77.84          762
                 NEU        30.43        11.48        16.67           61

               micro        72.64        73.84        73.24          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        64.24        65.54        64.88          148
                 POS        75.40        80.45        77.84          762
                 NEU        30.43        11.48        16.67           61

               micro        72.64        73.84        73.24          971

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
Train epoch 35: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 36: 100%|████████████████████████| 30/30 [00:04<00:00,  7.04it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.30        87.65        86.46          834
                   t        86.40        87.44        86.91          828

               micro        85.84        87.55        86.68         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.92        68.92        68.92          148
                 POS        75.78        79.66        77.67          762
                 NEU        20.00         4.92         7.89           61

               micro        73.86        73.33        73.59          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.92        68.92        68.92          148
                 POS        75.78        79.66        77.67          762
                 NEU        20.00         4.92         7.89           61

               micro        73.86        73.33        73.59          971

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
Train epoch 36: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 37: 100%|████████████████████████| 30/30 [00:04<00:00,  7.10it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.97        89.93        85.76          834
                   t        85.53        89.25        87.35          828

               micro        83.70        89.59        86.54         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        60.47        70.27        65.00          148
                 POS        72.70        81.76        76.96          762
                 NEU        17.24         8.20        11.11           61

               micro        69.19        75.39        72.15          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        60.47        70.27        65.00          148
                 POS        72.70        81.76        76.96          762
                 NEU        17.24         8.20        11.11           61

               micro        69.19        75.39        72.15          971

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
Train epoch 37: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 38: 100%|████████████████████████| 30/30 [00:04<00:00,  7.20it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.41        88.61        85.93          834
                   t        85.90        87.56        86.72          828

               micro        84.62        88.09        86.32         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        66.67        63.51        65.05          148
                 POS        75.21        80.84        77.93          762
                 NEU        25.00         8.20        12.35           61

               micro        72.96        73.64        73.30          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        66.67        63.51        65.05          148
                 POS        75.21        80.84        77.93          762
                 NEU        25.00         8.20        12.35           61

               micro        72.96        73.64        73.30          971

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
Train epoch 38: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.55it/s]
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
Evaluate epoch 39: 100%|████████████████████████| 30/30 [00:04<00:00,  7.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.75        90.53        85.36          834
                   t        83.74        91.43        87.41          828

               micro        82.22        90.97        86.38         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        59.24        73.65        65.66          148
                 POS        70.63        84.25        76.84          762
                 NEU        25.00        11.48        15.73           61

               micro        67.62        78.06        72.47          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        59.24        73.65        65.66          148
                 POS        70.63        84.25        76.84          762
                 NEU        25.00        11.48        15.73           61

               micro        67.62        78.06        72.47          971

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
Train epoch 39: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 40: 100%|████████████████████████| 30/30 [00:04<00:00,  7.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.71        88.37        86.50          834
                   t        86.68        88.04        87.36          828

               micro        85.68        88.21        86.93         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.00        68.92        68.46          148
                 POS        75.40        80.84        78.02          762
                 NEU        36.84        11.48        17.50           61

               micro        73.53        74.67        74.09          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.00        68.92        68.46          148
                 POS        75.40        80.84        78.02          762
                 NEU        36.84        11.48        17.50           61

               micro        73.53        74.67        74.09          971

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
Train epoch 40: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 41: 100%|████████████████████████| 30/30 [00:04<00:00,  7.09it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        87.25        85.37        86.30          834
                   t        85.50        89.01        87.22          828

               micro        86.35        87.18        86.77         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.52        64.19        64.85          148
                 POS        77.45        78.87        78.15          762
                 NEU        29.17        11.48        16.47           61

               micro        74.39        72.40        73.38          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.52        64.19        64.85          148
                 POS        77.45        78.87        78.15          762
                 NEU        29.17        11.48        16.47           61

               micro        74.39        72.40        73.38          971

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
Train epoch 41: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.58it/s]
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
Evaluate epoch 42: 100%|████████████████████████| 30/30 [00:04<00:00,  6.92it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.59        90.89        85.99          834
                   t        84.28        89.98        87.03          828

               micro        82.90        90.43        86.50         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        57.84        72.30        64.26          148
                 POS        73.05        82.55        77.51          762
                 NEU        20.00         8.20        11.63           61

               micro        69.19        76.31        72.58          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        57.84        72.30        64.26          148
                 POS        73.05        82.55        77.51          762
                 NEU        20.00         8.20        11.63           61

               micro        69.19        76.31        72.58          971

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
Train epoch 42: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 43: 100%|████████████████████████| 30/30 [00:04<00:00,  7.02it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.21        90.89        86.33          834
                   t        83.64        92.03        87.64          828

               micro        82.92        91.46        86.98         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        64.15        68.92        66.45          148
                 POS        72.70        83.86        77.88          762
                 NEU        21.05        13.11        16.16           61

               micro        69.61        77.14        73.18          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        64.15        68.92        66.45          148
                 POS        72.70        83.86        77.88          762
                 NEU        21.05        13.11        16.16           61

               micro        69.61        77.14        73.18          971

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
Train epoch 43: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.59it/s]
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
Evaluate epoch 44: 100%|████████████████████████| 30/30 [00:04<00:00,  7.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.15        89.93        86.41          834
                   t        84.63        90.46        87.45          828

               micro        83.88        90.19        86.92         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        66.01        68.24        67.11          148
                 POS        77.82        80.58        79.17          762
                 NEU        26.09         9.84        14.29           61

               micro        74.72        74.25        74.48          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        66.01        68.24        67.11          148
                 POS        77.82        80.58        79.17          762
                 NEU        26.09         9.84        14.29           61

               micro        74.72        74.25        74.48          971

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
Train epoch 44: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.60it/s]
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
Evaluate epoch 45: 100%|████████████████████████| 30/30 [00:04<00:00,  7.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.21        90.89        86.33          834
                   t        83.21        91.55        87.18          828

               micro        82.71        91.22        86.75         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        66.05        72.30        69.03          148
                 POS        71.90        83.60        77.31          762
                 NEU        21.21        11.48        14.89           61

               micro        69.47        77.34        73.20          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        66.05        72.30        69.03          148
                 POS        71.90        83.60        77.31          762
                 NEU        21.21        11.48        14.89           61

               micro        69.47        77.34        73.20          971

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
Train epoch 45: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.61it/s]
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
Evaluate epoch 46: 100%|████████████████████████| 30/30 [00:04<00:00,  7.24it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.00        90.05        86.92          834
                   t        87.45        87.56        87.51          828

               micro        85.66        88.81        87.21         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        63.69        72.30        67.72          148
                 POS        76.97        80.71        78.80          762
                 NEU        33.33         8.20        13.16           61

               micro        74.03        74.87        74.45          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        63.69        72.30        67.72          148
                 POS        76.97        80.71        78.80          762
                 NEU        33.33         8.20        13.16           61

               micro        74.03        74.87        74.45          971

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
Train epoch 46: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.61it/s]
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
Evaluate epoch 47: 100%|████████████████████████| 30/30 [00:04<00:00,  7.29it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.56        90.17        86.74          834
                   t        86.86        88.65        87.75          828

               micro        85.16        89.41        87.23         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        62.87        70.95        66.67          148
                 POS        74.05        82.02        77.83          762
                 NEU        26.09         9.84        14.29           61

               micro        71.18        75.80        73.42          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        62.87        70.95        66.67          148
                 POS        74.05        82.02        77.83          762
                 NEU        26.09         9.84        14.29           61

               micro        71.18        75.80        73.42          971

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
Train epoch 47: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 48: 100%|████████████████████████| 30/30 [00:04<00:00,  7.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.95        90.89        86.19          834
                   t        87.64        87.32        87.48          828

               micro        84.63        89.11        86.81         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        69.23        66.89        68.04          148
                 POS        74.94        81.23        77.96          762
                 NEU        23.53        13.11        16.84           61

               micro        72.38        74.77        73.56          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        69.23        66.89        68.04          148
                 POS        74.94        81.23        77.96          762
                 NEU        23.53        13.11        16.84           61

               micro        72.38        74.77        73.56          971

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
Train epoch 48: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 49: 100%|████████████████████████| 30/30 [00:04<00:00,  7.16it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.52        89.93        86.61          834
                   t        85.00        90.34        87.59          828

               micro        84.25        90.13        87.09         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        66.21        64.86        65.53          148
                 POS        75.91        81.89        78.79          762
                 NEU        25.93        11.48        15.91           61

               micro        73.14        74.87        73.99          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        66.21        64.86        65.53          148
                 POS        75.91        81.89        78.79          762
                 NEU        25.93        11.48        15.91           61

               micro        73.14        74.87        73.99          971

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
Train epoch 49: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 50: 100%|████████████████████████| 30/30 [00:04<00:00,  7.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.57        90.29        86.80          834
                   t        85.06        90.10        87.51          828

               micro        84.31        90.19        87.15         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        61.63        71.62        66.25          148
                 POS        74.91        81.50        78.06          762
                 NEU        24.14        11.48        15.56           61

               micro        71.26        75.59        73.36          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        61.63        71.62        66.25          148
                 POS        74.91        81.50        78.06          762
                 NEU        24.14        11.48        15.56           61

               micro        71.26        75.59        73.36          971

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
Train epoch 50: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.53it/s]
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
Evaluate epoch 51: 100%|████████████████████████| 30/30 [00:03<00:00,  7.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.99        90.65        86.65          834
                   t        84.96        91.43        88.07          828

               micro        83.96        91.03        87.36         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        62.79        72.97        67.50          148
                 POS        73.61        83.46        78.23          762
                 NEU        24.14        11.48        15.56           61

               micro        70.52        77.34        73.77          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        62.79        72.97        67.50          148
                 POS        73.61        83.46        78.23          762
                 NEU        24.14        11.48        15.56           61

               micro        70.52        77.34        73.77          971

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
Train epoch 51: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 52: 100%|████████████████████████| 30/30 [00:04<00:00,  7.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.43        87.89        86.64          834
                   t        88.88        86.84        87.84          828

               micro        87.10        87.36        87.23         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        73.02        62.16        67.15          148
                 POS        78.46        78.87        78.66          762
                 NEU        29.17        11.48        16.47           61

               micro        76.42        72.09        74.19          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        73.02        62.16        67.15          148
                 POS        78.46        78.87        78.66          762
                 NEU        29.17        11.48        16.47           61

               micro        76.42        72.09        74.19          971

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
Train epoch 52: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.58it/s]
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
Evaluate epoch 53: 100%|████████████████████████| 30/30 [00:04<00:00,  7.39it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.77        90.41        86.42          834
                   t        84.83        91.18        87.89          828

               micro        83.79        90.79        87.15         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.22        70.95        67.96          148
                 POS        73.75        83.33        78.25          762
                 NEU        21.88        11.48        15.05           61

               micro        70.87        76.93        73.78          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.22        70.95        67.96          148
                 POS        73.75        83.33        78.25          762
                 NEU        21.88        11.48        15.05           61

               micro        70.87        76.93        73.78          971

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
Train epoch 53: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.60it/s]
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
Evaluate epoch 54: 100%|████████████████████████| 30/30 [00:04<00:00,  7.09it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.63        90.05        86.72          834
                   t        86.37        88.77        87.55          828

               micro        84.96        89.41        87.13         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        63.25        70.95        66.88          148
                 POS        78.38        79.92        79.14          762
                 NEU        21.88        11.48        15.05           61

               micro        73.95        74.25        74.10          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        63.25        70.95        66.88          148
                 POS        78.38        79.92        79.14          762
                 NEU        21.88        11.48        15.05           61

               micro        73.95        74.25        74.10          971

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
Train epoch 54: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.57it/s]
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
Evaluate epoch 55: 100%|████████████████████████| 30/30 [00:04<00:00,  7.20it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.35        89.45        87.35          834
                   t        86.09        90.46        88.22          828

               micro        85.72        89.95        87.79         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        66.88        69.59        68.21          148
                 POS        77.36        81.63        79.44          762
                 NEU        29.17        11.48        16.47           61

               micro        74.54        75.39        74.96          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        66.88        69.59        68.21          148
                 POS        77.36        81.63        79.44          762
                 NEU        29.17        11.48        16.47           61

               micro        74.54        75.39        74.96          971

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
Train epoch 55: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.59it/s]
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
Evaluate epoch 56: 100%|████████████████████████| 30/30 [00:04<00:00,  6.99it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.99        88.97        86.94          834
                   t        85.68        89.61        87.60          828

               micro        85.34        89.29        87.27         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        70.80        65.54        68.07          148
                 POS        76.84        80.97        78.85          762
                 NEU        24.14        11.48        15.56           61

               micro        74.41        74.25        74.33          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        70.80        65.54        68.07          148
                 POS        76.84        80.97        78.85          762
                 NEU        24.14        11.48        15.56           61

               micro        74.41        74.25        74.33          971

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
Train epoch 56: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.57it/s]
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
Evaluate epoch 57: 100%|████████████████████████| 30/30 [00:04<00:00,  7.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.69        89.57        87.06          834
                   t        86.08        88.89        87.46          828

               micro        85.38        89.23        87.26         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.46        68.92        68.69          148
                 POS        76.62        80.84        78.67          762
                 NEU        25.93        11.48        15.91           61

               micro        73.98        74.67        74.32          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.46        68.92        68.69          148
                 POS        76.62        80.84        78.67          762
                 NEU        25.93        11.48        15.91           61

               micro        73.98        74.67        74.32          971

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
Train epoch 57: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.55it/s]
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
Evaluate epoch 58: 100%|████████████████████████| 30/30 [00:04<00:00,  7.33it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.65        90.17        86.79          834
                   t        85.42        90.58        87.92          828

               micro        84.52        90.37        87.35         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        63.31        72.30        67.51          148
                 POS        75.90        82.68        79.15          762
                 NEU        25.71        14.75        18.75           61

               micro        72.15        76.83        74.41          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        63.31        72.30        67.51          148
                 POS        75.90        82.68        79.15          762
                 NEU        25.71        14.75        18.75           61

               micro        72.15        76.83        74.41          971

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
Train epoch 58: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.59it/s]
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
Evaluate epoch 59: 100%|████████████████████████| 30/30 [00:04<00:00,  7.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.30        90.77        87.41          834
                   t        87.50        89.61        88.54          828

               micro        85.85        90.19        87.97         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.45        72.97        69.01          148
                 POS        76.73        81.36        78.98          762
                 NEU        24.14        11.48        15.56           61

               micro        73.35        75.70        74.51          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.45        72.97        69.01          148
                 POS        76.73        81.36        78.98          762
                 NEU        24.14        11.48        15.56           61

               micro        73.35        75.70        74.51          971

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
Train epoch 59: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 60: 100%|████████████████████████| 30/30 [00:04<00:00,  7.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.57        90.29        86.80          834
                   t        84.44        91.79        87.96          828

               micro        84.01        91.03        87.38         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.21        69.59        68.90          148
                 POS        74.07        83.20        78.37          762
                 NEU        21.05        13.11        16.16           61

               micro        71.29        76.73        73.91          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.21        69.59        68.90          148
                 POS        74.07        83.20        78.37          762
                 NEU        21.05        13.11        16.16           61

               micro        71.29        76.73        73.91          971

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
Train epoch 60: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.61it/s]
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
Evaluate epoch 61: 100%|████████████████████████| 30/30 [00:03<00:00,  7.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.67        89.57        87.57          834
                   t        87.27        87.80        87.54          828

               micro        86.45        88.69        87.56         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.16        68.24        66.67          148
                 POS        76.35        81.36        78.78          762
                 NEU        28.57         9.84        14.63           61

               micro        73.58        74.87        74.22          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.16        68.24        66.67          148
                 POS        76.35        81.36        78.78          762
                 NEU        28.57         9.84        14.63           61

               micro        73.58        74.87        74.22          971

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
Train epoch 61: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.64it/s]
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
Evaluate epoch 62: 100%|████████████████████████| 30/30 [00:03<00:00,  7.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.68        91.61        86.92          834
                   t        85.52        90.58        87.98          828

               micro        84.06        91.10        87.44         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        63.31        72.30        67.51          148
                 POS        74.53        82.94        78.51          762
                 NEU        23.33        11.48        15.38           61

               micro        71.25        76.83        73.93          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        63.31        72.30        67.51          148
                 POS        74.53        82.94        78.51          762
                 NEU        23.33        11.48        15.38           61

               micro        71.25        76.83        73.93          971

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
Train epoch 62: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.55it/s]
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
Evaluate epoch 63: 100%|████████████████████████| 30/30 [00:03<00:00,  7.64it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.39        90.89        86.98          834
                   t        85.09        91.67        88.26          828

               micro        84.23        91.28        87.61         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.82        70.27        67.97          148
                 POS        72.51        84.12        77.89          762
                 NEU        22.58        11.48        15.22           61

               micro        70.08        77.45        73.58          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.82        70.27        67.97          148
                 POS        72.51        84.12        77.89          762
                 NEU        22.58        11.48        15.22           61

               micro        70.08        77.45        73.58          971

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
Train epoch 63: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.67it/s]
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
Evaluate epoch 64: 100%|████████████████████████| 30/30 [00:03<00:00,  7.79it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.65        91.37        86.79          834
                   t        86.81        89.01        87.90          828

               micro        84.64        90.19        87.33         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.67        69.59        69.13          148
                 POS        75.61        81.76        78.56          762
                 NEU        25.00        11.48        15.73           61

               micro        73.15        75.49        74.30          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.67        69.59        69.13          148
                 POS        75.61        81.76        78.56          762
                 NEU        25.00        11.48        15.73           61

               micro        73.15        75.49        74.30          971

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
Train epoch 64: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.65it/s]
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
Evaluate epoch 65: 100%|████████████████████████| 30/30 [00:03<00:00,  7.53it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.71        89.69        87.13          834
                   t        85.80        90.46        88.07          828

               micro        85.25        90.07        87.60         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        69.78        65.54        67.60          148
                 POS        75.57        82.81        79.02          762
                 NEU        28.00        11.48        16.28           61

               micro        73.57        75.70        74.62          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        69.78        65.54        67.60          148
                 POS        75.57        82.81        79.02          762
                 NEU        28.00        11.48        16.28           61

               micro        73.57        75.70        74.62          971

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
Train epoch 65: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.70it/s]
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
Evaluate epoch 66: 100%|████████████████████████| 30/30 [00:03<00:00,  7.80it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        86.16        88.85        87.49          834
                   t        86.85        90.94        88.85          828

               micro        86.51        89.89        88.17         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.92        68.92        68.92          148
                 POS        76.27        82.68        79.35          762
                 NEU        25.93        11.48        15.91           61

               micro        73.83        76.11        74.95          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.92        68.92        68.92          148
                 POS        76.27        82.68        79.35          762
                 NEU        25.93        11.48        15.91           61

               micro        73.83        76.11        74.95          971

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
Train epoch 66: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.67it/s]
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
Evaluate epoch 67: 100%|████████████████████████| 30/30 [00:03<00:00,  7.76it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.47        91.37        86.69          834
                   t        84.12        92.15        87.95          828

               micro        83.29        91.76        87.32         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        64.02        70.95        67.31          148
                 POS        71.71        84.51        77.59          762
                 NEU        22.58        11.48        15.22           61

               micro        69.17        77.86        73.26          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        64.02        70.95        67.31          148
                 POS        71.71        84.51        77.59          762
                 NEU        22.58        11.48        15.22           61

               micro        69.17        77.86        73.26          971

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
Train epoch 67: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.67it/s]
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
Evaluate epoch 68: 100%|████████████████████████| 30/30 [00:03<00:00,  7.71it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.03        89.21        87.07          834
                   t        86.29        90.46        88.33          828

               micro        85.66        89.83        87.69         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.79        65.54        67.13          148
                 POS        78.31        81.50        79.87          762
                 NEU        26.92        11.48        16.09           61

               micro        75.52        74.67        75.09          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.79        65.54        67.13          148
                 POS        78.31        81.50        79.87          762
                 NEU        26.92        11.48        16.09           61

               micro        75.52        74.67        75.09          971

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
Train epoch 68: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.65it/s]
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
Evaluate epoch 69: 100%|████████████████████████| 30/30 [00:04<00:00,  7.41it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.02        90.89        86.78          834
                   t        85.14        91.30        88.11          828

               micro        84.06        91.10        87.44         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.45        72.97        69.01          148
                 POS        75.63        82.68        79.00          762
                 NEU        25.81        13.11        17.39           61

               micro        72.50        76.83        74.60          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.45        72.97        69.01          148
                 POS        75.63        82.68        79.00          762
                 NEU        25.81        13.11        17.39           61

               micro        72.50        76.83        74.60          971

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
Train epoch 69: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.63it/s]
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
Evaluate epoch 70: 100%|████████████████████████| 30/30 [00:03<00:00,  7.57it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.52        89.69        87.03          834
                   t        86.55        90.94        88.69          828

               micro        85.53        90.31        87.85         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        71.53        69.59        70.55          148
                 POS        78.10        81.89        79.95          762
                 NEU        28.00        11.48        16.28           61

               micro        75.83        75.59        75.71          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        71.53        69.59        70.55          148
                 POS        78.10        81.89        79.95          762
                 NEU        28.00        11.48        16.28           61

               micro        75.83        75.59        75.71          971

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
Train epoch 70: 100%|███████████████████████████| 79/79 [00:29<00:00,  2.64it/s]
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
Evaluate epoch 71: 100%|████████████████████████| 30/30 [00:04<00:00,  7.39it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.35        89.45        87.35          834
                   t        86.93        89.98        88.43          828

               micro        86.14        89.71        87.89         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        70.14        68.24        69.18          148
                 POS        78.16        81.23        79.67          762
                 NEU        33.33         9.84        15.19           61

               micro        76.10        74.77        75.43          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        70.14        68.24        69.18          148
                 POS        78.16        81.23        79.67          762
                 NEU        33.33         9.84        15.19           61

               micro        76.10        74.77        75.43          971

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
Train epoch 71: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.63it/s]
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
Evaluate epoch 72: 100%|████████████████████████| 30/30 [00:04<00:00,  7.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.93        89.21        87.02          834
                   t        84.98        91.55        88.14          828

               micro        84.95        90.37        87.58         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.00        68.92        68.46          148
                 POS        76.06        82.15        78.99          762
                 NEU        26.92        11.48        16.09           61

               micro        73.57        75.70        74.62          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.00        68.92        68.46          148
                 POS        76.06        82.15        78.99          762
                 NEU        26.92        11.48        16.09           61

               micro        73.57        75.70        74.62          971

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
Train epoch 72: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.60it/s]
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
Evaluate epoch 73: 100%|████████████████████████| 30/30 [00:04<00:00,  7.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.39        89.69        87.49          834
                   t        85.96        90.94        88.38          828

               micro        85.67        90.31        87.93         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        69.08        70.95        70.00          148
                 POS        77.03        82.28        79.57          762
                 NEU        25.00        11.48        15.73           61

               micro        74.35        76.11        75.22          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        69.08        70.95        70.00          148
                 POS        77.03        82.28        79.57          762
                 NEU        25.00        11.48        15.73           61

               micro        74.35        76.11        75.22          971

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
Train epoch 73: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.59it/s]
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
Evaluate epoch 74: 100%|████████████████████████| 30/30 [00:04<00:00,  7.16it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.31        89.81        87.50          834
                   t        86.65        90.94        88.74          828

               micro        85.98        90.37        88.12         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        73.28        64.86        68.82          148
                 POS        76.28        82.28        79.17          762
                 NEU        24.24        13.11        17.02           61

               micro        74.14        75.28        74.71          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        73.28        64.86        68.82          148
                 POS        76.28        82.28        79.17          762
                 NEU        24.24        13.11        17.02           61

               micro        74.14        75.28        74.71          971

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
Train epoch 74: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.57it/s]
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
Evaluate epoch 75: 100%|████████████████████████| 30/30 [00:04<00:00,  7.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.81        89.69        87.18          834
                   t        86.47        90.34        88.36          828

               micro        85.63        90.01        87.77         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        72.86        68.92        70.83          148
                 POS        79.23        80.58        79.90          762
                 NEU        30.00        14.75        19.78           61

               micro        76.72        74.67        75.68          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        72.86        68.92        70.83          148
                 POS        79.23        80.58        79.90          762
                 NEU        30.00        14.75        19.78           61

               micro        76.72        74.67        75.68          971

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
Train epoch 75: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.58it/s]
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
Evaluate epoch 76: 100%|████████████████████████| 30/30 [00:04<00:00,  7.12it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.82        89.81        87.25          834
                   t        86.88        90.34        88.57          828

               micro        85.84        90.07        87.90         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        71.94        67.57        69.69          148
                 POS        75.96        82.94        79.30          762
                 NEU        27.27         9.84        14.46           61

               micro        74.32        76.00        75.15          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        71.94        67.57        69.69          148
                 POS        75.96        82.94        79.30          762
                 NEU        27.27         9.84        14.46           61

               micro        74.32        76.00        75.15          971

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
Train epoch 76: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.57it/s]
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
Evaluate epoch 77: 100%|████████████████████████| 30/30 [00:04<00:00,  7.38it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.63        89.33        87.44          834
                   t        85.71        91.30        88.42          828

               micro        85.67        90.31        87.93         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.24        68.24        68.24          148
                 POS        78.47        82.28        80.33          762
                 NEU        26.32         8.20        12.50           61

               micro        75.88        75.49        75.68          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.24        68.24        68.24          148
                 POS        78.47        82.28        80.33          762
                 NEU        26.32         8.20        12.50           61

               micro        75.88        75.49        75.68          971

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
Train epoch 77: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 78: 100%|████████████████████████| 30/30 [00:04<00:00,  7.08it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.52        91.13        87.16          834
                   t        86.54        90.82        88.63          828

               micro        84.99        90.97        87.88         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        67.97        70.27        69.10          148
                 POS        75.69        82.94        79.15          762
                 NEU        26.92        11.48        16.09           61

               micro        73.27        76.52        74.86          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        67.97        70.27        69.10          148
                 POS        75.69        82.94        79.15          762
                 NEU        26.92        11.48        16.09           61

               micro        73.27        76.52        74.86          971

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
Train epoch 78: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.56it/s]
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
Evaluate epoch 79: 100%|████████████████████████| 30/30 [00:04<00:00,  7.14it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        81.65        92.33        86.66          834
                   t        84.62        91.67        88.00          828

               micro        83.10        92.00        87.32         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.85        72.97        69.23          148
                 POS        75.06        83.33        78.98          762
                 NEU        22.22         9.84        13.64           61

               micro        72.23        77.14        74.60          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.85        72.97        69.23          148
                 POS        75.06        83.33        78.98          762
                 NEU        22.22         9.84        13.64           61

               micro        72.23        77.14        74.60          971

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
Train epoch 79: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.55it/s]
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
Evaluate epoch 80: 100%|████████████████████████| 30/30 [00:04<00:00,  7.00it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.51        90.29        87.30          834
                   t        85.62        91.30        88.37          828

               micro        85.06        90.79        87.83         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.03        71.62        68.17          148
                 POS        78.25        82.15        80.15          762
                 NEU        30.43        11.48        16.67           61

               micro        74.95        76.11        75.52          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.03        71.62        68.17          148
                 POS        78.25        82.15        80.15          762
                 NEU        30.43        11.48        16.67           61

               micro        74.95        76.11        75.52          971

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
Train epoch 80: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.51it/s]
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
Evaluate epoch 81: 100%|████████████████████████| 30/30 [00:04<00:00,  6.89it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.21        90.89        86.88          834
                   t        84.47        91.30        87.75          828

               micro        83.83        91.10        87.31         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        66.03        69.59        67.76          148
                 POS        74.70        82.94        78.61          762
                 NEU        23.08         9.84        13.79           61

               micro        72.08        76.31        74.14          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        66.03        69.59        67.76          148
                 POS        74.70        82.94        78.61          762
                 NEU        23.08         9.84        13.79           61

               micro        72.08        76.31        74.14          971

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
Train epoch 81: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 82: 100%|████████████████████████| 30/30 [00:04<00:00,  7.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        82.28        91.85        86.80          834
                   t        85.89        91.18        88.46          828

               micro        84.03        91.52        87.62         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.66        73.65        69.43          148
                 POS        74.91        83.46        78.96          762
                 NEU        26.92        11.48        16.09           61

               micro        72.24        77.45        74.75          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.66        73.65        69.43          148
                 POS        74.91        83.46        78.96          762
                 NEU        26.92        11.48        16.09           61

               micro        72.24        77.45        74.75          971

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
Train epoch 82: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.52it/s]
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
Evaluate epoch 83: 100%|████████████████████████| 30/30 [00:04<00:00,  6.94it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.11        89.81        87.40          834
                   t        87.08        91.18        89.09          828

               micro        86.09        90.49        88.24         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        71.63        68.24        69.90          148
                 POS        76.90        82.15        79.44          762
                 NEU        28.00        11.48        16.28           61

               micro        74.90        75.59        75.24          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        71.63        68.24        69.90          148
                 POS        76.90        82.15        79.44          762
                 NEU        28.00        11.48        16.28           61

               micro        74.90        75.59        75.24          971

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
Train epoch 83: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.50it/s]
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
Evaluate epoch 84: 100%|████████████████████████| 30/30 [00:04<00:00,  6.85it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.98        90.53        87.13          834
                   t        86.01        91.30        88.58          828

               micro        84.98        90.91        87.85         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.21        69.59        68.90          148
                 POS        73.62        83.86        78.40          762
                 NEU        24.00         9.84        13.95           61

               micro        71.65        77.03        74.24          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.21        69.59        68.90          148
                 POS        73.62        83.86        78.40          762
                 NEU        24.00         9.84        13.95           61

               micro        71.65        77.03        74.24          971

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
Train epoch 84: 100%|███████████████████████████| 79/79 [00:30<00:00,  2.55it/s]
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
Evaluate epoch 85: 100%|████████████████████████| 30/30 [00:04<00:00,  6.91it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.19        90.77        86.81          834
                   t        85.76        91.67        88.62          828

               micro        84.46        91.22        87.71         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        69.59        69.59        69.59          148
                 POS        78.82        82.55        80.64          762
                 NEU        24.24        13.11        17.02           61

               micro        75.59        76.21        75.90          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        69.59        69.59        69.59          148
                 POS        78.82        82.55        80.64          762
                 NEU        24.24        13.11        17.02           61

               micro        75.59        76.21        75.90          971

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
Train epoch 85: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 86: 100%|████████████████████████| 30/30 [00:04<00:00,  7.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.07        91.13        87.46          834
                   t        86.97        91.06        88.97          828

               micro        85.49        91.10        88.20         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        72.66        68.24        70.38          148
                 POS        75.23        84.51        79.60          762
                 NEU        27.27         9.84        14.46           61

               micro        73.84        77.34        75.55          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        72.66        68.24        70.38          148
                 POS        75.23        84.51        79.60          762
                 NEU        27.27         9.84        14.46           61

               micro        73.84        77.34        75.55          971

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
Train epoch 86: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 87: 100%|████████████████████████| 30/30 [00:04<00:00,  7.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.58        90.77        87.57          834
                   t        86.19        91.18        88.62          828

               micro        85.38        90.97        88.09         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        69.66        68.24        68.94          148
                 POS        77.61        82.81        80.13          762
                 NEU        27.59        13.11        17.78           61

               micro        74.97        76.21        75.59          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        69.66        68.24        68.94          148
                 POS        77.61        82.81        80.13          762
                 NEU        27.59        13.11        17.78           61

               micro        74.97        76.21        75.59          971

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
Train epoch 87: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.55it/s]
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
Evaluate epoch 88: 100%|████████████████████████| 30/30 [00:04<00:00,  7.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.61        90.29        87.35          834
                   t        85.84        91.55        88.60          828

               micro        85.22        90.91        87.98         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.21        69.59        68.90          148
                 POS        75.65        83.99        79.60          762
                 NEU        21.43         9.84        13.48           61

               micro        73.07        77.14        75.05          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.21        69.59        68.90          148
                 POS        75.65        83.99        79.60          762
                 NEU        21.43         9.84        13.48           61

               micro        73.07        77.14        75.05          971

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
Train epoch 88: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.53it/s]
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
Evaluate epoch 89: 100%|████████████████████████| 30/30 [00:04<00:00,  7.10it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.34        90.05        87.63          834
                   t        86.65        90.94        88.74          828

               micro        85.99        90.49        88.19         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        70.07        69.59        69.83          148
                 POS        79.77        81.76        80.75          762
                 NEU        28.95        18.03        22.22           61

               micro        76.29        75.90        76.10          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        70.07        69.59        69.83          148
                 POS        79.77        81.76        80.75          762
                 NEU        28.95        18.03        22.22           61

               micro        76.29        75.90        76.10          971

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
Train epoch 89: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.53it/s]
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
Evaluate epoch 90: 100%|████████████████████████| 30/30 [00:04<00:00,  7.20it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.37        90.29        87.76          834
                   t        86.49        90.46        88.43          828

               micro        85.93        90.37        88.09         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        67.76        69.59        68.67          148
                 POS        77.26        82.94        80.00          762
                 NEU        31.03        14.75        20.00           61

               micro        74.47        76.62        75.53          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        67.76        69.59        68.67          148
                 POS        77.26        82.94        80.00          762
                 NEU        31.03        14.75        20.00           61

               micro        74.47        76.62        75.53          971

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
Train epoch 90: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.55it/s]
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
Evaluate epoch 91: 100%|████████████████████████| 30/30 [00:04<00:00,  7.06it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.00        90.65        87.20          834
                   t        84.29        92.03        87.99          828

               micro        84.15        91.34        87.59         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        67.08        72.97        69.90          148
                 POS        72.37        84.91        78.14          762
                 NEU        26.09         9.84        14.29           61

               micro        70.59        78.37        74.28          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        67.08        72.97        69.90          148
                 POS        72.37        84.91        78.14          762
                 NEU        26.09         9.84        14.29           61

               micro        70.59        78.37        74.28          971

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
Train epoch 91: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.51it/s]
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
Evaluate epoch 92: 100%|████████████████████████| 30/30 [00:04<00:00,  7.10it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.74        90.77        87.11          834
                   t        85.71        91.30        88.42          828

               micro        84.71        91.03        87.76         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.41        70.27        67.75          148
                 POS        74.59        83.60        78.84          762
                 NEU        28.57        13.11        17.98           61

               micro        71.95        77.14        74.45          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.41        70.27        67.75          148
                 POS        74.59        83.60        78.84          762
                 NEU        28.57        13.11        17.98           61

               micro        71.95        77.14        74.45          971

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
Train epoch 92: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.52it/s]
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
Evaluate epoch 93: 100%|████████████████████████| 30/30 [00:04<00:00,  7.08it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.85        90.89        87.23          834
                   t        86.09        91.18        88.56          828

               micro        84.95        91.03        87.89         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        70.29        65.54        67.83          148
                 POS        75.51        82.94        79.05          762
                 NEU        27.59        13.11        17.78           61

               micro        73.41        75.90        74.63          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        70.29        65.54        67.83          148
                 POS        75.51        82.94        79.05          762
                 NEU        27.59        13.11        17.78           61

               micro        73.41        75.90        74.63          971

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
Train epoch 93: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.53it/s]
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
Evaluate epoch 94: 100%|████████████████████████| 30/30 [00:04<00:00,  7.06it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.21        90.17        87.09          834
                   t        85.07        90.82        87.85          828

               micro        84.64        90.49        87.47         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.00        70.27        67.53          148
                 POS        76.00        82.68        79.20          762
                 NEU        27.27         9.84        14.46           61

               micro        73.19        76.21        74.67          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.00        70.27        67.53          148
                 POS        76.00        82.68        79.20          762
                 NEU        27.27         9.84        14.46           61

               micro        73.19        76.21        74.67          971

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
Train epoch 94: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.53it/s]
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
Evaluate epoch 95: 100%|████████████████████████| 30/30 [00:04<00:00,  7.03it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.09        90.65        87.25          834
                   t        85.14        91.30        88.11          828

               micro        84.61        90.97        87.68         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        66.03        69.59        67.76          148
                 POS        75.15        83.33        79.03          762
                 NEU        26.92        11.48        16.09           61

               micro        72.54        76.73        74.57          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        66.03        69.59        67.76          148
                 POS        75.15        83.33        79.03          762
                 NEU        26.92        11.48        16.09           61

               micro        72.54        76.73        74.57          971

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
Train epoch 95: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.52it/s]
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
Evaluate epoch 96: 100%|████████████████████████| 30/30 [00:04<00:00,  7.14it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.72        90.41        87.47          834
                   t        85.84        90.82        88.26          828

               micro        85.28        90.61        87.86         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.67        69.59        69.13          148
                 POS        74.97        83.33        78.93          762
                 NEU        23.33        11.48        15.38           61

               micro        72.54        76.73        74.57          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.67        69.59        69.13          148
                 POS        74.97        83.33        78.93          762
                 NEU        23.33        11.48        15.38           61

               micro        72.54        76.73        74.57          971

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
Train epoch 96: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.53it/s]
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
Evaluate epoch 97: 100%|████████████████████████| 30/30 [00:04<00:00,  7.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.85        90.89        87.23          834
                   t        85.23        91.30        88.16          828

               micro        84.53        91.10        87.69         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        67.55        68.92        68.23          148
                 POS        74.42        83.99        78.91          762
                 NEU        25.93        11.48        15.91           61

               micro        72.16        77.14        74.56          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        67.55        68.92        68.23          148
                 POS        74.42        83.99        78.91          762
                 NEU        25.93        11.48        15.91           61

               micro        72.16        77.14        74.56          971

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
Train epoch 97: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.52it/s]
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
Evaluate epoch 98: 100%|████████████████████████| 30/30 [00:04<00:00,  7.09it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.04        90.29        87.05          834
                   t        86.27        91.06        88.60          828

               micro        85.14        90.67        87.82         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.46        68.92        68.69          148
                 POS        77.06        82.41        79.64          762
                 NEU        25.00        11.48        15.73           61

               micro        74.29        75.90        75.09          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.46        68.92        68.69          148
                 POS        77.06        82.41        79.64          762
                 NEU        25.00        11.48        15.73           61

               micro        74.29        75.90        75.09          971

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
Train epoch 98: 100%|███████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 99: 100%|████████████████████████| 30/30 [00:04<00:00,  7.16it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.21        90.17        87.09          834
                   t        84.44        91.79        87.96          828

               micro        84.33        90.97        87.53         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        69.18        68.24        68.71          148
                 POS        73.79        83.86        78.50          762
                 NEU        23.33        11.48        15.38           61

               micro        71.69        76.93        74.22          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        69.18        68.24        68.71          148
                 POS        73.79        83.86        78.50          762
                 NEU        23.33        11.48        15.38           61

               micro        71.69        76.93        74.22          971

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
Train epoch 102: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.51it/s]
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
Evaluate epoch 103: 100%|███████████████████████| 30/30 [00:04<00:00,  7.20it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.37        90.29        87.76          834
                   t        86.82        90.70        88.72          828

               micro        86.09        90.49        88.24         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.46        68.92        68.69          148
                 POS        77.10        83.07        79.97          762
                 NEU        28.57        13.11        17.98           61

               micro        74.45        76.52        75.47          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.46        68.92        68.69          148
                 POS        77.10        83.07        79.97          762
                 NEU        28.57        13.11        17.98           61

               micro        74.45        76.52        75.47          971

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
Train epoch 103: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.52it/s]
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
Evaluate epoch 104: 100%|███████████████████████| 30/30 [00:04<00:00,  6.90it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.33        91.13        87.06          834
                   t        85.04        91.30        88.06          828

               micro        84.18        91.22        87.55         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        65.82        70.27        67.97          148
                 POS        74.22        83.86        78.74          762
                 NEU        25.93        11.48        15.91           61

               micro        71.70        77.24        74.37          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        65.82        70.27        67.97          148
                 POS        74.22        83.86        78.74          762
                 NEU        25.93        11.48        15.91           61

               micro        71.70        77.24        74.37          971

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
Train epoch 104: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.51it/s]
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
Evaluate epoch 105: 100%|███████████████████████| 30/30 [00:04<00:00,  6.86it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.42        90.29        87.25          834
                   t        86.16        90.94        88.48          828

               micro        85.28        90.61        87.86         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        67.76        69.59        68.67          148
                 POS        76.14        82.94        79.40          762
                 NEU        26.92        11.48        16.09           61

               micro        73.61        76.42        74.99          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        67.76        69.59        68.67          148
                 POS        76.14        82.94        79.40          762
                 NEU        26.92        11.48        16.09           61

               micro        73.61        76.42        74.99          971

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
Train epoch 105: 100%|██████████████████████████| 79/79 [00:33<00:00,  2.35it/s]
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
Evaluate epoch 106: 100%|███████████████████████| 30/30 [00:04<00:00,  6.96it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.57        90.89        87.08          834
                   t        85.49        91.06        88.19          828

               micro        84.52        90.97        87.63         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        67.76        69.59        68.67          148
                 POS        77.01        83.07        79.92          762
                 NEU        24.24        13.11        17.02           61

               micro        73.88        76.62        75.23          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        67.76        69.59        68.67          148
                 POS        77.01        83.07        79.92          762
                 NEU        24.24        13.11        17.02           61

               micro        73.88        76.62        75.23          971

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
Train epoch 106: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.51it/s]
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
Evaluate epoch 107: 100%|███████████████████████| 30/30 [00:04<00:00,  6.94it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.38        90.65        87.40          834
                   t        85.57        90.94        88.17          828

               micro        84.97        90.79        87.78         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.92        68.92        68.92          148
                 POS        76.64        82.68        79.55          762
                 NEU        25.00        13.11        17.20           61

               micro        73.85        76.21        75.01          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.92        68.92        68.92          148
                 POS        76.64        82.68        79.55          762
                 NEU        25.00        13.11        17.20           61

               micro        73.85        76.21        75.01          971

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
Train epoch 107: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.51it/s]
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
Evaluate epoch 108: 100%|███████████████████████| 30/30 [00:04<00:00,  7.18it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.72        90.41        87.47          834
                   t        86.06        90.94        88.43          828

               micro        85.38        90.67        87.95         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.00        68.92        68.46          148
                 POS        77.20        83.07        80.03          762
                 NEU        28.57        13.11        17.98           61

               micro        74.45        76.52        75.47          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.00        68.92        68.46          148
                 POS        77.20        83.07        80.03          762
                 NEU        28.57        13.11        17.98           61

               micro        74.45        76.52        75.47          971

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
Train epoch 108: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.53it/s]
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
Evaluate epoch 109: 100%|███████████████████████| 30/30 [00:04<00:00,  7.08it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.63        90.05        87.78          834
                   t        86.75        90.94        88.80          828

               micro        86.19        90.49        88.29         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        70.34        68.92        69.62          148
                 POS        78.79        81.89        80.31          762
                 NEU        28.57        13.11        17.98           61

               micro        76.06        75.59        75.83          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        70.34        68.92        69.62          148
                 POS        78.79        81.89        80.31          762
                 NEU        28.57        13.11        17.98           61

               micro        76.06        75.59        75.83          971

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
Train epoch 109: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.52it/s]
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
Evaluate epoch 110: 100%|███████████████████████| 30/30 [00:04<00:00,  7.04it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.53        90.41        87.37          834
                   t        85.49        91.06        88.19          828

               micro        85.01        90.73        87.78         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        69.18        68.24        68.71          148
                 POS        75.44        83.46        79.25          762
                 NEU        25.00        11.48        15.73           61

               micro        73.16        76.62        74.85          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        69.18        68.24        68.71          148
                 POS        75.44        83.46        79.25          762
                 NEU        25.00        11.48        15.73           61

               micro        73.16        76.62        74.85          971

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
Train epoch 110: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.53it/s]
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
Evaluate epoch 111: 100%|███████████████████████| 30/30 [00:04<00:00,  7.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.80        90.29        87.46          834
                   t        85.76        90.94        88.28          828

               micro        85.28        90.61        87.86         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        69.39        68.92        69.15          148
                 POS        76.95        82.81        79.77          762
                 NEU        25.93        11.48        15.91           61

               micro        74.45        76.21        75.32          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        69.39        68.92        69.15          148
                 POS        76.95        82.81        79.77          762
                 NEU        25.93        11.48        15.91           61

               micro        74.45        76.21        75.32          971

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
Train epoch 111: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 112: 100%|███████████████████████| 30/30 [00:04<00:00,  6.98it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.97        90.17        87.49          834
                   t        85.67        90.94        88.22          828

               micro        85.32        90.55        87.86         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        70.07        69.59        69.83          148
                 POS        74.33        83.20        78.51          762
                 NEU        26.92        11.48        16.09           61

               micro        72.51        76.62        74.51          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        70.07        69.59        69.83          148
                 POS        74.33        83.20        78.51          762
                 NEU        26.92        11.48        16.09           61

               micro        72.51        76.62        74.51          971

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
Train epoch 112: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 113: 100%|███████████████████████| 30/30 [00:04<00:00,  7.00it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.91        90.65        87.15          834
                   t        85.29        91.06        88.08          828

               micro        84.59        90.85        87.61         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        66.67        68.92        67.77          148
                 POS        75.03        83.60        79.08          762
                 NEU        24.14        11.48        15.56           61

               micro        72.36        76.83        74.53          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        66.67        68.92        67.77          148
                 POS        75.03        83.60        79.08          762
                 NEU        24.14        11.48        15.56           61

               micro        72.36        76.83        74.53          971

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
Train epoch 113: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 114: 100%|███████████████████████| 30/30 [00:04<00:00,  6.92it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.91        90.65        87.15          834
                   t        85.76        90.94        88.28          828

               micro        84.82        90.79        87.71         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.92        68.92        68.92          148
                 POS        76.42        82.94        79.55          762
                 NEU        25.00        11.48        15.73           61

               micro        73.88        76.31        75.08          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.92        68.92        68.92          148
                 POS        76.42        82.94        79.55          762
                 NEU        25.00        11.48        15.73           61

               micro        73.88        76.31        75.08          971

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
Train epoch 114: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 115: 100%|███████████████████████| 30/30 [00:04<00:00,  7.10it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.30        90.77        87.41          834
                   t        85.68        91.06        88.29          828

               micro        84.98        90.91        87.85         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        67.55        68.92        68.23          148
                 POS        75.59        83.73        79.45          762
                 NEU        25.00        11.48        15.73           61

               micro        73.02        76.93        74.92          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        67.55        68.92        68.23          148
                 POS        75.59        83.73        79.45          762
                 NEU        25.00        11.48        15.73           61

               micro        73.02        76.93        74.92          971

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
Train epoch 115: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.53it/s]
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
Evaluate epoch 116: 100%|███████████████████████| 30/30 [00:04<00:00,  7.03it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.09        90.65        87.25          834
                   t        85.49        91.06        88.19          828

               micro        84.78        90.85        87.71         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        67.11        68.92        68.00          148
                 POS        75.51        83.33        79.23          762
                 NEU        24.14        11.48        15.56           61

               micro        72.80        76.62        74.66          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        67.11        68.92        68.00          148
                 POS        75.51        83.33        79.23          762
                 NEU        24.14        11.48        15.56           61

               micro        72.80        76.62        74.66          971

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
Train epoch 116: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.53it/s]
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
Evaluate epoch 117: 100%|███████████████████████| 30/30 [00:04<00:00,  7.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.83        90.77        87.16          834
                   t        85.39        91.06        88.14          828

               micro        84.60        90.91        87.65         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.21        69.59        68.90          148
                 POS        74.68        83.60        78.89          762
                 NEU        25.00        11.48        15.73           61

               micro        72.38        76.93        74.59          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.21        69.59        68.90          148
                 POS        74.68        83.60        78.89          762
                 NEU        25.00        11.48        15.73           61

               micro        72.38        76.93        74.59          971

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
Train epoch 117: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.54it/s]
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
Evaluate epoch 118: 100%|███████████████████████| 30/30 [00:04<00:00,  6.96it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.47        90.65        87.45          834
                   t        85.58        91.06        88.24          828

               micro        85.02        90.85        87.84         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        69.39        68.92        69.15          148
                 POS        75.44        83.46        79.25          762
                 NEU        24.14        11.48        15.56           61

               micro        73.11        76.73        74.87          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        69.39        68.92        69.15          148
                 POS        75.44        83.46        79.25          762
                 NEU        24.14        11.48        15.56           61

               micro        73.11        76.73        74.87          971

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
Train epoch 118: 100%|██████████████████████████| 79/79 [00:31<00:00,  2.52it/s]
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
Evaluate epoch 119: 100%|███████████████████████| 30/30 [00:04<00:00,  7.15it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.28        90.65        87.35          834
                   t        85.58        91.06        88.24          828

               micro        84.93        90.85        87.79         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        69.39        68.92        69.15          148
                 POS        75.09        83.46        79.06          762
                 NEU        25.00        11.48        15.73           61

               micro        72.90        76.73        74.76          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        69.39        68.92        69.15          148
                 POS        75.09        83.46        79.06          762
                 NEU        25.00        11.48        15.73           61

               micro        72.90        76.73        74.76          971

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
Train epoch 119: 100%|██████████████████████████| 79/79 [00:32<00:00,  2.46it/s]
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
Evaluate epoch 120: 100%|███████████████████████| 30/30 [00:04<00:00,  7.14it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        84.19        90.65        87.30          834
                   t        85.58        91.06        88.24          828

               micro        84.88        90.85        87.77         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        68.92        68.92        68.92          148
                 POS        75.09        83.46        79.06          762
                 NEU        25.00        11.48        15.73           61

               micro        72.83        76.73        74.72          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        68.92        68.92        68.92          148
                 POS        75.09        83.46        79.06          762
                 NEU        25.00        11.48        15.73           61

               micro        72.83        76.73        74.72          971
