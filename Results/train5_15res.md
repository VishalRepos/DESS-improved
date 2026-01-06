Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Parse dataset 'train': 100%|████████████████| 592/592 [00:00<00:00, 1665.48it/s]
Parse dataset 'test': 100%|█████████████████| 320/320 [00:00<00:00, 1603.05it/s]
    15res    8
Some weights of the model checkpoint at microsoft/deberta-v3-base were not used when initializing DebertaV2Model: ['mask_predictions.dense.bias', 'mask_predictions.classifier.weight', 'lm_predictions.lm_head.bias', 'lm_predictions.lm_head.LayerNorm.weight', 'mask_predictions.dense.weight', 'lm_predictions.lm_head.LayerNorm.bias', 'mask_predictions.classifier.bias', 'lm_predictions.lm_head.dense.weight', 'mask_predictions.LayerNorm.weight', 'mask_predictions.LayerNorm.bias', 'lm_predictions.lm_head.dense.bias']
- This IS expected if you are initializing DebertaV2Model from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DebertaV2Model from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Using Enhanced Syntactic GCN with GATv2, SAGE, Chebyshev, EdgeConv, and hybrid fusion
Using Enhanced Semantic GCN with relative position, global context, and multi-scale features
Some weights of the model checkpoint at microsoft/deberta-v3-base were not used when initializing D2E2SModel: ['mask_predictions.dense.bias', 'mask_predictions.classifier.weight', 'lm_predictions.lm_head.bias', 'deberta.embeddings.position_embeddings.weight', 'lm_predictions.lm_head.LayerNorm.weight', 'mask_predictions.dense.weight', 'lm_predictions.lm_head.LayerNorm.bias', 'mask_predictions.classifier.bias', 'lm_predictions.lm_head.dense.weight', 'mask_predictions.LayerNorm.weight', 'mask_predictions.LayerNorm.bias', 'lm_predictions.lm_head.dense.bias']
- This IS expected if you are initializing D2E2SModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing D2E2SModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of D2E2SModel were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['TIN.residual_layer2.3.weight', 'TIN.residual_layer3.0.weight', 'TIN.residual_layer2.0.weight', 'TIN.GatedGCN.conv3.rnn.bias_hh', 'TIN.residual_layer3.2.weight', 'Syn_gcn.gat_layers.0.W.weight', 'TIN.residual_layer4.2.weight', 'TIN.lstm.bias_hh_l1', 'TIN.residual_layer4.0.bias', 'lstm.weight_ih_l1', 'lstm.weight_ih_l0_reverse', 'TIN.feature_fusion.3.bias', 'Sem_gcn.multi_scale.fusion.bias', 'Sem_gcn.W.1.bias', 'lstm.weight_hh_l1_reverse', 'Syn_gcn.sage_layers.1.W.weight', 'TIN.lstm.bias_ih_l0_reverse', 'attention_layer.w_value.weight', 'TIN.residual_layer4.3.weight', 'TIN.residual_layer1.0.weight', 'TIN.feature_fusion.0.bias', 'lstm.bias_hh_l1', 'Sem_gcn.global_context.gate.bias', 'TIN.GatedGCN.conv2.lin.weight', 'lstm.bias_ih_l1', 'TIN.residual_layer3.0.bias', 'lstm.bias_ih_l1_reverse', 'Syn_gcn.fusion.attention.bias', 'Syn_gcn.sage_layers.0.W.bias', 'TIN.residual_layer1.3.weight', 'Sem_gcn.multi_scale.fusion.weight', 'TIN.residual_layer4.0.weight', 'attention_layer.v.weight', 'TIN.lstm.weight_hh_l1', 'Sem_gcn.W.0.weight', 'Sem_gcn.global_context.fc.bias', 'Sem_gcn.multi_scale.scale_weights', 'attention_layer.w_query.bias', 'Sem_gcn.attn.linears.0.weight', 'Sem_gcn.W.1.weight', 'Sem_gcn.attn.linears.0.bias', 'Sem_gcn.W.0.bias', 'senti_classifier.weight', 'Syn_gcn.sage_layers.0.W.weight', 'TIN.residual_layer2.3.bias', 'Sem_gcn.global_context.fc.weight', 'TIN.lstm.weight_hh_l1_reverse', 'TIN.residual_layer2.0.bias', 'Sem_gcn.attn.relative_position_k.weight', 'deberta.embeddings.position_ids', 'TIN.GatedGCN.conv2.bias', 'entity_classifier.bias', 'Syn_gcn.fusion.fusion.bias', 'TIN.lstm.weight_ih_l0_reverse', 'TIN.residual_layer3.2.bias', 'TIN.GatedGCN.conv1.lin.weight', 'Syn_gcn.gat_layers.1.W.weight', 'TIN.lstm.weight_ih_l0', 'Sem_gcn.attn.linears.1.bias', 'TIN.residual_layer2.2.bias', 'fc.weight', 'lstm.bias_ih_l0', 'TIN.GatedGCN.conv3.rnn.weight_hh', 'TIN.residual_layer3.3.weight', 'TIN.feature_fusion.2.bias', 'Sem_gcn.global_context.gate.weight', 'attention_layer.w_value.bias', 'TIN.lstm.bias_ih_l1_reverse', 'TIN.residual_layer3.3.bias', 'Sem_gcn.attn.relative_position_v.weight', 'Syn_gcn.sage_layers.1.W.bias', 'TIN.lstm.weight_hh_l0', 'TIN.feature_fusion.2.weight', 'TIN.lstm.bias_hh_l0_reverse', 'lstm.bias_hh_l0_reverse', 'TIN.lstm.bias_ih_l1', 'TIN.residual_layer4.2.bias', 'lstm.bias_ih_l0_reverse', 'Syn_gcn.gat_layers.1.a', 'TIN.GatedGCN.conv1.bias', 'TIN.residual_layer1.3.bias', 'TIN.residual_layer1.2.weight', 'Syn_gcn.gat_layers.0.a', 'senti_classifier.bias', 'attention_layer.linear_q.bias', 'TIN.GatedGCN.conv3.rnn.weight_ih', 'TIN.lstm.weight_hh_l0_reverse', 'fc.bias', 'lstm.weight_hh_l1', 'TIN.lstm.weight_ih_l1', 'Syn_gcn.fusion.attention.weight', 'lstm.weight_hh_l0_reverse', 'Syn_gcn.fusion.fusion.weight', 'TIN.residual_layer1.0.bias', 'TIN.residual_layer2.2.weight', 'lstm.weight_ih_l0', 'TIN.GatedGCN.conv3.rnn.bias_ih', 'Sem_gcn.attn.linears.1.weight', 'TIN.lstm.bias_ih_l0', 'TIN.lstm.weight_ih_l1_reverse', 'size_embeddings.weight', 'attention_layer.linear_q.weight', 'lstm.weight_hh_l0', 'TIN.GatedGCN.conv3.weight', 'TIN.residual_layer1.2.bias', 'TIN.feature_fusion.3.weight', 'TIN.feature_fusion.0.weight', 'TIN.lstm.bias_hh_l1_reverse', 'entity_classifier.weight', 'lstm.bias_hh_l1_reverse', 'lstm.weight_ih_l1_reverse', 'attention_layer.w_query.weight', 'lstm.bias_hh_l0', 'TIN.residual_layer4.3.bias', 'TIN.lstm.bias_hh_l0']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
2025-12-31 06:41:51.834687: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1767163311.851465    4279 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1767163311.856523    4279 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
Train epoch 0:   0%|                                     | 0/37 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Train epoch 0: 100%|████████████████████████████| 37/37 [00:13<00:00,  2.77it/s]
Evaluate epoch 1:   0%|                                  | 0/20 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Evaluate epoch 1: 100%|█████████████████████████| 20/20 [18:32<00:00, 55.61s/it]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00          459
                   t         1.32        61.40         2.59          430

               micro         1.32        29.70         2.54          889


--- Aspect Sentiment Triplet Extraction ---