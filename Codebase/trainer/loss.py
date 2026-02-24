import torch
import torch.nn.functional as F


class FocalBCEWithLogitsLoss(torch.nn.Module):
    """Focal loss for binary classification with logits.
    
    Reduces the loss contribution from easy negatives, focusing training
    on hard examples. Particularly useful with heavy negative sampling.
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        # p_t = prob for positive class, (1-prob) for negative class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        # alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce_loss
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


class D2E2SLoss():
    def __init__(self, senti_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm, batch_loss_weight=10.0):
        self._senti_criterion = senti_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._batch_loss_weight = batch_loss_weight

    def compute(self, entity_logits, senti_logits, batch_loss, entity_types, senti_types, entity_sample_masks, senti_sample_masks):
        # term loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # sentiment loss
        senti_sample_masks = senti_sample_masks.view(-1).float()
        senti_count = senti_sample_masks.sum()

        if senti_count.item() != 0:
            senti_logits = senti_logits.view(-1, senti_logits.shape[-1])
            senti_types = senti_types.view(-1, senti_types.shape[-1])

            senti_loss = self._senti_criterion(senti_logits, senti_types)
            senti_loss = senti_loss.sum(-1) / senti_loss.shape[-1]
            senti_loss = (senti_loss * senti_sample_masks).sum() / senti_count

            train_loss = entity_loss + senti_loss + self._batch_loss_weight * batch_loss
        else:
            train_loss = entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
