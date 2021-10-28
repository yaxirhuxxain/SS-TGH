# -*- coding: utf-8 -*-

# Author Yasir Hussain (yaxirhuxxain@yahoo.com)

import torch
from torchmetrics.functional.classification import accuracy
from torchmetrics.functional.classification.f_beta import f1
from torchmetrics.functional.classification.precision_recall import (precision,
                                                                     recall)

# Tell PyTorch to use the GPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _mrr(preds, target, k=1):
    total = target.size(0)
    _, pred = preds.topk(k, 1, True, True)

    hits = torch.nonzero(pred == target)
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks) / total
    return mrr


def compute_metrics(pred):
    scores = {}
    labels = torch.as_tensor(pred.label_ids, device=device)
    logits = torch.as_tensor(pred.predictions, device=device)
    vocab_size = pred.predictions.size(-1)
    for i in [1, 2, 3, 4, 5, 10]:
        scores[f'acc_{i}'] = accuracy(logits, labels, average='weighted', num_classes=vocab_size, top_k=i)
        scores[f'mrr_{i}'] = _mrr(logits, labels, k=i)

    logits = logits.argmax(-1)
    scores['precision'] = precision(logits, labels, average='weighted', num_classes=vocab_size)
    scores['recall'] = recall(logits, labels, average='weighted', num_classes=vocab_size)
    scores['f1'] = f1(logits, labels, average='weighted', num_classes=vocab_size)

    return scores
