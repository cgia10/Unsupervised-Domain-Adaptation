# Adapted from https://github.com/corenel/pytorch-adda

"""Test script to classify target data."""

import torch
import torch.nn as nn
import numpy as np

from utils import make_variable


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # evaluate network
    predictions = []
    gts = []

    with torch.no_grad():
        for (images, labels) in data_loader:
            images = make_variable(images).cuda()
            labels = make_variable(labels).squeeze_()

            preds = classifier(encoder(images))
            pred_cls = preds.data.max(1)[1]

            predictions.append(pred_cls.cpu())
            gts.append(labels.cpu())

    predictions = np.concatenate(predictions)
    gts = np.concatenate(gts)

    # Calculate accuracy
    i = 0
    correct = 0
    total = 0
    for prediction in predictions:
        if prediction == gts[i]:
            correct += 1
        total += 1
        i += 1

    acc = (correct/total) * 100

    print("Acc: %d/%d (%.2f%%)" % (correct, total, acc))
