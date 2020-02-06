# Adapted from https://github.com/corenel/pytorch-adda

"""Pre-train encoder and classifier for source dataset."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import params
from utils import make_variable, save_model


def train_src(encoder, classifier, data_loader, data_loader_test):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        
        encoder.train()
        classifier.train()

        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.item()))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(encoder, classifier, data_loader_test)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # evaluate network
    predictions = []
    gts = []

    with torch.no_grad():
        for (images, labels) in data_loader:
            images = make_variable(images).cuda()
            labels = make_variable(labels)

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