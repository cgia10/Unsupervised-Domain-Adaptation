import numpy as np
import torch

# Function to calculate classification accuracy during training
def evaluate(model, data_loader):

    # Set model to evaluation mode
    model.eval()
    preds = []
    gts = []

    with torch.no_grad(): # do not need to calculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            
            # Generate prediction
            imgs = imgs.cuda()
            class_output = model(input_data=imgs)
            pred = class_output.data.max(1, keepdim=True)[1]
            
            # Create list of predictions and ground truths
            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    # Calculate accuracy
    i = 0
    correct = 0
    total = 0
    for prediction in preds:
        if prediction == gts[i]:
            correct += 1
        total += 1
        i += 1

    acc = (correct/total) * 100
    
    return correct, total, acc