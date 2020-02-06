import os
import numpy as np
import argparse
import math
import random
import pandas as pd
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import LeNetClassifier, LeNetEncoder
import data_test
from utils import make_variable


# ---------
#  Parser
# ---------

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=40, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--target", type=str, default="svhn", help="name of target dataset")
parser.add_argument("--csv_output_dir", type=str, default="./output/preds.csv", help="path to output .csv file")
parser.add_argument("--data_dir", type=str, default="./data/", help="root path to the testing images")
parser.add_argument("--save_dir", type=str, default="./saved_models/", help="directory where the saved model is located")
opt = parser.parse_args()


# ---------------------
#  Miscellaneous Setup
# ---------------------

# Set up GPU
if torch.cuda.is_available():
    print("GPU found")
    cuda = True
    torch.cuda.set_device(opt.gpu)
else:
    print("GPU not found")
    cuda = False


# ------------
#  Dataloader
# ------------

print("---> preparing dataloader...")

dataloader_target_test = torch.utils.data.DataLoader(
    dataset=data_test.DATA(opt),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu
)


# ------------
#  Load Model
# ------------

print("---> preparing models...")

target_encoder = LeNetEncoder()
source_classifier = LeNetClassifier()

# Select models that were trained on the correct dataset
if opt.target == "mnistm":
    target_encoder_name = "ADDA-target-encoder-SVHN_MNISTM.pt"
    source_classifier_name = "ADDA-source-classifier-SVHN_MNISTM.pt"

    target_encoder_pth = os.path.join(opt.save_dir, target_encoder_name)
    source_classifier_pth = os.path.join(opt.save_dir, source_classifier_name)
elif opt.target == "svhn":
    target_encoder_name = "ADDA-target-encoder-MNISTM_SVHN.pt"
    source_classifier_name = "ADDA-source-classifier-MNISTM_SVHN.pt"

    target_encoder_pth = os.path.join(opt.save_dir, target_encoder_name)
    source_classifier_pth = os.path.join(opt.save_dir, source_classifier_name)
else:
    print("ERROR: unrecognized target dataset name. Please input 'mnistm' or 'svhn', lower case only.")

# Load models
if os.path.exists(target_encoder_pth):
    print("---> found previously saved {}, loading checkpoint...".format(target_encoder_name))
    target_encoder.load_state_dict( torch.load(target_encoder_pth) )
else:
    print("Error: target encoder not loaded")

if os.path.exists(source_classifier_pth):
    print("---> found previously saved {}, loading checkpoint...".format(source_classifier_name))
    source_classifier.load_state_dict( torch.load(source_classifier_pth) )
else:
    print("Error: source classifier not loaded")

# Move to GPU
if cuda:
    target_encoder = target_encoder.cuda()
    source_classifier = source_classifier.cuda()


# -------------
#  Predictions
# -------------

print("---> generating predictions...")

# Set model to evaluation mode
target_encoder.eval()
source_classifier.eval()
predictions = []

# Make predictions
with torch.no_grad():   # do not need to calculate information for gradient during eval
    for imgs in dataloader_target_test:
        
        images = make_variable(imgs).cuda()
        preds = source_classifier(target_encoder(images))
        pred_cls = preds.data.max(1)[1]
        predictions.append(pred_cls.cpu())

preds = np.concatenate(predictions)   # numpy vector, size = number of test data


# ------
#  .CSV
# ------

# Create image names
img_names = []
for i in range(len(preds)):
    if i < 10:
        img_names.append("0000{}.png".format(i))
    elif i < 100:
        img_names.append("000{}.png".format(i))
    elif i < 1000:
        img_names.append("00{}.png".format(i))
    elif i < 10000:
        img_names.append("0{}.png".format(i))
    else:
        img_names.append("{}.png".format(i))

# Save predictions to .csv file
print("---> saving predictions to %s" % opt.csv_output_dir)

i = 0
preds_str = preds.astype(np.str)

with open(opt.csv_output_dir, "w") as file:
    
    # Write header lines
    file.write("image_name")
    file.write(",")
    file.write("label")
    file.write("\n")

    # Write predictions
    for line in img_names:
        file.write(line)
        file.write(",")
        file.write(preds_str[i])
        file.write("\n")
        i += 1


print("***** FINISHED *****")