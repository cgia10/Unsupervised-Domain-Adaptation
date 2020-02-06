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

from model_dann import CNNModel
import data_test


# ---------
#  Parser
# ---------

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=40, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--target", type=str, default="svhn", help="name of target dataset")
parser.add_argument("--csv_output_dir", type=str, default="./preds.csv", help="path to output .csv file")
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

print("---> preparing model...")

my_net = CNNModel()

# Select model that was trained on the correct dataset
if opt.target == "mnistm":
    model_name = "dann_svhn_mnistm.pth.tar"
    model_pth = os.path.join(opt.save_dir, model_name)
elif opt.target == "svhn":
    model_name = "dann_mnistm_svhn.pth.tar"
    model_pth = os.path.join(opt.save_dir, model_name)
else:
    print("ERROR: unrecognized target dataset name. Please input 'mnistm' or 'svhn', lower case only.")

# Load model
if os.path.exists(model_pth):
    print("---> found previously saved {}, loading checkpoint...".format(model_name))
    checkpoint = torch.load(model_pth)
    my_net.load_state_dict(checkpoint)
else:
    print("Error: model not loaded")

# Move to GPU
if cuda:
    my_net = my_net.cuda()


# -------------
#  Predictions
# -------------

print("---> generating predictions...")

# Set model to evaluation mode
my_net.eval()
preds = []

# Make predictions
with torch.no_grad():   # do not need to calculate information for gradient during eval
    for idx, imgs in enumerate(dataloader_target_test):
        
        # Generate prediction
        imgs = imgs.cuda()
        class_output, _ = my_net(input_data=imgs, alpha=0)  # tensor, size = batch size x num classes
        pred = class_output.data.max(1, keepdim=True)[1]    # tensor, size = batch size x 1

        # Create list of predictions
        pred = pred.cpu().numpy().squeeze()     # numpy vector, size = 40
        preds.append(pred)                      # list of numpy arrays, length = number of test data / batch size

preds = np.concatenate(preds)   # numpy vector, size = number of test data


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