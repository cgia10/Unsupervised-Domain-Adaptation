#https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html

import os
import numpy as np
import argparse
import math
import random
import pandas as pd
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import params
from models import LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed, make_variable


# ---------
#  Parser
# ---------

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--source", type=str, default="MNISTM", help="name of source dataset")
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

print("---> preparing dataloaders...")

if opt.source == "MNISTM":
    dataloader_source_test = get_data_loader("MNIST", train=False)
    dataloader_target_test = get_data_loader("SVHN", train=False)
elif opt.source == "SVHN":
    dataloader_source_test = get_data_loader("SVHN", train=False)
    dataloader_target_test = get_data_loader("MNIST", train=False)

# ------------
#  Load Model
# ------------

print("---> preparing model...")

target_encoder = LeNetEncoder()

# Select model that was trained on the correct dataset
if opt.source == "SVHN":
    target_encoder_name = "ADDA-target-encoder-SVHN_MNISTM.pt"
    target_encoder_pth = os.path.join(opt.save_dir, target_encoder_name)

elif opt.source == "MNISTM":
    target_encoder_name = "ADDA-target-encoder-MNISTM_SVHN.pt"
    target_encoder_pth = os.path.join(opt.save_dir, target_encoder_name)

# Load model
if os.path.exists(target_encoder_pth):
    print("---> found previously saved {}, loading checkpoint...".format(target_encoder_name))
    target_encoder.load_state_dict( torch.load(target_encoder_pth) )
else:
    print("Error: target encoder not loaded")

# Move to GPU
if cuda:
    target_encoder = target_encoder.cuda()

# ----------
#  Features
# ----------

print("---> generating features...")

target_encoder.eval()

# Load data from source
with torch.no_grad():   # do not need to calculate information for gradient during eval
    for idx, (imgs, label) in enumerate(dataloader_source_test):
        
        images = make_variable(imgs).cuda()
        source_features = target_encoder(images).cpu().numpy()          # tensor/numpy array, size = batch size x 500 (each row is an image feature)
        source_class = label.cpu().numpy()                              # numpy vector, size = batch size

        if idx == 0:
            break

# Load data from target
with torch.no_grad():   # do not need to calculate information for gradient during eval
    for idx, (imgs, label) in enumerate(dataloader_target_test):
        
        images = make_variable(imgs).cuda()
        target_features = target_encoder(images).cpu().numpy()          # tensor/numpy array, size = batch size x 500 (each row is an image feature)
        target_class = label.cpu().numpy()                              # numpy vector, size = batch size

        if idx == 0:
            break


# -------
#  t-SNE
# -------

# Combine source and target features/classes into one array
combined_features = np.vstack( (source_features, target_features) )     # 2*batch size x 500
combined_class = np.hstack( (source_class, target_class) )              # 2*batch size

# Combine source and target domains into one array
source_domain = np.zeros(200, dtype=np.int16)
target_domain = np.ones(200, dtype=np.int16)
combined_domain = np.hstack( (source_domain, target_domain) )           # 2*batch size

# Perform t-SNE on features
print("---> performing t-SNE...")    
tsne = TSNE(n_components=2)
features_tsne = tsne.fit_transform(combined_features)                   # numpy array, batch size x 2

# Assign different colors for each class and domain
colors_class = []
colors_domain = []

for i in range(features_tsne.shape[0]):
    if combined_class[i] == 0:
        colors_class.append('k')
    elif combined_class[i] == 1:
        colors_class.append('r')
    elif combined_class[i] == 2:
        colors_class.append('g')
    elif combined_class[i] == 3:
        colors_class.append('b')
    elif combined_class[i] == 4:
        colors_class.append('c')
    elif combined_class[i] == 5:
        colors_class.append('m')
    elif combined_class[i] == 6:
        colors_class.append('pink')
    elif combined_class[i] == 7:
        colors_class.append('gold')
    elif combined_class[i] == 8:
        colors_class.append('cyan')
    elif combined_class[i] == 9:
        colors_class.append('orange')

    # Black = source domain, red = target domain
    if combined_domain[i] == 0:
        colors_domain.append('k')
    elif combined_domain[i] == 1:
        colors_domain.append('r')

# Plot t-SNE features, with color indicating class
plt.figure()
plt.scatter(features_tsne[:,0], features_tsne[:,1], s=4, c=colors_class)
plt.savefig("class")

# Plot t-SNE features, with color indicating domain
plt.figure()
plt.scatter(features_tsne[:,0], features_tsne[:,1], s=4, c=colors_domain)
plt.savefig("domain")

print("***** Plots Saved *****")
plt.close("all")