# Adapted from https://github.com/corenel/pytorch-adda

import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image
import params


class MNISTM(data.Dataset):

    def __init__(self, root, train=True, transform=None):

        if train:
            mode = "train"
        else:
            mode = "test"

        data_directory = os.path.join(root, "mnistm/")     
        self.img_dir = os.path.join(data_directory, mode + "/")    
        self.labels_dir = data_directory                            

        # Filenames of all files in img directory, as a list
        img_names = sorted(os.listdir(self.img_dir))

        # Read CSV of ground truth classes into a list 'labels'. Element 0 corresponds to 00000.png
        df = pd.read_csv(os.path.join(self.labels_dir, mode + ".csv"))  
        labels = df["label"].tolist()

        # self.data is a list with length equal to the number of images
        # Each element is another list, with first element being the filepath to an image, and the second element being the image class
        self.data = []
        for i in range(len(img_names)):
            self.data.append( [os.path.join(self.img_dir, img_names[i]), labels[i]] )

        # Transform the image
        self.transform = transforms.Compose([
            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])


    def __getitem__(self, index):

        # Get data
        img_path, label = self.data[index]

        # Read image
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), int(label)


    def __len__(self):
        return len(self.data)


def get_mnist(train):

    # dataset and data loader
    mnist_dataset = MNISTM(root=params.data_root, train=train)

    if train:
        mnist_data_loader = torch.utils.data.DataLoader(
            dataset=mnist_dataset,
            batch_size=params.batch_size,
            shuffle=True)
    else:
        mnist_data_loader = torch.utils.data.DataLoader(
            dataset=mnist_dataset,
            batch_size=params.batch_size,
            shuffle=False)

    return mnist_data_loader



