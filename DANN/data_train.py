import os
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


""" MNISTM dataloader """
class MNISTM(Dataset):
    def __init__(self, args, mode):
                
        data_directory = os.path.join(args.data_dir, "mnistm/")     
        self.img_dir = os.path.join(data_directory, mode + "/")     

        self.labels_dir = data_directory                           

        # Filenames of all files in img directory, as a list
        img_names = sorted(os.listdir(self.img_dir))

        # Read CSV of ground truth classes into a list 'labels'. Element 0 corresponds to 00000.png
        df = pd.read_csv(os.path.join(self.labels_dir, mode + ".csv"))  # train.csv or test.csv
        labels = df["label"].tolist()

        # self.data is a list with length equal to the number of images
        # Each element is another list, with first element being the filepath to an image, and the second element being the image class
        self.data = []
        for i in range(len(img_names)):
            self.data.append( [os.path.join(self.img_dir, img_names[i]), labels[i]] )
        
        # Transform the image
        self.transform = transforms.Compose([
            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])
        

    def __getitem__(self, idx):
        
        # Get data
        img_path, label = self.data[idx]

        # Read image
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), int(label)


    def __len__(self):
        return len(self.data)


""" SVHN dataloader """
class SVHN(Dataset):
    def __init__(self, args, mode):
                
        data_directory = os.path.join(args.data_dir, "svhn/")     
        self.img_dir = os.path.join(data_directory, mode + "/")     
        
        self.labels_dir = data_directory                            

        # Filenames of all files in img directory, as a list
        img_names = sorted(os.listdir(self.img_dir))

        # Read CSV of ground truth classes into a list 'labels'. Element 0 corresponds to 00000.png
        df = pd.read_csv(os.path.join(self.labels_dir, mode + ".csv"))  # train.csv or test.csv
        labels = df["label"].tolist()

        # self.data is a list with length equal to the number of images
        # Each element is another list, with first element being the filepath to an image, and the second element being the image class
        self.data = []
        for i in range(len(img_names)):
            self.data.append( [os.path.join(self.img_dir, img_names[i]), labels[i]] )
        
        # Transform the image
        self.transform = transforms.Compose([
            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])
        

    def __getitem__(self, idx):
        
        # Get data
        img_path, label = self.data[idx]

        # Read image
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), int(label)


    def __len__(self):
        return len(self.data)