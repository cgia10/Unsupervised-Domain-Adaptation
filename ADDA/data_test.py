import os
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


# Used to load test images in target domain, to generate predictions

class DATA(Dataset):
    def __init__(self, args):
        
        self.img_dir = args.data_dir                            

        # Filenames of all files in img directory, as a list
        img_names = sorted(os.listdir(self.img_dir))

        # self.data is a list with length equal to the number of images. Each element is the filepath to an image
        self.data = []
        for i in range(len(img_names)):
            self.data.append(os.path.join(self.img_dir, img_names[i]))
        
        # Transform the image
        self.transform = transforms.Compose([
            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])
        

    def __getitem__(self, idx):
        
        # Get image
        img_path = self.data[idx]

        # Read image
        img = Image.open(img_path).convert('RGB')

        return self.transform(img)


    def __len__(self):
        return len(self.data)
