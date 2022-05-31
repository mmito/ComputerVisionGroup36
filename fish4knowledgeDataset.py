import pycocotools
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
import csv
from data_extract import extract_data

class fish4knowledge(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.root = 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\data\jrchive'
        mask_path = os.path.join(root, "annotations")
        img_path = os.path.join(root, "images")
        self.imgs = list(sorted(os.listdir(img_path)))

    def __getitem__(self, idx):
        # load images and masks
        target = {}
        while not "image_id" in target:
            img, target = extract_data(self.imgs[idx], self.root)
            idx = idx + 1
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)