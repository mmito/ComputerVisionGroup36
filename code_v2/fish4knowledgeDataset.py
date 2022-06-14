from operator import mod
import pycocotools
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
import csv
from data_extract import extract_data
from collections import defaultdict
import json
from pathlib import Path
import fnmatch
from PIL import Image, ImageDraw
import torchvision
import cv2
import matplotlib.pyplot as plt

class fish4knowledge(torch.utils.data.Dataset):
    def __init__(self, root, json_file):
        self.root = Path(root)
        self.transforms = torchvision.transforms.ToTensor()
        ann_path = self.root/json_file

        with open(str(ann_path)) as file_obj:
            self.annots = json.load(file_obj)
        self.img_id_to_annot_id = {}

        for image in self.annots['images']:
            for annotation in self.annots['annotations']:
                if image['id'] == annotation['image_id']:
                    if image['id'] in self.img_id_to_annot_id:
                        self.img_id_to_annot_id[image['id']].append(annotation['id'])
                    else:
                        self.img_id_to_annot_id[image['id']] = [annotation['id']]
        self.num_classes = 0

        for annotation in self.annots['annotations']:
            if self.num_classes < annotation['category_id']:
                self.num_classes = annotation['category_id']

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.annots['images'][idx]['file_name'])).convert("RGB")
        while idx not in self.img_id_to_annot_id:
            if idx - 1 < len(self.annots['images']):
                idx = idx + 1
            else:
                idx = 0
        image = Image.open(os.path.join(self.root, self.annots['images'][idx]['file_name'])).convert("RGB")
        width, height = image.size
        target = {}
        target["image_id"] = [idx]
        target["labels"] = []
        target["masks"] = []
        target["boxes"] = []
        target["area"] = []
        target["iscrowd"] = []
        for annot_id in self.img_id_to_annot_id[idx]:
            annot = self.annots['annotations'][annot_id]
            box = [float(annot['bbox'][0]), float(annot['bbox'][1]), float(annot['bbox'][0]) + float(annot['bbox'][2]), float(annot['bbox'][1]) + float(annot['bbox'][3])]
            mask_coord = annot['segmentation']
            mask_coord = list(np.reshape(annot['segmentation'], (int(len(annot['segmentation'][0])/2), int(2))))
            mask_coord_tup = []
            for inst in mask_coord:
                mask_coord_tup.append((inst[0], inst[1]))
            mask_img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(mask_img).polygon(mask_coord_tup, outline = 128, fill=128)
            full_mask = np.array((np.array(mask_img) > 0)).astype(np.uint8).tolist()
            target["labels"].append(annot['category_id'])
            target["masks"].append(full_mask)
            target["boxes"].append(box)
            target["area"].append(annot['area'])
            target["iscrowd"].append(annot['iscrowd'])
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        target["masks"] = torch.as_tensor(np.array(target["masks"]), dtype=torch.uint8)
        target["image_id"] = torch.as_tensor(target["image_id"], dtype=torch.int64)
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["area"] = torch.as_tensor(target["area"], dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.float32)

        image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.annots['images'])