from data_extract import extract_data
import os
import torch
import torchvision
from fish4knowledgeDataset import fish4knowledge
import utils
import helloworld
import numpy
from PIL import Image, ImageDraw, ImageFilter
from coco_utils_imported import draw_segmentation_map, get_outputs
import cv2
import argparse
import torchvision.datasets as dset
import torchvision.transforms as T
from coco_utils_imported import FISH4KNOWLEDGE_INSTANCE_CATEGORY_NAMES, COCO_INSTANCE_CATEGORY_NAMES



if __name__ == '__main__':
    root = 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\data\jrchive'
    cocopath2data= 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\data\jal2017\jal2017'
    cocopath2json= 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\data\jnnotations_trainval2017\jnnotations\instances_val2017.json'

    trans = torchvision.transforms.ToTensor()
    dataset = fish4knowledge(root, trans)
    dataset_coco = dset.CocoDetection(root = cocopath2data, annFile = cocopath2json)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)
    data_loader_coco = torch.utils.data.DataLoader(
        dataset_coco, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    # initialize the model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                            num_classes=91)
    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the modle on to the computation device and set to eval mode
    model.to(device).eval()
    
    image, target = next(iter(data_loader))
    image = image[0]
    tensor_to_img = T.ToPILImage()
    image = tensor_to_img(image)
    orig_image = image.copy()

    image = trans(image)

    image = image.unsqueeze(0).to(device)
    masks, boxes, labels = get_outputs(image, model, 0.5, COCO_INSTANCE_CATEGORY_NAMES)
    result = draw_segmentation_map(orig_image, masks, boxes, labels, COCO_INSTANCE_CATEGORY_NAMES)

    # visualize the image
    cv2.imshow('Segmented image', result)
    cv2.cv2.waitKey(0)
