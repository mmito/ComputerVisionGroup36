from fish4knowledgeDataset import fish4knowledge
from cocoDatasetClass import CocoDataset
import torch
import torchvision
import utils
import engine
from segmentation_model import create_model
import torchvision.datasets as dset
from coco_utils_imported import draw_segmentation_map, get_outputs
import cv2
from PIL import Image
from coco_utils_imported import FISH4KNOWLEDGE_INSTANCE_CATEGORY_NAMES, COCO_INSTANCE_CATEGORY_NAMES
from torchinfo import summary
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision import datasets
import numpy as np
from PIL import Image, ImageDraw

def main():
    # Train or evaluate
    train = 0
    evaluate = 0
    plot_segm = 1

    # Classification threshold
    threshold = 0.1
    cat_names = FISH4KNOWLEDGE_INSTANCE_CATEGORY_NAMES

    # Define transform
    trans = torchvision.transforms.ToTensor()

    modelpath = 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\code\jishmodel100epoch'
    root = 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\data\jarchive'
    json_file = 'fish4knowledge_2K_annotations.json'
    dataset  = fish4knowledge(root, json_file)
    data_loader = torch.utils.data.DataLoader(
         dataset, batch_size=2, shuffle=True, num_workers=1,
         collate_fn=utils.collate_fn)

    # Create model
    num_classes = 38
    model = create_model(num_classes)   
    try:
        model.load_state_dict(torch.load(modelpath,  map_location=torch.device('cpu')))
    except:
        print('Warning: No model loaded')
        pass
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if plot_segm:
        model.to(device).eval()

        while True:
            # Load a datapoint
            image, target = next(iter(data_loader))
            image = image[0]

            tensor_to_img = torchvision.transforms.ToPILImage()
            image_pil = tensor_to_img(image)
            image_pil_mod = image_pil.copy()
            image_pil_gt  = image_pil.copy()

            # Transform image to tensor
            image_tens = trans(image_pil_mod)

            # Feed image to model
            image_model = image_tens.unsqueeze(0).to(device)
            masks_mod, boxes_mod, labels_mod = get_outputs(image_model, model, threshold, cat_names)
            # Show segmented image if any classifications are made
            if len(masks_mod) >= 1:
                result_model = draw_segmentation_map(image_pil_mod, masks_mod, boxes_mod, labels_mod, cat_names)
                cv2.imshow('Segmentation map prediction', result_model)
                target = target[0]
                # Extract masks, labels segmentation maps on correct format
                masks_gt = []
                for mask in target['masks']:
                    mask = mask.tolist()
                    masks_gt.append(np.array(mask))
                labels_raw = target['labels'].tolist()
                labels_gt = []
                for label in labels_raw:
                    labels_gt.append(cat_names[label])
                boxes_raw = target['boxes'].tolist()
                boxes_gt = []
                for box in boxes_raw:
                    boxes_gt.append([(int(box[0]), int(box[1])),(int(box[2]), int(box[3]))])
                result_gt = draw_segmentation_map(image_pil_gt, masks_gt, boxes_gt, labels_gt, cat_names)
                cv2.imshow('Segmentation map ground truth', result_gt)
                cv2.waitKey(0)
            print('No detection')

    if train:   
        # Model in train state
        model.to(device).train()

        # Define optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        # Define learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

        # Training loop
        num_epochs = 1
        for epoch in range(num_epochs):
            print("Start an epoch")
            # train for one epoch, printing every 10 iterations
            engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            #engine.evaluate(model, data_loader_small, device=device)
            torch.save(model.state_dict(), modelpath)
    if evaluate:
        engine.evaluate(model, data_loader, device=device)

if __name__ == "__main__":
    main()