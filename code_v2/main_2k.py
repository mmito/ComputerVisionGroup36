from fish4knowledgeDataset import fish4knowledge
import torch
import torchvision
#from segmentation_model import create_model
import torchvision.datasets as dset
import cv2
from PIL import Image
from torchinfo import summary
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision import datasets
import torch.utils.data
import torchvision
from .coco import build as build_coco

def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

def main():
    # Classification threshold
    threshold = 0.1
    cat_names = FISH4KNOWLEDGE_INSTANCE_CATEGORY_NAMES
    trans = torchvision.transforms.ToTensor()

    modelpath = 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\code\jishmodel'
    data_folder = 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\data\jish2k'
    json_file = 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\data\jish2k/fish4knowledge_2K_annotations.json'
    #dataset = torchvision.datasets.CocoDetection(
    #    root=data_folder, annFile=json_file, transform=torchvision.transforms.ToTensor()
    #)
    dataset = build_dataset(image_set='train')
    print('Length train set: ', len(dataset))

    # Create model
    num_classes = 50
    model = create_model(num_classes)   
    try:
        model.load_state_dict(torch.load(modelpath))
    except:
        print('Warning: No model loaded')
        pass
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        #engine.evaluate(model, data_loader, device=device)
        torch.save(model.state_dict(), modelpath)

if __name__ == "__main__":
    main()