from fish4knowledgeDataset import fish4knowledge
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

def main():
    # Train or evaluate
    train = 1

    # Coco or fish4knowledge
    fish = 1

    # Classification threshold
    threshold = 0.1
    
    if fish:
        cat_names = FISH4KNOWLEDGE_INSTANCE_CATEGORY_NAMES
    if not fish:
        cat_names = COCO_INSTANCE_CATEGORY_NAMES
    # Define transform
    trans = torchvision.transforms.ToTensor()

    # Import dataset
    if fish:
        root = 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\data\jrchive'
        modelpath = 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\code\jishmodel'
        dataset = fish4knowledge(root, trans)
    if not fish:
        cocopath2data= 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\data\jal2017\jal2017'
        cocopath2json= 'C:\Egne_filer\Filer\Skole\Vaar2022\Q4\CS4245 Seminar Computer Vision by Deep Learning\data\jnnotations_trainval2017\jnnotations\instances_val2017.json'
        dataset = dset.CocoDetection(root = cocopath2data, annFile = cocopath2json)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    # Create model
    num_classes = 24
    if fish:
        model = create_model(num_classes)
        try:
            model.load_state_dict(torch.load(modelpath))
        except:
            print('Warning: No model loaded')
            pass
    if not fish:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not train:
        # Model in eval state
        model.to(device).eval()

        # Load a datapoint
        image, target = next(iter(data_loader))
        image = image[0]

        # Save image copy
        if fish:
            tensor_to_img = torchvision.transforms.ToPILImage()
            image = tensor_to_img(image)
        orig_image = image.copy()
        # Transform image to model format
        image = trans(image)

        # Feed image to model
        image = image.unsqueeze(0).to(device)
        masks, boxes, labels = get_outputs(image, model, threshold, cat_names)

        # Show segmented image if any classifications are made
        if len(masks) >= 1:
            result = draw_segmentation_map(orig_image, masks, boxes, labels, cat_names)
            cv2.imshow('Segmentation map', result)
            cv2.waitKey(0)

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
            engine.evaluate(model, data_loader, device=device)
            torch.save(model.state_dict(), modelpath)

if __name__ == "__main__":
    main()