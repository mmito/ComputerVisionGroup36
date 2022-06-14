from collections import defaultdict
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    """PyTorch dataset for COCO annotations."""

    def __init__(self, data_dir, transforms=None):
        """Load COCO annotation data."""
        self.data_dir = Path(data_dir)
        self.transforms = transforms

        # load the COCO annotations json
        anno_file_path = self.data_dir/'jish4knowledge_2K_annotations.json'
        with open(str(anno_file_path)) as file_obj:
            self.coco_data = json.load(file_obj)
        # put all of the annos into a dict where keys are image IDs to speed up retrieval
        self.image_id_to_annos = defaultdict(list)
        for anno in self.coco_data['annotations']:
            image_id = anno['image_id']
            self.image_id_to_annos[image_id] += [anno]

    def __len__(self):
        return len(self.coco_data['images'])

    def __getitem__(self, index):
        """Return tuple of image and labels as torch tensors."""
        image_data = self.coco_data['images'][index]
        image_id = image_data['id']
        image_path = self.data_dir/image_data['file_name']
        image = Image.open(image_path)

        annos = self.image_id_to_annos[image_id]
        anno_data = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': [],
        }
        for anno in annos:
            coco_bbox = anno['bbox']
            left = coco_bbox[0]
            top = coco_bbox[1]
            right = coco_bbox[0] + coco_bbox[2]
            bottom = coco_bbox[1] + coco_bbox[3]
            area = coco_bbox[2] * coco_bbox[3]
            anno_data['boxes'].append([left, top, right, bottom])
            anno_data['labels'].append(anno['category_id'])
            anno_data['area'].append(area)
            anno_data['iscrowd'].append(anno['iscrowd'])

        target = {
            'boxes': torch.as_tensor(anno_data['boxes'], dtype=torch.float32),
            'labels': torch.as_tensor(anno_data['labels'], dtype=torch.int64),
            'image_id': torch.tensor([image_id]),  # pylint: disable=not-callable (false alarm)
            'area': torch.as_tensor(anno_data['area'], dtype=torch.float32),
            'iscrowd': torch.as_tensor(anno_data['iscrowd'], dtype=torch.int64),
        }

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target