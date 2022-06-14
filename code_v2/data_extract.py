import csv
import os
import torch
import numpy as np
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw


def extract_data(img, img_info, annot_idx, root):
    #if annot_idx[]
    print('Annot idx: ', annot_idx)
    print(img_info)
    img = Image.open(os.path.join(root, img)).convert("RGB")
    width, height = img.size
    for i in range(len(annots['segmentation'])):
        box = [float(annots['bbox'][0]),float(annots['bbox'][1]), float(annots['bbox'][0]) + float(annots['bbox'][2]), float(annots['bbox'][1]) + float(annots['bbox'][3])]
        mask_img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(mask_img).polygon(annots['segmentation'], outline = 128, fill=128)
        full_mask = np.array((np.array(mask_img) > 0)).astype(np.uint8).tolist()
        if "image_id" not in target:
            target["image_id"] = [int(row[5])]
            target["labels"] = [int(row[10])]
            target["masks"] = [full_mask]
            target["boxes"] = [box]
            target["area"] = [area]
            target["iscrowd"] = [False]
        else:
            target["labels"].append(int(row[10]))
            target["masks"].append(full_mask)
            target["boxes"].append(box)
            target["area"].append(area)
            target["iscrowd"].append(False)
    if "image_id" in target:
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        target["masks"] = torch.as_tensor(np.array(target["masks"]), dtype=torch.uint8)
        target["image_id"] = torch.as_tensor(target["image_id"], dtype=torch.int64)
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["area"] = torch.as_tensor(target["area"], dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.float32)
    return img, target

# import csv

# import os

# import torch

# import numpy as np

# from PIL import Image

# import numpy as np

# from PIL import Image, ImageDraw





# def extract_data(img, root):

#     mask_path = os.path.join(root, "annotations")

#     img_path = os.path.join(root, "images")

#     mask_file = "data_" + img[0:30] + ".csv"

#     img_id_pre = img.split('.')

#     img_id = img_id_pre[0][30:]

#     img = Image.open(os.path.join(img_path, img)).convert("RGB")

#     width, height = img.size



#     with open(os.path.join(mask_path, mask_file), newline='') as csvfile:

#         reader = csv.reader(csvfile, delimiter=',')

#         info_strs = next(iter(reader))

#         target = {}

#         for row in reader:

#             if int(row[5]) == int(img_id):

#                 area = float(row[8])*float(row[9])

#                 box = [float(row[6]),float(row[7]), float(row[6]) + float(row[8]), float(row[7]) + float(row[9])]



#                 poly_mask_str = row[14][3:-2].split(';')

#                 poly_mask = []

#                 for i in range(1, len(poly_mask_str)):

#                     try:

#                         poly_mask.append((float(poly_mask_str[i].split(' ')[0]), float(poly_mask_str[i].split(' ')[1])))

#                     except ValueError:

#                         continue

#                 mask_img = Image.new('L', (width, height), 0)

#                 ImageDraw.Draw(mask_img).polygon(poly_mask, outline = 128, fill=128)

#                 full_mask = np.array((np.array(mask_img) > 0)).astype(np.uint8).tolist()

#                 if "image_id" not in target:

#                     target["image_id"] = [int(row[5])]

#                     target["labels"] = [int(row[10])]

#                     target["masks"] = [full_mask]

#                     target["boxes"] = [box]

#                     target["area"] = [area]

#                     target["iscrowd"] = [False]

#                 else:

#                     target["labels"].append(int(row[10]))

#                     target["masks"].append(full_mask)

#                     target["boxes"].append(box)

#                     target["area"].append(area)

#                     target["iscrowd"].append(False)

#         if "image_id" in target:

#             target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)

#             target["masks"] = torch.as_tensor(np.array(target["masks"]), dtype=torch.uint8)

#             target["image_id"] = torch.as_tensor(target["image_id"], dtype=torch.int64)

#             target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)

#             target["area"] = torch.as_tensor(target["area"], dtype=torch.float32)

#             target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.float32)

#         return img, target


