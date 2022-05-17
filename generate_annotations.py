import json
import re
import csv
import numpy as np
import os
from PIL import Image

DETECTION_ID = 0
VIDEO_ID = 4
FRAME_ID = 5
BB_X = 6
BB_Y = 7
BB_W = 8
BB_H = 9
SPECIE_ID = 10
SEGMENTATION = 14

curr_dir = os.path.dirname(__file__)
IMG_DIR = os.fsencode(os.path.join(curr_dir, 'images'))
ANN_DIR = os.fsencode(os.path.join(curr_dir, 'annotations'))


def generate_annotations():
    annotations_filenames = np.array(os.listdir(ANN_DIR), dtype=object)
    annotations = []
    for file in annotations_filenames:
        with open(os.fsdecode(file)) as fp:
            reader = csv.reader(fp, delimiter=",", quotechar='"')
            headers = next(reader, None)
            annotations = annotations + [annotate_row(row) for row in reader]

    return np.array(annotations, dtype=dict)


def generate_images():
    images = np.array(os.listdir(IMG_DIR), dtype=dict)
    for index, file in enumerate(images):
        img_id = os.fsdecode(file)
        width, height = Image.open(img_id).size
        images[index] = {
            'id': img_id,
            'width': width,
            'height': height,
            'file_name': img_id,
            'date_captured': os.path.getctime(img_id)
        }

    return images


def generate_categories(annotations):
    unique_categories = set()
    extract_unique_elements(annotations, unique_categories)

    # TODO: Add species names!
    return np.array([{'id': category, 'name': category} for category in unique_categories])


def extract_unique_elements(lists, result):
    if isinstance(lists, list):
        for item in lists:
            extract_unique_elements(item, result)
    elif isinstance(lists, dict):
        result.add(lists['category_id'])
    else:
        result.add(lists)


def annotate_row(row):
    index_id = row[DETECTION_ID]
    image_id = str(row[VIDEO_ID]) + str(row[FRAME_ID])
    category_id = row[SPECIE_ID]
    bbox = row[BB_X:BB_H + 1]
    segmentation = re.findall(r"\d*\w+", row[SEGMENTATION], re.M)

    return {
        'id': index_id,
        'image_id': image_id,
        'category_id': category_id,
        'bbox': bbox,
        'segmentation': [segmentation]
    }


if __name__ == '__main__':
    images_dict = generate_images()
    annotations_dict = generate_annotations()
    categories_dict = generate_categories(annotations_dict)
    complete_dict = {
        'images': images_dict,
        'annotations': annotations_dict,
        'categories': categories_dict
    }
    out_file = open('fish4knowledge_annotations.json', 'w+')
    json.dump(complete_dict, out_file)
    json.dumps(complete_dict)
