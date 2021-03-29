from pathlib import Path
from os.path import splitext
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

import os
import cv2
import pandas as pd
import random

box20List = ["IMG_170406_035932_0022_RGB4.JPG", "IMG_170406_035933_0023_RGB3.JPG", "IMG_170406_035939_0028_RGB3.JPG",
             "IMG_170406_040009_0053_RGB1.JPG", "IMG_170406_040033_0073_RGB2.JPG", "IMG_170406_040105_0099_RGB4.JPG",
             "IMG_170406_040108_0102_RGB3.JPG", "IMG_170406_040108_0102_RGB4.JPG", "IMG_170406_040156_0142_RGB1.JPG",
             "IMG_170406_040202_0147_RGB3.JPG", "IMG_170406_040202_0147_RGB4.JPG", "IMG_170406_040308_0202_RGB1.JPG"]

def get_dicts(root_dir):
    dataset_dicts = []
    root = Path(root_dir)
    imgs = sorted(root.glob("*.JPG"))
    
    for idx, img in enumerate(imgs):
        
        fname = str(img).replace(f'{root_dir}/', '')
        height, width = cv2.imread(os.path.join(root_dir, fname)).shape[:2]
        record = dict()
        record['file_name'] = os.path.join(root_dir, fname)
        record['image_id'] = idx
        record['height']= height
        record['width']= width

        objs = list()

        result = pd.read_csv(os.path.join('../rice_data/train_label', fname.replace('JPG', 'csv')), names=['x', 'y'])

        for i in range(0, result.shape[0]):
            if fname in box20List:
                xMin, yMin, xMax, yMax = result.x[i]-10, result.y[i]-10, result.x[i]+10, result.y[i]+10
            else:
                xMin, yMin, xMax, yMax = result.x[i]-20, result.y[i]-20, result.x[i]+20, result.y[i]+20

            XYXY = [xMin, yMin, xMax, yMax] 
            boxes = list(map(float,[XYXY[0], XYXY[1], XYXY[2], XYXY[3]]))

            obj = {
                "bbox": boxes,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)

        record['annotations'] = objs
        dataset_dicts.append(record)

    return dataset_dicts



if __name__ == "__main__":
    
    
    folder_name = '../rice_data'
    for dir in ["train"]:        
        DatasetCatalog.register(dir, lambda d = dir: get_dicts(folder_name + "/" + d))
        MetadataCatalog.get(dir).set(thing_classes=['plant'], stuff_classes=[], thing_colors=[(0,0,0)])
    
    tttt = MetadataCatalog.get("train")
    dataset_dicts = get_dicts(folder_name + "/train")
    
    # print(dataset_dicts)

    # for d in random.sample(dataset_dicts, 1):
    #     print(d['file_name'])
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=tttt, scale=0.5)
    #     out = visualizer.draw_dataset_dict(d)
    #     # imshow(out.get_image()[:, :, ::-1])
    #     cv2.imshow('image',out.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
