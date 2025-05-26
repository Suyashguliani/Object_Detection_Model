import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
import xml.etree.ElementTree as ET
import os

class VOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='train', transform=None, S=14, B=2, C=20):
        super(VOCDataset, self).__init__()
        self.voc = VOCDetection(root, year=year, image_set=image_set, download=True)
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.class_to_idx = {cls: idx for idx, cls in enumerate([
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ])}

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]
        original_img_width, original_img_height = img.size

        if self.transform:
            img = self.transform(img)

        label_tensor = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])

            img_h, img_w = 448, 448 # Explicitly use the resized dimensions

            center_x_abs = (xmin + xmax) / 2.0
            center_y_abs = (ymin + ymax) / 2.0
            box_width_abs = xmax - xmin
            box_height_abs = ymax - ymin

            box_x_rel = center_x_abs / img_w
            box_y_rel = center_y_abs / img_h
            box_w_rel = box_width_abs / img_w
            box_h_rel = box_height_abs / img_h

            class_idx = self.class_to_idx[obj['name']]

            i = int(box_y_rel * self.S)
            j = int(box_x_rel * self.S)

            if i >= self.S: i = self.S - 1
            if j >= self.S: j = self.S - 1

            label_tensor[i, j, self.C] = 1.0

            label_tensor[i, j, self.C + 1:self.C + 5] = torch.tensor([
                box_x_rel * self.S - j,
                box_y_rel * self.S - i,
                box_w_rel,
                box_h_rel
            ])

            label_tensor[i, j, class_idx] = 1.0

        return img, label_tensor

def get_transform():
    return T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
    ])