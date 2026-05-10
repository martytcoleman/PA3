import glob
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET

from dataset.voc import load_images_and_anns


class VOCMaskDataset(Dataset):
    """voc dataset extended with instance segmentation masks for mask r-cnn training.
    only images in the VOC segmentation split have mask annotations.
    images without masks still contribute to detection training."""

    def __init__(self, split, im_dir, ann_dir, seg_dir):
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        self.seg_dir = seg_dir  # path to VOCdevkit/VOC2007/SegmentationObject

        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, ann_dir, self.label2idx)

    def __len__(self):
        return len(self.images_info)

    def _load_seg_masks(self, img_id, gt_boxes):
        """load binary per-object masks from the VOC segmentation PNG.
        returns (num_objects, H, W) float tensor or None if no seg file exists."""
        seg_path = os.path.join(self.seg_dir, f'{img_id}.png')
        if not os.path.exists(seg_path):
            return None

        seg = np.array(Image.open(seg_path))  # uint8 palette PNG
        num_objects = len(gt_boxes)
        H, W = seg.shape
        masks = np.zeros((num_objects, H, W), dtype=np.float32)

        # instance index i in the xml -> pixel value i+1 in the seg mask
        # use overlap-based matching as a robust fallback
        for obj_idx in range(num_objects):
            # first try the direct index mapping (works for most VOC images)
            instance_val = obj_idx + 1
            direct_mask = (seg == instance_val).astype(np.float32)

            if direct_mask.sum() > 0:
                masks[obj_idx] = direct_mask
            else:
                # fallback: find the instance that overlaps most with the gt box
                x1, y1, x2, y2 = gt_boxes[obj_idx]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
                crop = seg[y1:y2+1, x1:x2+1]
                valid = crop[(crop > 0) & (crop < 255)]
                if len(valid) > 0:
                    best_val = np.bincount(valid.astype(np.int32)).argmax()
                    if best_val > 0:
                        masks[obj_idx] = (seg == best_val).astype(np.float32)
                # if nothing found, mask stays all zeros (will be ignored in loss)

        return torch.from_numpy(masks)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        img_id = im_info['img_id']
        im = Image.open(im_info['filename'])

        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        im_tensor = torchvision.transforms.ToTensor()(im)
        im_w = im_tensor.shape[-1]

        bboxes = torch.as_tensor([d['bbox'] for d in im_info['detections']])
        labels = torch.as_tensor([d['label'] for d in im_info['detections']])

        if to_flip:
            for idx, box in enumerate(bboxes):
                x1, y1, x2, y2 = box
                w = x2 - x1
                x1 = im_w - x1 - w
                x2 = x1 + w
                bboxes[idx] = torch.as_tensor([x1, y1, x2, y2])

        # load segmentation masks
        masks = self._load_seg_masks(img_id, bboxes.numpy())

        if to_flip and masks is not None:
            masks = torch.flip(masks, dims=[-1])  # flip along width

        targets = {
            'bboxes': bboxes,
            'labels': labels,
            'masks': masks,
        }
        return im_tensor, targets, im_info['filename']
