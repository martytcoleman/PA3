"""
Evaluate Mask R-CNN mask mIoU on the VOC 2007 segmentation test split (210 images).
Each SegmentationObject PNG encodes instances as integer values 1..N (0=bg, 255=void).
Predicted 28x28 masks are placed back at the detected box and compared against GT.

Usage:
    python test/evaluate_mask_iou.py
    python test/evaluate_mask_iou.py --ckpt mask_rcnn_voc2007.pth
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mask_rcnn import MaskRCNN

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print('device:', device)

CLASSES = sorted([
    'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
    'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
    'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
])
CLASSES = ['background'] + CLASSES
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for i, c in enumerate(CLASSES)}


def box_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def mask_iou(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return inter / max(union, 1)


def predicted_mask_at_full_size(mask28, box, img_h, img_w):
    """Resize 28x28 predicted mask to box region and place on H x W canvas."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img_w - 1), min(y2, img_h - 1)
    bw, bh = x2 - x1, y2 - y1
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    if bw <= 0 or bh <= 0:
        return canvas
    from PIL import Image as PILImage
    m = PILImage.fromarray((mask28 * 255).astype(np.uint8))
    m = m.resize((bw, bh), PILImage.NEAREST)
    m_arr = (np.array(m) > 127).astype(np.uint8)
    canvas[y1:y1 + bh, x1:x1 + bw] = m_arr
    return canvas


def load_gt_instances(ann_path, seg_path):
    """
    Returns list of dicts with keys: label_idx, gt_mask (H x W bool array).
    Instance k in the XML corresponds to pixel value k+1 in the seg PNG.
    """
    tree = ET.parse(ann_path)
    root = tree.getroot()
    objs = root.findall('object')

    seg = np.array(Image.open(seg_path))

    instances = []
    for k, obj in enumerate(objs):
        name = obj.find('name').text
        if name not in LABEL2IDX:
            continue
        label_idx = LABEL2IDX[name]
        instance_val = k + 1
        gt_mask = (seg == instance_val)
        if gt_mask.sum() == 0:
            continue
        instances.append({'label': label_idx, 'gt_mask': gt_mask})
    return instances


def evaluate(args):
    seg_list = args.seg_list
    voc_dir = args.voc_dir
    ann_dir = os.path.join(voc_dir, 'Annotations')
    img_dir = os.path.join(voc_dir, 'JPEGImages')
    seg_dir = os.path.join(voc_dir, 'SegmentationObject')

    with open(seg_list) as f:
        image_ids = [l.strip() for l in f if l.strip()]

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg['model_params']
    num_classes = cfg['dataset_params']['num_classes']

    model = MaskRCNN(model_cfg, num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval().to(device)
    model.roi_head.low_score_threshold = 0.5

    all_ious = []
    matched = 0
    unmatched = 0

    for img_id in tqdm(image_ids):
        ann_path = os.path.join(ann_dir, img_id + '.xml')
        img_path = os.path.join(img_dir, img_id + '.jpg')
        seg_path = os.path.join(seg_dir, img_id + '.png')

        if not os.path.exists(seg_path):
            continue

        gt_instances = load_gt_instances(ann_path, seg_path)
        if not gt_instances:
            continue

        img = Image.open(img_path).convert('RGB')
        img_h, img_w = np.array(img).shape[:2]
        im_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0).float().to(device)

        with torch.no_grad():
            _, frcnn_out = model(im_tensor, None)

        pred_boxes = frcnn_out['boxes'].cpu().numpy()
        pred_labels = frcnn_out['labels'].cpu().numpy()
        pred_masks = frcnn_out.get('masks', None)
        if pred_masks is not None:
            pred_masks = pred_masks.cpu().numpy()

        used_pred = [False] * len(pred_boxes)

        for gt in gt_instances:
            best_iou = -1
            best_j = -1
            for j, (pb, pl) in enumerate(zip(pred_boxes, pred_labels)):
                if used_pred[j]:
                    continue
                if pl != gt['label']:
                    continue
                gt_box = np.where(gt['gt_mask'])
                if len(gt_box[0]) == 0:
                    continue
                gt_bbox = [gt_box[1].min(), gt_box[0].min(),
                           gt_box[1].max(), gt_box[0].max()]
                iou = box_iou(pb, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0 and best_iou >= 0.5 and pred_masks is not None:
                m28 = pred_masks[best_j]
                full_mask = predicted_mask_at_full_size(m28, pred_boxes[best_j], img_h, img_w)
                miou = mask_iou(full_mask, gt['gt_mask'])
                all_ious.append(miou)
                used_pred[best_j] = True
                matched += 1
            else:
                unmatched += 1

    print(f'\nMatched GT instances : {matched}')
    print(f'Unmatched GT instances: {unmatched}')
    if all_ious:
        print(f'Mask mIoU            : {np.mean(all_ious):.4f}  ({np.mean(all_ious)*100:.2f}%)')
    else:
        print('No matched instances found.')


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(base, 'config/voc_mask.yaml'))
    parser.add_argument('--ckpt', default=os.path.join(base, 'mask_rcnn_voc2007.pth'))
    parser.add_argument('--voc_dir', default=os.path.join(base, 'VOCdevkit/VOC2007'))
    parser.add_argument('--seg_list', default=os.path.join(
        base, 'VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt'))
    args = parser.parse_args()
    evaluate(args)
