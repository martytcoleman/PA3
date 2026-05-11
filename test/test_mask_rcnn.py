import torch
import numpy as np
import cv2
import argparse
import random
import os
import yaml
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mask_rcnn import MaskRCNN
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)


def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    return area_intersection / area_union


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    aps = []
    for idx, label in enumerate(gt_labels):
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                gt_matched[im_idx][max_iou_gt_idx] = True
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)
        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            i = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                prec_interp_pt = precisions[recalls >= interp_pt]
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps


def load_model_and_dataset(args):
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset('test', im_dir=dataset_config['im_test_path'], ann_dir=dataset_config['ann_test_path'])
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False)

    model = MaskRCNN(model_config, num_classes=dataset_config['num_classes'])
    model.eval()
    model.to(device)

    # allow overriding checkpoint path via --ckpt argument
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print('loaded checkpoint from', ckpt_path)
    return model, voc, test_dataset


def overlay_mask_on_image(im, mask_28, box, color):
    """paste a 28x28 binary mask back onto the image at the detected box location."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, im.shape[1] - 1), min(y2, im.shape[0] - 1)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return
    mask_full = cv2.resize(mask_28.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST)
    roi = im[y1:y2, x1:x2]
    colored = np.zeros_like(roi)
    colored[:] = color
    roi[mask_full == 1] = (roi[mask_full == 1] * 0.5 + colored[mask_full == 1] * 0.5).astype(np.uint8)


def infer(args):
    if not os.path.exists('samples'):
        os.mkdir('samples')
    model, voc, test_dataset = load_model_and_dataset(args)
    model.roi_head.low_score_threshold = 0.7

    # distinct colors per class for mask overlays
    rng = np.random.RandomState(42)
    class_colors = {cls: rng.randint(50, 220, 3).tolist() for cls in voc.label2idx}

    for sample_count in tqdm(range(10)):
        random_idx = random.randint(0, len(voc))
        im, target, fname = voc[random_idx]
        im_tensor = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = [int(v) for v in box.detach().cpu().numpy()]
            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = voc.idx2label[target['labels'][idx].item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_size[0], y1 + 10 + text_size[1]), [255, 255, 255], -1)
            cv2.putText(gt_im, text=text, org=(x1 + 5, y1 + 15), thickness=1,
                        fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text, org=(x1 + 5, y1 + 15), thickness=1,
                        fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('samples/output_mask_rcnn_gt_{}.png'.format(sample_count), gt_im)

        rpn_output, frcnn_output = model(im_tensor, None)
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        masks = frcnn_output.get('masks', None)

        im_pred = cv2.imread(fname)
        im_pred_copy = im_pred.copy()

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box.detach().cpu().numpy()]
            label_name = voc.idx2label[labels[idx].item()]
            score = scores[idx].item()
            color = class_colors[label_name]

            # overlay mask if available
            if masks is not None:
                mask_np = masks[idx].detach().cpu().numpy().astype(np.uint8)
                overlay_mask_on_image(im_pred, mask_np, [x1, y1, x2, y2], color)
                overlay_mask_on_image(im_pred_copy, mask_np, [x1, y1, x2, y2], color)

            cv2.rectangle(im_pred, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_pred_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = '{} : {:.2f}'.format(label_name, score)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            cv2.rectangle(im_pred_copy, (x1, y1), (x1 + 10 + text_size[0], y1 + 10 + text_size[1]), [255, 255, 255], -1)
            cv2.putText(im_pred, text=text, org=(x1 + 5, y1 + 15), thickness=1,
                        fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_pred_copy, text=text, org=(x1 + 5, y1 + 15), thickness=1,
                        fontScale=1, color=[0, 0, 0], fontFace=cv2.FONT_HERSHEY_PLAIN)

        cv2.addWeighted(im_pred_copy, 0.7, im_pred, 0.3, 0, im_pred)
        cv2.imwrite('samples/output_mask_rcnn_{}.jpg'.format(sample_count), im_pred)


def evaluate_map(args):
    model, voc, test_dataset = load_model_and_dataset(args)
    gts = []
    preds = []
    for im, target, fname in tqdm(test_dataset):
        im = im.float().to(device)
        target_boxes = target['bboxes'].float().to(device)[0]
        target_labels = target['labels'].long().to(device)[0]
        rpn_output, frcnn_output = model(im, None)

        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']

        pred_boxes = {label_name: [] for label_name in voc.label2idx}
        gt_boxes = {label_name: [] for label_name in voc.label2idx}

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label_name = voc.idx2label[labels[idx].item()]
            score = scores[idx].item()
            pred_boxes[label_name].append([x1, y1, x2, y2, score])
        for idx, box in enumerate(target_boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label_name = voc.idx2label[target_labels[idx].item()]
            gt_boxes[label_name].append([x1, y1, x2, y2])

        gts.append(gt_boxes)
        preds.append(pred_boxes)

    mean_ap, all_aps = compute_map(preds, gts, method='interp')
    print('Class Wise Average Precisions')
    for idx in range(len(voc.idx2label)):
        print('AP for class {} = {:.4f}'.format(voc.idx2label[idx], all_aps[voc.idx2label[idx]]))
    print('Mean Average Precision : {:.4f}'.format(mean_ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for mask rcnn inference')
    parser.add_argument('--config', dest='config_path', default='config/voc_mask.yaml', type=str)
    parser.add_argument('--ckpt', dest='ckpt_path', default=None, type=str,
                        help='override checkpoint path from config')
    parser.add_argument('--evaluate', dest='evaluate', default=False, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples', default=True, type=bool)
    args = parser.parse_args()
    if args.infer_samples:
        infer(args)
    else:
        print('Not inferring samples')
    if args.evaluate:
        evaluate_map(args)
    else:
        print('Not evaluating')
