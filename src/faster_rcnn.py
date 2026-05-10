import torch
import torch.nn as nn
import torchvision
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


# ======================================================================
# utility functions (already implemented)
# ======================================================================

def get_iou(boxes1, boxes2):
    """compute iou between box sets (N x 4) and (M x 4)"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    # intersection coordinates
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # (N, M)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # (N, M)
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # (N, M)
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # (N, M)

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)  # (N, M)
    union = area1[:, None] + area2 - intersection_area  # (N, M)
    iou = intersection_area / union  # (N, M)
    return iou


def boxes_to_transformation_targets(ground_truth_boxes, anchors_or_proposals):
    """convert bbox coordinates to regression targets (tx, ty, tw, th)"""
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * widths
    center_y = anchors_or_proposals[:, 1] + 0.5 * heights

    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_center_x - center_x) / widths
    targets_dy = (gt_center_y - center_y) / heights
    targets_dw = torch.log(gt_widths / widths)
    targets_dh = torch.log(gt_heights / heights)

    regression_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return regression_targets


def apply_regression_pred_to_anchors_or_proposals(box_transform_pred, anchors_or_proposals):
    """apply predicted transformations to anchors/proposals to get predicted boxes"""
    box_transform_pred = box_transform_pred.reshape(box_transform_pred.size(0), -1, 4)

    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h

    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]

    # clamp to avoid exp overflow
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))

    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]

    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h

    pred_boxes = torch.stack((pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2), dim=2)
    return pred_boxes


def sample_positive_negative(labels, positive_count, total_count):
    """sample positive and negative examples for training"""
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]

    num_pos = min(positive.numel(), positive_count)
    num_neg = min(negative.numel(), total_count - num_pos)

    perm_positive_idxs = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm_negative_idxs = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idxs = positive[perm_positive_idxs]
    neg_idxs = negative[perm_negative_idxs]

    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True

    return sampled_neg_idx_mask, sampled_pos_idx_mask


def clamp_boxes_to_image_boundary(boxes, image_shape):
    """clip boxes to stay within image boundaries"""
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]

    height, width = image_shape[-2:]
    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)

    boxes = torch.cat((
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None]),
        dim=-1)
    return boxes


def transform_boxes_to_original_size(boxes, new_size, original_size):
    """scale bounding boxes back to original image dimensions"""
    ratios = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


# ======================================================================
# part 2: region proposal network (20%)
# ======================================================================

class RegionProposalNetwork(nn.Module):
    """region proposal network for faster r-cnn"""

    def __init__(self, in_channels, scales, aspect_ratios, model_config):
        super(RegionProposalNetwork, self).__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.low_iou_threshold = model_config['rpn_bg_threshold']
        self.high_iou_threshold = model_config['rpn_fg_threshold']
        self.rpn_nms_threshold = model_config['rpn_nms_threshold']
        self.rpn_batch_size = model_config['rpn_batch_size']
        self.rpn_pos_count = int(model_config['rpn_pos_fraction'] * self.rpn_batch_size)
        self.rpn_topk = model_config['rpn_train_topk'] if self.training else model_config['rpn_test_topk']
        self.rpn_prenms_topk = model_config['rpn_train_prenms_topk'] if self.training else model_config['rpn_test_prenms_topk']

        # one anchor per (scale, aspect_ratio) combination per location
        self.num_anchors = len(scales) * len(aspect_ratios)

        # 3x3 conv shared layer
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # objectness score: one value per anchor
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1)
        # box regression: 4 values per anchor
        self.reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1)

        # init weights - standard rpn initialization
        for layer in [self.rpn_conv, self.cls_layer, self.reg_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def generate_anchors(self, image, feat):
        """generate anchors for all feature map locations"""
        # figure out stride from image to feature map
        feat_h, feat_w = feat.shape[-2:]
        img_h, img_w = image.shape[-2:]
        stride_h = img_h // feat_h
        stride_w = img_w // feat_w

        # build base anchors centered at (0,0) for each scale/ratio combo
        base_anchors = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                # width and height adjusted so area = scale^2
                w = scale * math.sqrt(1.0 / ratio)
                h = scale * math.sqrt(ratio)
                base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])

        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=feat.device)  # (num_anchors, 4)

        # shifts for every grid cell center
        shifts_x = torch.arange(0, feat_w, device=feat.device) * stride_w + stride_w // 2
        shifts_y = torch.arange(0, feat_h, device=feat.device) * stride_h + stride_h // 2
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).float()  # (H*W, 4)

        # combine shifts with base anchors: (H*W, num_anchors, 4)
        anchors = shifts[:, None, :] + base_anchors[None, :, :]
        anchors = anchors.reshape(-1, 4)  # (H*W*num_anchors, 4)
        return anchors

    def assign_targets_to_anchors(self, anchors, gt_boxes):
        """assign ground truth labels and boxes to anchors based on iou"""
        # gt_boxes shape: (1, num_gt, 4) - squeeze batch dim
        gt_boxes = gt_boxes.squeeze(0)

        iou_matrix = get_iou(anchors, gt_boxes)  # (num_anchors, num_gt)

        # best gt for each anchor
        best_gt_iou_per_anchor, best_gt_idx_per_anchor = iou_matrix.max(dim=1)

        # start with all ignored (-1)
        labels = torch.full((anchors.shape[0],), -1, dtype=torch.float32, device=anchors.device)

        # negatives below low threshold
        labels[best_gt_iou_per_anchor < self.low_iou_threshold] = 0

        # positives above high threshold
        labels[best_gt_iou_per_anchor >= self.high_iou_threshold] = 1

        # also mark the best anchor for each gt box as positive (guarantees coverage)
        best_anchor_iou_per_gt, best_anchor_idx_per_gt = iou_matrix.max(dim=0)
        labels[best_anchor_idx_per_gt] = 1

        # get matched gt boxes for each anchor
        matched_gt_boxes = gt_boxes[best_gt_idx_per_anchor]

        return labels, matched_gt_boxes

    def filter_proposals(self, proposals, cls_scores, image_shape):
        """filter proposals using score threshold and nms"""
        # proposals: (num_anchors, num_anchors_per_loc, 4) -> flatten
        proposals = proposals.reshape(-1, 4)
        cls_scores = cls_scores.reshape(-1)

        # convert raw logits to probabilities
        cls_scores = torch.sigmoid(cls_scores)

        # take top-k before nms to speed things up
        prenms_topk = min(self.rpn_prenms_topk, cls_scores.shape[0])
        top_scores, top_idx = cls_scores.topk(prenms_topk)
        proposals = proposals[top_idx]
        cls_scores = top_scores

        # clip to image bounds
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)

        # remove degenerate boxes (width or height < 1 pixel)
        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        valid = (widths >= 1) & (heights >= 1)
        proposals = proposals[valid]
        cls_scores = cls_scores[valid]

        # nms
        keep = torchvision.ops.nms(proposals, cls_scores, self.rpn_nms_threshold)

        # topk after nms
        topk = min(self.rpn_topk, keep.shape[0])
        keep = keep[:topk]
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]

        return proposals, cls_scores

    def forward(self, image, feat, target=None):
        """forward pass for rpn"""
        # shared conv
        t = torch.relu(self.rpn_conv(feat))

        # classification and regression heads
        cls_scores = self.cls_layer(t)   # (B, num_anchors, H, W)
        box_transform_pred = self.reg_layer(t)  # (B, num_anchors*4, H, W)

        # generate all anchors for this feature map
        anchors = self.generate_anchors(image, feat)  # (N, 4)

        # reshape predictions to match anchors
        # cls: (B, num_anchors, H, W) -> (B, H*W*num_anchors)
        B, _, H, W = cls_scores.shape
        cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(B, -1)
        # reg: (B, num_anchors*4, H, W) -> (B, H*W*num_anchors, 4)
        box_transform_pred = box_transform_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)

        # decode predicted boxes from anchors
        # apply_regression returns (num_anchors, num_per_loc, 4) shape
        proposals = apply_regression_pred_to_anchors_or_proposals(
            box_transform_pred[0], anchors
        )  # (num_anchors, 1, 4)

        # filter proposals
        proposals, scores = self.filter_proposals(proposals, cls_scores[0].detach(), image.shape)

        rpn_output = {
            'proposals': proposals,
            'scores': scores,
        }

        if self.training and target is not None:
            # assign gt to anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, target['bboxes'])

            # sample balanced pos/neg set
            sampled_neg_mask, sampled_pos_mask = sample_positive_negative(
                labels, self.rpn_pos_count, self.rpn_batch_size
            )
            sampled_mask = sampled_pos_mask | sampled_neg_mask

            # regression targets for positive anchors
            reg_targets = boxes_to_transformation_targets(matched_gt_boxes, anchors)

            # cls loss over sampled anchors
            # labels are float (0 or 1), negatives have label 0
            sampled_cls_scores = cls_scores[0][sampled_mask]
            sampled_labels = labels[sampled_mask]
            rpn_cls_loss = nn.functional.binary_cross_entropy_with_logits(
                sampled_cls_scores, sampled_labels
            )

            # reg loss only over positive anchors (smooth l1)
            pos_box_preds = box_transform_pred[0][sampled_pos_mask]
            pos_reg_targets = reg_targets[sampled_pos_mask]

            if pos_box_preds.shape[0] > 0:
                rpn_reg_loss = nn.functional.smooth_l1_loss(pos_box_preds, pos_reg_targets)
            else:
                rpn_reg_loss = torch.tensor(0.0, device=feat.device)

            rpn_output['rpn_classification_loss'] = rpn_cls_loss
            rpn_output['rpn_localization_loss'] = rpn_reg_loss

        return rpn_output


# ======================================================================
# part 3: roi feature extraction and part 4: detection head (40%)
# ======================================================================

class ROIHead(nn.Module):
    """roi head for final classification and box refinement"""

    def __init__(self, model_config, num_classes, in_channels):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.roi_batch_size = model_config['roi_batch_size']
        self.roi_pos_count = int(model_config['roi_pos_fraction'] * self.roi_batch_size)
        self.iou_threshold = model_config['roi_iou_threshold']
        self.low_bg_iou = model_config['roi_low_bg_iou']
        self.nms_threshold = model_config['roi_nms_threshold']
        self.topK_detections = model_config['roi_topk_detections']
        self.low_score_threshold = model_config['roi_score_threshold']
        self.pool_size = model_config['roi_pool_size']
        self.fc_inner_dim = model_config['fc_inner_dim']

        # two fc layers after roi pooling
        self.fc1 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc2 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)

        # classification head: num_classes scores (including background)
        self.cls_layer = nn.Linear(self.fc_inner_dim, num_classes)
        # regression head: 4 offsets per class
        self.reg_layer = nn.Linear(self.fc_inner_dim, num_classes * 4)

        # weights: slightly different stds following the paper
        nn.init.normal_(self.cls_layer.weight, std=0.01)
        nn.init.constant_(self.cls_layer.bias, 0)
        nn.init.normal_(self.reg_layer.weight, std=0.001)
        nn.init.constant_(self.reg_layer.bias, 0)

    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        """assign gt boxes and labels to proposals based on iou"""
        # gt_boxes: (1, num_gt, 4), gt_labels: (1, num_gt)
        gt_boxes = gt_boxes.squeeze(0)
        gt_labels = gt_labels.squeeze(0)

        iou_matrix = get_iou(proposals, gt_boxes)  # (num_proposals, num_gt)
        best_gt_iou, best_gt_idx = iou_matrix.max(dim=1)

        # matched gt box for each proposal
        matched_gt_boxes = gt_boxes[best_gt_idx]
        labels = gt_labels[best_gt_idx]

        # background: iou in [low_bg_iou, iou_threshold)
        bg_mask = (best_gt_iou < self.iou_threshold) & (best_gt_iou >= self.low_bg_iou)
        labels[bg_mask] = 0

        # ignore: below low_bg_iou entirely
        ignore_mask = best_gt_iou < self.low_bg_iou
        labels[ignore_mask] = -1

        return labels, matched_gt_boxes

    def forward(self, feat, proposals, image_shape, target):
        """forward pass for roi head"""
        if self.training and target is not None:
            # add gt boxes to proposals so they definitely get trained on
            gt_boxes = target['bboxes'].squeeze(0)
            proposals = torch.cat([proposals, gt_boxes], dim=0)

            # assign gt to each proposal
            labels, matched_gt_boxes = self.assign_target_to_proposals(
                proposals, target['bboxes'], target['labels']
            )

            # sample balanced pos/neg proposals
            sampled_neg_mask, sampled_pos_mask = sample_positive_negative(
                labels, self.roi_pos_count, self.roi_batch_size
            )
            sampled_mask = sampled_pos_mask | sampled_neg_mask
            proposals = proposals[sampled_mask]
            labels = labels[sampled_mask]
            matched_gt_boxes = matched_gt_boxes[sampled_mask]

        # scale from image coords to feature map coords
        img_h, img_w = image_shape
        feat_h, feat_w = feat.shape[-2:]
        scale = min(feat_h / img_h, feat_w / img_w)

        # roi pooling to get fixed-size features (pool_size x pool_size)
        pooled_feat = torchvision.ops.roi_pool(
            feat,
            [proposals],  # list of proposal tensors, one per image
            output_size=self.pool_size,
            spatial_scale=scale,
        )

        # flatten and pass through fc layers
        pooled_feat = pooled_feat.flatten(start_dim=1)
        pooled_feat = torch.relu(self.fc1(pooled_feat))
        pooled_feat = torch.relu(self.fc2(pooled_feat))

        cls_scores = self.cls_layer(pooled_feat)    # (num_rois, num_classes)
        box_transform_pred = self.reg_layer(pooled_feat)  # (num_rois, num_classes*4)

        if self.training and target is not None:
            # regression targets for matched gt boxes
            reg_targets = boxes_to_transformation_targets(matched_gt_boxes, proposals)

            # classification loss over all sampled proposals
            frcnn_cls_loss = nn.functional.cross_entropy(cls_scores, labels.long())

            # regression loss only for foreground proposals
            fg_mask = labels > 0
            if fg_mask.sum() > 0:
                # pick the predicted offsets for the correct class
                fg_labels = labels[fg_mask].long()
                fg_box_preds = box_transform_pred[fg_mask]
                # select the 4 values corresponding to the gt class
                fg_box_preds = fg_box_preds.reshape(-1, self.num_classes, 4)
                fg_box_preds = fg_box_preds[torch.arange(fg_mask.sum()), fg_labels]
                fg_reg_targets = reg_targets[fg_mask]
                frcnn_reg_loss = nn.functional.smooth_l1_loss(fg_box_preds, fg_reg_targets)
            else:
                frcnn_reg_loss = torch.tensor(0.0, device=feat.device)

            return {
                'frcnn_classification_loss': frcnn_cls_loss,
                'frcnn_localization_loss': frcnn_reg_loss,
            }

        else:
            # inference: decode boxes and filter
            # pick class with highest score (excluding background=0)
            cls_probs = torch.softmax(cls_scores, dim=1)
            pred_scores, pred_labels = cls_probs[:, 1:].max(dim=1)
            pred_labels = pred_labels + 1  # shift back since we excluded background

            # decode boxes using the predicted class offsets
            box_transform_pred = box_transform_pred.reshape(-1, self.num_classes, 4)
            pred_boxes = apply_regression_pred_to_anchors_or_proposals(
                box_transform_pred[torch.arange(proposals.shape[0]), pred_labels],
                proposals
            )  # (num_rois, 1, 4)
            pred_boxes = pred_boxes.squeeze(1)

            # clip to image
            pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)

            pred_boxes, pred_labels, pred_scores = self.filter_predictions(
                pred_boxes, pred_labels, pred_scores
            )

            return {
                'boxes': pred_boxes,
                'labels': pred_labels,
                'scores': pred_scores,
            }

    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        """filter predictions by score, size, and nms"""
        # remove low confidence detections
        keep = pred_scores >= self.low_score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # remove tiny boxes
        widths = pred_boxes[:, 2] - pred_boxes[:, 0]
        heights = pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (widths >= 1) & (heights >= 1)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # per-class nms
        keep_indices = torchvision.ops.batched_nms(pred_boxes, pred_scores, pred_labels, self.nms_threshold)

        # topk overall
        topk = min(self.topK_detections, keep_indices.shape[0])
        keep_indices = keep_indices[:topk]

        return pred_boxes[keep_indices], pred_labels[keep_indices], pred_scores[keep_indices]


# ======================================================================
# part 5: faster r-cnn model (20%)
# ======================================================================

class FasterRCNN(nn.Module):
    """faster r-cnn object detection model"""

    def __init__(self, model_config, num_classes):
        super(FasterRCNN, self).__init__()
        self.model_config = model_config

        # vgg16 backbone - use features up to (but not including) the last max pool
        vgg16 = torchvision.models.vgg16(weights="DEFAULT")
        self.backbone = vgg16.features[:-1]

        # rpn and roi head
        self.rpn = RegionProposalNetwork(
            in_channels=model_config['backbone_out_channels'],
            scales=model_config['scales'],
            aspect_ratios=model_config['aspect_ratios'],
            model_config=model_config,
        )
        self.roi_head = ROIHead(
            model_config=model_config,
            num_classes=num_classes,
            in_channels=model_config['backbone_out_channels'],
        )

        # freeze the first two conv blocks (layers 0-9) per standard practice
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']

    def normalize_resize_image_and_boxes(self, image, bboxes=None):
        """normalize and resize image, adjusting bboxes accordingly"""
        if image.dim() == 3:
            image = image.unsqueeze(0)

        c, h, w = image.shape[-3:]

        image = image.float()
        mean = torch.as_tensor(self.image_mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor(self.image_std, dtype=image.dtype, device=image.device)
        image = (image - mean[:, None, None]) / std[:, None, None]

        min_original_size = float(min((h, w)))
        max_original_size = float(max((h, w)))
        scale_factor_min = self.min_size / min_original_size

        if max_original_size * scale_factor_min > self.max_size:
            scale_factor = self.max_size / max_original_size
        else:
            scale_factor = scale_factor_min

        image = torch.nn.functional.interpolate(
            image, scale_factor=scale_factor, mode='bilinear',
            recompute_scale_factor=True, align_corners=False
        )

        if bboxes is not None:
            if bboxes.dim() == 2:
                ratios = [
                    torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
                    torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                    for s, s_orig in zip(image.shape[-2:], (h, w))
                ]
                ratio_height, ratio_width = ratios
                xmin = bboxes[:, 0] * ratio_width
                ymin = bboxes[:, 1] * ratio_height
                xmax = bboxes[:, 2] * ratio_width
                ymax = bboxes[:, 3] * ratio_height
                bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)

            elif bboxes.dim() == 3:
                ratios = [
                    torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
                    torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                    for s, s_orig in zip(image.shape[-2:], (h, w))
                ]
                ratio_height, ratio_width = ratios
                xmin, ymin, xmax, ymax = bboxes.unbind(2)
                xmin = xmin * ratio_width
                xmax = xmax * ratio_width
                ymin = ymin * ratio_height
                ymax = ymax * ratio_height
                bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)

        return image, bboxes

    def forward(self, image, target=None):
        """forward pass for faster r-cnn"""
        # save original size before any resizing
        original_image_size = image.shape[-2:]

        # normalize and resize image (and gt boxes if training)
        if target is not None:
            image, target['bboxes'] = self.normalize_resize_image_and_boxes(image, target['bboxes'])
        else:
            image, _ = self.normalize_resize_image_and_boxes(image)

        resized_image_size = image.shape[-2:]

        # extract features through vgg backbone
        feat = self.backbone(image)

        # get region proposals from rpn
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']

        # run roi head
        frcnn_output = self.roi_head(feat, proposals, resized_image_size, target)

        # scale predicted boxes back to original image coordinates during inference
        if not self.training:
            frcnn_output['boxes'] = transform_boxes_to_original_size(
                frcnn_output['boxes'],
                resized_image_size,
                original_image_size,
            )

        return rpn_output, frcnn_output
