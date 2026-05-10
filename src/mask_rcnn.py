import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.faster_rcnn import (
    FasterRCNN, ROIHead,
    get_iou, boxes_to_transformation_targets,
    apply_regression_pred_to_anchors_or_proposals,
    sample_positive_negative, clamp_boxes_to_image_boundary,
    transform_boxes_to_original_size,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def project_masks_on_boxes(gt_masks, boxes, mask_size):
    """crop gt binary masks to their corresponding gt box and resize to mask_size x mask_size.
    uses roi_align so it's differentiable and handles sub-pixel alignment cleanly."""
    # gt_masks: (N, H, W) float tensors (0 or 1)
    # boxes: (N, 4) in image pixel coords
    padded = gt_masks.unsqueeze(1).float()  # (N, 1, H, W)

    # roi_align needs boxes as [batch_idx, x1, y1, x2, y2]
    batch_idx = torch.zeros(boxes.shape[0], 1, device=boxes.device)
    rois = torch.cat([batch_idx, boxes], dim=1)

    # for each mask crop to its own box - treat each mask as its own "batch"
    masks_out = []
    for i in range(padded.shape[0]):
        roi = rois[i:i+1]  # (1, 5)
        m = torchvision.ops.roi_align(
            padded[i:i+1],  # (1, 1, H, W)
            [roi[:, 1:]],   # just x1y1x2y2
            output_size=mask_size,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        masks_out.append(m[:, 0])  # (1, mask_size, mask_size)

    masks_out = torch.cat(masks_out, dim=0)  # (N, mask_size, mask_size)
    # threshold to binary — values from bilinear interp are float
    return (masks_out >= 0.5).float()


# ======================================================================
# mask head - small fcn that generates per-class binary masks
# ======================================================================

class MaskHead(nn.Module):
    """fcn mask branch: 4 convs -> deconv upsample -> per-class mask"""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), nn.ReLU()]
            in_channels = 256
        self.convs = nn.Sequential(*layers)
        # 2x upsample via deconv
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        # final per-class prediction (binary logits)
        self.mask_pred = nn.Conv2d(256, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convs(x)
        x = torch.relu(self.deconv(x))
        return self.mask_pred(x)  # (N, num_classes, 2*H, 2*W)


# ======================================================================
# mask roi head - extends detection roi head with mask branch
# ======================================================================

class MaskROIHead(ROIHead):
    """roi head with additional mask prediction branch (mask r-cnn)"""

    def __init__(self, model_config, num_classes, in_channels):
        super().__init__(model_config, num_classes, in_channels)
        # larger roi align output for the mask branch: 14x14 -> 28x28 after deconv
        self.mask_pool_size = 14
        self.mask_out_size = 28  # after deconv 2x upsample
        self.mask_head = MaskHead(in_channels, num_classes)

    def forward(self, feat, proposals, image_shape, target):
        """forward pass with mask prediction added to the detection head"""
        img_h, img_w = image_shape
        feat_h, feat_w = feat.shape[-2:]
        scale = min(feat_h / img_h, feat_w / img_w)

        if self.training and target is not None:
            gt_boxes = target['bboxes'].squeeze(0)
            gt_labels = target['labels'].squeeze(0)
            gt_masks = target.get('masks', None)  # (num_gt, H, W) or None

            # add gt boxes to proposals
            proposals = torch.cat([proposals, gt_boxes], dim=0)
            labels, matched_gt_boxes = self.assign_target_to_proposals(
                proposals, target['bboxes'], target['labels']
            )
            sampled_neg_mask, sampled_pos_mask = sample_positive_negative(
                labels, self.roi_pos_count, self.roi_batch_size
            )
            sampled_mask = sampled_pos_mask | sampled_neg_mask

            proposals = proposals[sampled_mask]
            labels = labels[sampled_mask]
            matched_gt_boxes = matched_gt_boxes[sampled_mask]

            # track which gt object each proposal matched (for mask lookup)
            # rebuild matched gt indices for the full (pre-sample) proposal set
            all_gt_boxes = target['bboxes'].squeeze(0)
            iou_full = get_iou(
                torch.cat([target['bboxes'].squeeze(0), all_gt_boxes], dim=0)
                if False else
                # just reuse the already-sampled proposals + gt_boxes approach
                proposals,
                all_gt_boxes
            )
            best_gt_idx = iou_full.max(dim=1)[1]  # (num_sampled,)

            # detection branch - roi pool at pool_size x pool_size
            pooled_det = torchvision.ops.roi_pool(
                feat, [proposals], output_size=self.pool_size, spatial_scale=scale
            )
            pooled_det = pooled_det.flatten(start_dim=1)
            pooled_det = torch.relu(self.fc1(pooled_det))
            pooled_det = torch.relu(self.fc2(pooled_det))

            cls_scores = self.cls_layer(pooled_det)
            box_transform_pred = self.reg_layer(pooled_det)

            reg_targets = boxes_to_transformation_targets(matched_gt_boxes, proposals)
            frcnn_cls_loss = F.cross_entropy(cls_scores, labels.long())

            fg_mask = labels > 0
            if fg_mask.sum() > 0:
                fg_labels = labels[fg_mask].long()
                fg_box_preds = box_transform_pred[fg_mask].reshape(-1, self.num_classes, 4)
                fg_box_preds = fg_box_preds[torch.arange(fg_mask.sum()), fg_labels]
                frcnn_reg_loss = F.smooth_l1_loss(fg_box_preds, reg_targets[fg_mask])
            else:
                frcnn_reg_loss = torch.tensor(0.0, device=feat.device)

            # mask branch - only for foreground proposals
            mask_loss = torch.tensor(0.0, device=feat.device)
            if gt_masks is not None and fg_mask.sum() > 0:
                fg_proposals = proposals[fg_mask]
                fg_gt_idx = best_gt_idx[fg_mask]

                # roi align at 14x14 for mask branch
                pooled_mask_feat = torchvision.ops.roi_align(
                    feat, [fg_proposals],
                    output_size=self.mask_pool_size,
                    spatial_scale=scale,
                    sampling_ratio=2,
                    aligned=True,
                )
                mask_logits = self.mask_head(pooled_mask_feat)  # (num_fg, num_classes, 28, 28)

                # select predicted mask for each proposal's gt class
                fg_labels_for_mask = labels[fg_mask].long()
                mask_logits = mask_logits[torch.arange(fg_mask.sum()), fg_labels_for_mask]  # (num_fg, 28, 28)

                # get gt masks for the matched gt boxes
                fg_gt_masks = gt_masks[fg_gt_idx]  # (num_fg, H, W)
                fg_gt_boxes_for_mask = all_gt_boxes[fg_gt_idx]  # (num_fg, 4)

                # crop gt masks to their gt boxes and resize to 28x28
                gt_masks_cropped = project_masks_on_boxes(
                    fg_gt_masks, fg_gt_boxes_for_mask, self.mask_out_size
                )  # (num_fg, 28, 28)

                mask_loss = F.binary_cross_entropy_with_logits(mask_logits, gt_masks_cropped)

            return {
                'frcnn_classification_loss': frcnn_cls_loss,
                'frcnn_localization_loss': frcnn_reg_loss,
                'mask_loss': mask_loss,
            }

        else:
            # inference: run detection head
            pooled_det = torchvision.ops.roi_pool(
                feat, [proposals], output_size=self.pool_size, spatial_scale=scale
            )
            pooled_det = pooled_det.flatten(start_dim=1)
            pooled_det = torch.relu(self.fc1(pooled_det))
            pooled_det = torch.relu(self.fc2(pooled_det))

            cls_scores = self.cls_layer(pooled_det)
            box_transform_pred = self.reg_layer(pooled_det)

            cls_probs = torch.softmax(cls_scores, dim=1)
            pred_scores, pred_labels = cls_probs[:, 1:].max(dim=1)
            pred_labels = pred_labels + 1

            box_transform_pred = box_transform_pred.reshape(-1, self.num_classes, 4)
            pred_boxes = apply_regression_pred_to_anchors_or_proposals(
                box_transform_pred[torch.arange(proposals.shape[0]), pred_labels], proposals
            ).squeeze(1)
            pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)

            pred_boxes, pred_labels, pred_scores = self.filter_predictions(
                pred_boxes, pred_labels, pred_scores
            )

            # mask branch for final detections
            pred_masks = None
            if len(pred_boxes) > 0:
                pooled_mask_feat = torchvision.ops.roi_align(
                    feat, [pred_boxes],
                    output_size=self.mask_pool_size,
                    spatial_scale=scale,
                    sampling_ratio=2,
                    aligned=True,
                )
                mask_logits = self.mask_head(pooled_mask_feat)  # (N, num_classes, 28, 28)
                # pick the mask for each detection's predicted class
                mask_logits = mask_logits[torch.arange(len(pred_boxes)), pred_labels]  # (N, 28, 28)
                pred_masks = torch.sigmoid(mask_logits) > 0.5  # (N, 28, 28) binary

            return {
                'boxes': pred_boxes,
                'labels': pred_labels,
                'scores': pred_scores,
                'masks': pred_masks,
            }


# ======================================================================
# mask r-cnn - extends faster r-cnn with mask branch
# ======================================================================

class MaskRCNN(FasterRCNN):
    """mask r-cnn: extends faster r-cnn with instance segmentation masks"""

    def __init__(self, model_config, num_classes):
        super().__init__(model_config, num_classes)
        # replace roi head with mask-aware version
        self.roi_head = MaskROIHead(
            model_config=model_config,
            num_classes=num_classes,
            in_channels=model_config['backbone_out_channels'],
        )

    def forward(self, image, target=None):
        """forward pass - same as faster r-cnn but mask head fires too"""
        original_image_size = image.shape[-2:]

        if target is not None:
            image, target['bboxes'] = self.normalize_resize_image_and_boxes(image, target['bboxes'])
            # scale masks if present
            if 'masks' in target and target['masks'] is not None:
                # masks stay at full image resolution - we resize them lazily
                pass
        else:
            image, _ = self.normalize_resize_image_and_boxes(image)

        resized_image_size = image.shape[-2:]
        feat = self.backbone(image)

        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']

        frcnn_output = self.roi_head(feat, proposals, resized_image_size, target)

        if not self.training:
            frcnn_output['boxes'] = transform_boxes_to_original_size(
                frcnn_output['boxes'], resized_image_size, original_image_size
            )
            # scale masks back to original image size if any detections
            if frcnn_output.get('masks') is not None and len(frcnn_output['boxes']) > 0:
                # masks are 28x28 crops - keep as-is, they're relative to the detected box
                pass

        return rpn_output, frcnn_output
