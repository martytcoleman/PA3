import torch
import argparse
import os
import numpy as np
import yaml
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mask_rcnn import MaskRCNN
from dataset.voc_mask import VOCMaskDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('using device:', device)


def collate_fn(batch):
    """custom collate since masks can be None for some images"""
    images, targets, fnames = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets), list(fnames)


def train(args):
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
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

    voc = VOCMaskDataset(
        'train',
        im_dir=dataset_config['im_train_path'],
        ann_dir=dataset_config['ann_train_path'],
        seg_dir=dataset_config['seg_train_path'],
    )
    train_dataset = DataLoader(
        voc, batch_size=1, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )

    model = MaskRCNN(model_config, num_classes=dataset_config['num_classes'])
    model.train()
    model.to(device)

    os.makedirs(train_config['task_name'], exist_ok=True)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_config['lr'],
        weight_decay=5e-4,
        momentum=0.9,
    )
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)

    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1

    for epoch in range(num_epochs):
        rpn_cls_losses, rpn_loc_losses = [], []
        frcnn_cls_losses, frcnn_loc_losses, mask_losses = [], [], []
        optimizer.zero_grad()

        for im, targets, fnames in tqdm(train_dataset):
            im = im.float().to(device)
            target = targets[0]  # batch size 1
            target['bboxes'] = target['bboxes'].float().to(device).unsqueeze(0)
            target['labels'] = target['labels'].long().to(device).unsqueeze(0)
            if target['masks'] is not None:
                target['masks'] = target['masks'].to(device)

            rpn_output, frcnn_output = model(im, target)

            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = (frcnn_output['frcnn_classification_loss']
                         + frcnn_output['frcnn_localization_loss']
                         + frcnn_output['mask_loss'])
            loss = (rpn_loss + frcnn_loss) / acc_steps

            rpn_cls_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_loc_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_cls_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_loc_losses.append(frcnn_output['frcnn_localization_loss'].item())
            mask_losses.append(frcnn_output['mask_loss'].item())

            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1

        print(f'epoch {epoch} done')
        optimizer.step()
        optimizer.zero_grad()

        ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
        torch.save(model.state_dict(), ckpt_path)

        print(
            f"rpn cls: {np.mean(rpn_cls_losses):.4f} | "
            f"rpn loc: {np.mean(rpn_loc_losses):.4f} | "
            f"frcnn cls: {np.mean(frcnn_cls_losses):.4f} | "
            f"frcnn loc: {np.mean(frcnn_loc_losses):.4f} | "
            f"mask: {np.mean(mask_losses):.4f}"
        )
        scheduler.step()

    print('done training mask r-cnn')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/voc_mask.yaml', type=str)
    args = parser.parse_args()
    train(args)
