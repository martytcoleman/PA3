"""run this from the report/ directory after training to generate figures for the report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# actual training losses from 20 epochs
epochs = list(range(20))

rpn_cls = [0.1826, 0.1301, 0.1146, 0.1007, 0.0914, 0.0847, 0.0773, 0.0714,
           0.0669, 0.0628, 0.0579, 0.0568, 0.0522, 0.0490, 0.0366, 0.0333,
           0.0327, 0.0319, 0.0302, 0.0304]

rpn_loc = [0.0267, 0.0199, 0.0179, 0.0165, 0.0158, 0.0146, 0.0141, 0.0137,
           0.0131, 0.0126, 0.0123, 0.0119, 0.0116, 0.0114, 0.0092, 0.0086,
           0.0085, 0.0083, 0.0084, 0.0080]

frcnn_cls = [0.4400, 0.2876, 0.2472, 0.2165, 0.1991, 0.1880, 0.1741, 0.1642,
             0.1567, 0.1503, 0.1443, 0.1443, 0.1348, 0.1294, 0.1002, 0.0955,
             0.0935, 0.0920, 0.0898, 0.0889]

frcnn_loc = [0.0198, 0.0164, 0.0145, 0.0136, 0.0128, 0.0121, 0.0117, 0.0113,
             0.0110, 0.0108, 0.0105, 0.0102, 0.0099, 0.0099, 0.0077, 0.0075,
             0.0074, 0.0073, 0.0072, 0.0071]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# rpn losses
axes[0].plot(epochs, rpn_cls, 'b-o', markersize=4, label='RPN cls loss')
axes[0].plot(epochs, rpn_loc, 'r-s', markersize=4, label='RPN loc loss')
axes[0].axvline(x=14, color='gray', linestyle='--', alpha=0.5, label='LR decay')
axes[0].axvline(x=18, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('RPN Losses')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# detection head losses
axes[1].plot(epochs, frcnn_cls, 'g-o', markersize=4, label='FRCNN cls loss')
axes[1].plot(epochs, frcnn_loc, 'm-s', markersize=4, label='FRCNN loc loss')
axes[1].axvline(x=14, color='gray', linestyle='--', alpha=0.5, label='LR decay')
axes[1].axvline(x=18, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Detection Head Losses')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/loss_curves.png', dpi=150, bbox_inches='tight')
print('saved figures/loss_curves.png')

# mask r-cnn loss curves
mask_rpn_cls = [0.1833, 0.1313, 0.1141, 0.1011, 0.0908, 0.0849, 0.0782, 0.0713,
                0.0675, 0.0620, 0.0587, 0.0552, 0.0518, 0.0504, 0.0370, 0.0341,
                0.0326, 0.0319, 0.0308, 0.0305]

mask_frcnn_cls = [0.4454, 0.2842, 0.2455, 0.2202, 0.1990, 0.1850, 0.1735, 0.1633,
                  0.1594, 0.1495, 0.1442, 0.1390, 0.1321, 0.1330, 0.1010, 0.0952,
                  0.0947, 0.0921, 0.0899, 0.0892]

mask_loss = [0.0580, 0.0506, 0.0469, 0.0448, 0.0434, 0.0426, 0.0419, 0.0413,
             0.0410, 0.0403, 0.0398, 0.0395, 0.0391, 0.0389, 0.0376, 0.0373,
             0.0372, 0.0372, 0.0370, 0.0371]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs, mask_rpn_cls, 'b-o', markersize=4, label='RPN cls loss')
axes[0].plot(epochs, mask_frcnn_cls, 'g-s', markersize=4, label='FRCNN cls loss')
axes[0].axvline(x=14, color='gray', linestyle='--', alpha=0.5, label='LR decay')
axes[0].axvline(x=18, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Mask R-CNN Classification Losses')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, mask_loss, 'r-o', markersize=4, label='Mask loss')
axes[1].axvline(x=14, color='gray', linestyle='--', alpha=0.5, label='LR decay')
axes[1].axvline(x=18, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Mask R-CNN Mask Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/mask_loss_curves.png', dpi=150, bbox_inches='tight')
print('saved figures/mask_loss_curves.png')

# faster r-cnn per-class ap bar chart
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
aps = [59.66, 68.31, 56.66, 42.45, 33.78, 68.08, 68.24, 76.43, 37.29, 63.49,
       53.84, 72.33, 71.82, 67.89, 65.59, 34.35, 55.48, 52.20, 72.38, 66.60]

fig, ax = plt.subplots(figsize=(14, 5))
colors = ['#2196F3' if ap >= 60 else '#FF9800' if ap >= 45 else '#F44336' for ap in aps]
bars = ax.bar(classes, aps, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=59.34, color='black', linestyle='--', linewidth=1.5, label='mAP = 59.34%')
ax.set_xlabel('Class')
ax.set_ylabel('Average Precision (%)')
ax.set_title('Faster R-CNN Per-Class AP on VOC 2007 Test Set')
ax.set_ylim(0, 90)
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures/per_class_ap.png', dpi=150, bbox_inches='tight')
print('saved figures/per_class_ap.png')

# mask r-cnn per-class ap bar chart
mask_aps = [66.03, 67.76, 55.02, 40.49, 33.71, 66.37, 68.31, 76.64, 37.00, 63.95,
            53.20, 70.82, 75.68, 68.42, 66.08, 37.21, 56.76, 54.12, 71.72, 63.19]

fig, ax = plt.subplots(figsize=(14, 5))
colors = ['#2196F3' if ap >= 60 else '#FF9800' if ap >= 45 else '#F44336' for ap in mask_aps]
ax.bar(classes, mask_aps, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=59.62, color='black', linestyle='--', linewidth=1.5, label='mAP = 59.62%')
ax.set_xlabel('Class')
ax.set_ylabel('Average Precision (%)')
ax.set_title('Mask R-CNN Per-Class AP on VOC 2007 Test Set')
ax.set_ylim(0, 90)
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures/mask_per_class_ap.png', dpi=150, bbox_inches='tight')
print('saved figures/mask_per_class_ap.png')
