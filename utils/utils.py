import torch
import numpy as np
from hausdorff import hausdorff_distance
color_list = [0.1, 0.1, 0.9]

def dice_coef(y_true, y_pred, thr=0.5, epsilon=0.001):
    y_true = (y_true > thr).to(torch.float32).squeeze(0).cpu().detach().numpy()
    if y_pred.shape[1] > 1:
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    else:
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
    inter_map = y_true * y_pred
    inter = inter_map.sum()
    den = y_true.sum() + y_pred.sum()
    dice = 2 * inter / (den + epsilon) if den > 0 else 0
    return dice

def evaluate_95hd(y_true, y_pred, thr=0.5):
    y_true = (y_true > thr).to(torch.float32).squeeze(0).cpu().detach().numpy()
    if y_pred.shape[1] > 1:
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    else:
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
    hd = hausdorff_distance(y_true, y_pred)
    return hd * 0.95

def patients_to_slices(dataset, patiens_num):
    ref_dict = {}
    if 'BUSI' in dataset:
        ref_dict = {'1': 5, '5': 26, '10': 52, '20': 104, '30': 155, '50': 260, '80': 416}
    elif 'tumor' in dataset:
        ref_dict = {'1': 15, '5': 73, '10': 145, '20': 290, '30': 435, '50': 725, '70': 1015, '80': 1160, '90': 1305}
    elif 'ISIC' in dataset:
        ref_dict = {'1': 21, '5': 104, '10': 207, '20': 414, '30': 622, '50': 1035, '70': 1449, '80': 1656, '90': 1863}
    elif 'thyroid' in dataset:
        ref_dict = {'1': 61, '5': 307, '10': 613, '20': 1226, '30': 1841, '50': 3065, '70': 4291, '80': 4904, '90': 5517}
    elif 'BrainMRI' in dataset:
        ref_dict = {'1': 10, '5': 52, '10': 103, '20': 206, '30': 310, '50': 515, '70': 721, '80': 824, '90': 927}
    elif 'MRI_Hippocampus_Seg' in dataset:
        ref_dict = {'10': 282, '30': 846}
    elif 'BCSS' in dataset:
        ref_dict = {'10': 272, '30': 816}
    else:
        print('Error')
    return ref_dict[str(patiens_num)]

def generate_mask(img):
    batch_size, channel, img_x, img_y = (img.shape[0], img.shape[1], img.shape[2], img.shape[3])
    loss_mask = torch.ones(batch_size, img_x, img_y, device=img.device)
    mask = torch.ones(img_x, img_y, device=img.device)
    patch_x, patch_y = (int(img_x * 2 / 3), int(img_y * 2 / 3))
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return (mask.long(), loss_mask.long())
