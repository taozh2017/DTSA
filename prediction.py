import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
import torch
import argparse
import numpy as np
from utils.metrics import eval
from torch.utils.data import DataLoader
from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms


color_list = [0.1, 0.1, 0.9]


def load_model_checkpoint(model, path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    key = "decoder_img.CSA_256.text_storage_buffer"
    if key in state:
        model.decoder_img.CSA_256.text_storage_buffer = state[key].to(device)
    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def test():
    from model.model_prototype_SingleCLS import MC_clip

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/opt/data/private/data_2D',
                        help='Name of Experiment')

    parser.add_argument('--dataset', type=str, default='/tumor_10',
                        help='Name of Experiment')

    parser.add_argument('--hw_size', type=int, default=1,
                        help='output channel of network')
    parser.add_argument('--c_width', type=int, default=1,
                        help='output channel of network')

    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='input channel of network')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')

    parser.add_argument('--model_path', type=str,
                        default="./Results/Results_compare/result_tumor_10_B72_NoShuffle_LossDe_c1p1_Best(Opt)/fold_0/best_model_cls_27600.pth",
                        help='model weight path')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_splits', nargs='+', default=None, help='Dataset split names to evaluate')
    args = parser.parse_args()

    data_transforms = build_transforms(args)

    if args.test_splits:
        test_dataset_list = args.test_splits
    elif "tumor" in args.dataset:
        test_dataset_list = ["test_CVC-ClinicDB", "test_Kvasir"]
    elif "ISIC" in args.dataset:
        test_dataset_list = ["test_ISIC2018"]
    elif "BrainMRI" in args.dataset:
        test_dataset_list = ["test_BrainMRI"]
    elif "thyroid" in args.dataset:
        test_dataset_list = ["test_DDTI", "test_tn3k"]
    elif "BUSI" in args.dataset:
        test_dataset_list = ["test_BUSI"]
    elif "BCSS" in args.dataset:
        test_dataset_list = ["test_BCSS"]
    else:
        parser.error('Specify --test_splits for this dataset.')


    model = MC_clip(args=args, in_chns=args.in_channels, class_num=args.num_classes).to(args.device)

    load_model_checkpoint(model, args.model_path, args.device)
    model.eval()

    for test_dataset_name in test_dataset_list:
        test_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split=test_dataset_name,
                                     transform=data_transforms["valid_test"])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        avg_dice_list = []
        avg_iou_list = []
        avg_hd95_list = []

        for i_batch, sampled_batch in enumerate(test_loader):
            test_image, test_label, image_text, class_text  = sampled_batch['image'].to(args.device), sampled_batch['label'].to(args.device), sampled_batch['image_text'].squeeze(1).to(args.device), sampled_batch['class_text'].squeeze(1).to(args.device)
            feats_pro, feats, output_cls, cls_attn, cls_attn_pro, output_img, img_attn, img_attn_pro = model(test_image, None, class_text)

            test_label = test_label.unsqueeze(1).float()

            output_sig = torch.sigmoid((output_cls + output_img) / 2)
            eval_list = eval(test_label, output_sig, thr=0.5)

            avg_dice_list.append(eval_list[0])
            avg_iou_list.append(eval_list[1])
            avg_hd95_list.append(eval_list[2])

        avg_dice = np.mean(avg_dice_list)
        avg_hd95 = np.mean(avg_hd95_list)
        avg_iou = np.mean(avg_iou_list)

        print(test_dataset_name, " :")
        print("avg_dice: ", avg_dice)
        print("avg_iou: ", avg_iou)
        print("avg_Rec: ", avg_hd95)

    return avg_dice, avg_iou, avg_hd95


if __name__ == '__main__':
    test()

