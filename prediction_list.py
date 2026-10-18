import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

def load_model_checkpoint(model, path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    key = "decoder_img.CSA_256.text_storage_buffer"
    if key in state:
        model.decoder_img.CSA_256.text_storage_buffer = state[key].to(device)
    model.load_state_dict(state, strict=True)
    return model


def getLargestCC(segmentation):
    from skimage.measure import label
    labels = label(segmentation)

    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        largestCC = segmentation
    return largestCC


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice_res = []
    if pred.sum() > 0:
        dice_res.append(metric.binary.dc(pred, gt))
    else:
        dice_res.append(0)

    return dice_res


def test_single_volume(args, image, label, class_text, model):
    classes = args.num_classes
    patch_size = [args.image_size, args.image_size]
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(next(model.parameters()).device)

        with torch.no_grad():
            feats_pro, feats, output_cls, cls_attn, cls_attn_pro, output_img, img_attn, img_attn_pro = model(input, None, class_text)

            output = (output_cls + output_img) / 2
            output_soft = torch.sigmoid(output)
            out = (output_soft > 0.5).squeeze(0).squeeze(0).cpu().detach().numpy()

            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        pred = getLargestCC(prediction == i)
        metric_list.append(calculate_metric_percase(pred, label == i))
    return metric_list




if __name__ == '__main__':
    import cv2
    import torch
    import argparse
    from dataloader.dataset import build_Dataset
    from dataloader.transforms import build_transforms
    from torch.utils.data import DataLoader
    import numpy as np
    from model.CLIP.clip import clip
    from model.model_prototype_SingleCLS import MC_clip

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/opt/data/private/data_2D',
                        help='Name of Experiment')

    parser.add_argument('--dataset', type=str, default='/PROMISE12',
                        help='Name of Experiment')

    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='input channel of network')
    parser.add_argument('--image_size', type=int, default=256,
                        help='patch size of network input')

    parser.add_argument('--model_path', type=str,
                        default="./Results/Results_compare/result_PROMISE_20_256_NoShuffle_LossDe_B72_c4p2_NoConv/fold_0/best_model_22400.pth",
                        help='model weight path')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--hw_size', type=int, default=1)
    parser.add_argument('--c_width', type=int, default=1)
    args = parser.parse_args()
    data_transforms = build_transforms(args)

    test_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="test_list",
                                 transform=data_transforms["valid_test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)



    model = MC_clip(args=args, in_chns=args.in_channels, class_num=args.num_classes).to(args.device)
    load_model_checkpoint(model, args.model_path, args.device)
    model.eval()

    class_text = test_dataset.cls_text.to(args.device)

    classes = args.num_classes
    patch_size = [args.image_size, args.image_size]
    final_res = np.zeros(4, dtype=np.float64)

    for i_batch, sampled_batch in enumerate(test_loader):
        test_image, test_label = sampled_batch["image"].to(args.device), sampled_batch["label"].to(args.device)
        image, label = test_image.squeeze(0).cpu().detach().numpy(), test_label.squeeze(0).cpu().detach().numpy()
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(next(model.parameters()).device)

            with torch.no_grad():
                feats_pro, feats, output_cls, cls_attn, cls_attn_pro, output_img, img_attn, img_attn_pro = model(input, None, class_text)
                output_sig = torch.sigmoid((output_cls + output_img) / 2)

                out = (output_sig > 0.5).squeeze(0).squeeze(0).cpu().detach().numpy()

                pred_SGDL = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred_SGDL

        metric_list = []
        for i in range(1, classes):
            disc_pred = prediction == i
            gt = label == i
            disc_pred[disc_pred > 0] = 1
            disc_pred = getLargestCC(disc_pred)
            gt[gt > 0] = 1
            single_class_res = []
            if disc_pred.sum() > 0:
                single_class_res.append(metric.binary.dc(disc_pred, gt))
                single_class_res.append(metric.binary.jc(disc_pred, gt))
                single_class_res.append(metric.binary.asd(disc_pred, gt))
                single_class_res.append(metric.binary.hd95(disc_pred, gt))

            else:
                single_class_res = [0, 0, 0, 0]
            metric_list.append(single_class_res)

        metric_list = np.array(metric_list).astype("float32")
        metric_list = np.mean(metric_list, axis=0)

        print(metric_list)
        final_res += metric_list

    final_res = [x / len(test_loader) for x in final_res]

    print("avg_dice: ", final_res[0])
    print("avg_iou: ", final_res[1])
    print("avg_asd: ", final_res[2])
    print("avg_hd95: ", final_res[3])

