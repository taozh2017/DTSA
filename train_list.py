import argparse
import numpy as np
import random
import torch
import os
import logging
from tqdm import tqdm
from dataloader.dataset import build_Dataset
from torch.utils.data import DataLoader
from utils.utils import patients_to_slices
from dataloader.transforms import build_transforms
from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler


from trainer_list_MutiGPU import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/opt/data/private/data_2D',
                    help='Name of Experiment')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='Percentage of label quantity')


parser.add_argument('--dataset', type=str, default='/PROMISE12',
                    help='Name of Experiment')

parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--in_channels', type=int, default=1,
                    help='input channel of network')

parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')

parser.add_argument('--image_size', type=int, default=256, help='image_size')

parser.add_argument('--batch_size', type=int, default=124,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=62,
                    help='labeled_batch_size per gpu')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

parser.add_argument('--max_iterations', type=int, default=30000,
                    help='maximum epoch number to train')

parser.add_argument('--n_fold', type=int, default=1,
                    help='maximum epoch number to train')
parser.add_argument('--consistency', type=float, default=0.1,
                    help='consistency')

parser.add_argument('--hw_size', type=int, default=1,
                    help='output channel of network')
parser.add_argument('--c_width', type=int, default=1,
                    help='output channel of network')


parser.add_argument('--weit_coef', type=float, default=2,
                    help='Loss weit_coef')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--device', type=str, default='cuda')


parser.add_argument('--output_dir', type=str, default='./Results', help='Directory for training outputs')
parser.add_argument('--eval_interval', type=int, default=200, help='Validation interval in training steps')
parser.add_argument('--num_workers', type=int, default=2, help='Training data loader workers')
args = parser.parse_args()
if args.batch_size != 2 * args.labeled_bs or args.labeled_bs % 2:
    parser.error('This method requires batch_size == 2 * labeled_bs and an even labeled_bs.')
if args.max_iterations < 1 or args.eval_interval < 1:
    parser.error('max_iterations and eval_interval must be positive.')


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def train(args, snapshot_path, logger):
    batch_size = args.batch_size

    max_iterations = args.max_iterations

    trainer = Trainer(args)

    data_transforms = build_transforms(args)
    train_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="train_semi_list",
                                  transform=data_transforms["train"])

    val_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="val_list",
                                transform=data_transforms["valid_test"])

    total_slices = len(train_dataset)
    labeled_slice = patients_to_slices(args.dataset, args.labeled_num)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=torch.device(args.device).type == "cuda", worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    logger.info("{} iterations per epoch".format(len(train_loader)))

    max_epoch = max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    iter_num = 0
    for _ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch, image_text, class_text = sampled_batch['image'].to(args.device), sampled_batch['label'].to(args.device), sampled_batch['image_text'].squeeze(1).to(args.device), sampled_batch['class_text'].squeeze(1).to(args.device)
            trainer.train(volume_batch, label_batch, image_text, class_text, logger, iter_num)
            iter_num = iter_num + 1
            if iter_num > 0 and iter_num % args.eval_interval == 0:
                trainer.val_list(val_loader, snapshot_path, logger, iter_num)
            if iter_num >= max_iterations:
                torch.save(trainer.model.state_dict(), os.path.join(snapshot_path, "last_model.pth"))
                return


if __name__ == '__main__':
    import shutil

    for fold in range(args.n_fold):
        torch.autograd.set_detect_anomaly(True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        snapshot_path = os.path.join(args.output_dir, args.dataset.strip('/'), 'fold_' + str(fold))

        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        if not os.path.exists(snapshot_path + '/code'):
            os.makedirs(snapshot_path + '/code')

        shutil.copyfile("./train_list.py", snapshot_path + "/code/train_list.py")
        shutil.copyfile("./trainer_list_MutiGPU.py", snapshot_path + "/code/trainer_list_MutiGPU.py")

        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(snapshot_path, "log.txt"))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Log file created successfully.")

        train(args, snapshot_path, logger)

