import os
from model.CLIP.clip import clip
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class build_Dataset(Dataset):

    def __init__(self, args, data_dir, split, transform=None, model='None'):
        self.args = args
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.model = model
        self.Clip, _ = clip.load('ViT-B/32', device='cpu')
        for param in self.Clip.parameters():
            param.requires_grad = False
        if 'tumor' in args.dataset:
            cls_text_token = clip.tokenize(['A pixel of polyp'])
        elif 'ISIC' in args.dataset:
            cls_text_token = clip.tokenize(['A pixel of lesion'])
        elif 'thyroid' in args.dataset:
            cls_text_token = clip.tokenize(['A pixel of nodule'])
        elif 'BrainMRI' in args.dataset:
            cls_text_token = clip.tokenize(['A pixel of brain tumor'])
        elif 'BUSI' in args.dataset:
            cls_text_token = clip.tokenize(['A pixel of tumor'])
        elif 'BCSS' in args.dataset:
            cls_text_token = clip.tokenize(['A pixel of tumor'])
        with torch.no_grad():
            self.cls_text = self.Clip.encode_text(cls_text_token).to(torch.float32).unsqueeze(0)
        del self.Clip
        if self.split == 'train_semi':
            labeled_path = os.path.join(self.data_dir + '/labeled/image')
            unlabeled_path = os.path.join(self.data_dir + '/unlabeled/image')
            sample_list_labeled = os.listdir(labeled_path)
            sample_list_unlabeled = os.listdir(unlabeled_path)
            self.sample_list_labeled = [os.path.join(labeled_path, item) for item in sample_list_labeled]
            self.sample_list_unlabeled = [os.path.join(unlabeled_path, item) for item in sample_list_unlabeled]
            self.sample_list = self.sample_list_labeled + self.sample_list_unlabeled
            print('train total {} labeled samples, {} unlabeled samples'.format(len(sample_list_labeled), len(sample_list_unlabeled)))
        elif self.split == 'train_BUSI':
            labeled_path = os.path.join(self.data_dir + '/train.list')
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + '/images/' + image_name for image_name in self.image_list]
            print('train total {} samples'.format(len(self.sample_list)))
        elif self.split == 'val':
            val_path = os.path.join(self.data_dir + '/val/image')
            sample_list_val = os.listdir(val_path)
            self.sample_list = [os.path.join(val_path, item) for item in sample_list_val]
            print('val total {} samples'.format(len(self.sample_list)))
        elif self.split == 'test_BUSI':
            labeled_path = os.path.join(self.data_dir + '/test.list')
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + '/images/' + image_name for image_name in self.image_list]
            print('test total {} samples'.format(len(self.sample_list)))
        elif self.split == 'train_BCSS':
            labeled_path = os.path.join(self.data_dir + '/train.list')
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + '/images/' + image_name for image_name in self.image_list]
            print('train total {} samples'.format(len(self.sample_list)))
        elif self.split == 'test_BCSS':
            test_list_path = os.path.join(self.data_dir + '/test.list')
            with open(test_list_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + '/images/' + image_name for image_name in self.image_list]
            print('test total {} samples'.format(len(self.sample_list)))
        elif 'test' in self.split:
            if 'tumor' in self.split:
                test_path = os.path.join(self.data_dir + '/test_all_3/image')
            if 'thyroid' in self.split:
                test_path = os.path.join(self.data_dir + '/test_all/image')
            elif 'CVC-300' in self.split:
                test_path = os.path.join(self.data_dir + '/TestDataset/CVC-300/image')
            elif 'CVC-ClinicDB' in self.split:
                test_path = os.path.join(self.data_dir + '/TestDataset/CVC-ClinicDB/image')
            elif 'CVC-ColonDB' in self.split:
                test_path = os.path.join(self.data_dir + '/TestDataset/CVC-ColonDB/image')
            elif 'ETIS-LaribPolypDB' in self.split:
                test_path = os.path.join(self.data_dir + '/TestDataset/ETIS-LaribPolypDB/image')
            elif 'Kvasir' in self.split:
                test_path = os.path.join(self.data_dir + '/TestDataset/Kvasir/image')
            elif 'ISIC2018' in self.split:
                test_path = os.path.join(self.data_dir + '/TestDataset/image')
            elif 'DDTI' in self.split:
                test_path = os.path.join(self.data_dir + '/TestDataset/DDTI/image')
            elif 'tn3k' in self.split:
                test_path = os.path.join(self.data_dir + '/TestDataset/tn3k/image')
            elif 'BrainMRI' in self.split:
                test_path = os.path.join(self.data_dir + '/TestDataset/image')
            elif 'MRI_Hippocampus' in self.split:
                test_path = os.path.join(self.data_dir + '/TestDataset/image')
            print('test_path: ', test_path)
            sample_list_val = os.listdir(test_path)
            self.sample_list = [os.path.join(test_path, item) for item in sample_list_val]
            print('test total {} samples'.format(len(self.sample_list)))
        if 'train' in self.split or 'val' in self.split:
            if 'tumor' in args.dataset:
                self.image_text_dict = torch.load('./text_prompt/train_tumor_simple.pth', map_location='cpu', weights_only=True)
            elif 'ISIC' in args.dataset:
                self.image_text_dict = torch.load('./text_prompt/train_ISIC_simple.pth', map_location='cpu', weights_only=True)
            elif 'thyroid' in args.dataset:
                self.image_text_dict = torch.load('./text_prompt/train_thyroid_simple.pth', map_location='cpu', weights_only=True)
            elif 'BrainMRI' in args.dataset:
                self.image_text_dict = torch.load('./text_prompt/train_BrainMRI_simple.pth', map_location='cpu', weights_only=True)
            elif 'BUSI' in args.dataset:
                self.image_text_dict = torch.load('./text_prompt/BUSI_simple.pth', map_location='cpu', weights_only=True)
            elif 'BCSS' in args.dataset:
                self.image_text_dict = torch.load('./text_prompt/train_BCSS_simple.pth', map_location='cpu', weights_only=True)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image_name = os.path.basename(case)
        if 'test' in self.split or 'val' in self.split:
            image_text_features = torch.zeros([1, 1, 512])
        else:
            image_text_features = self.image_text_dict[image_name]
        image = cv2.cvtColor(cv2.imread(case), cv2.IMREAD_COLOR) / 255.0
        image = image.astype(np.float32)
        label_path = case.replace('image', 'mask')
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) / 255
        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']
        image = image.transpose(2, 0, 1).astype('float32')
        image, label = (torch.tensor(image), torch.tensor(label))
        sample = {'image': image, 'label': label, 'image_text': image_text_features, 'class_text': self.cls_text}
        return sample
