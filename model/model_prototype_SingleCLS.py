from __future__ import division, print_function
import torch
import torch.nn as nn
from model.transformer import Img_TransformerDecoderLayer, Cls_TransformerDecoderLayer

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(), nn.Dropout(dropout_p), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.LeakyReLU())

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool = nn.Sequential(nn.MaxPool2d(2))
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert len(self.ft_chns) == 5
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Decoder_cls(nn.Module):

    def __init__(self, params):
        super(Decoder_cls, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dataset = self.params['dataset']
        self.c_width = self.params['c_width']
        assert len(self.ft_chns) == 5
        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, bilinear=True)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, bilinear=True)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, bilinear=True)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, bilinear=True)
        self.CSA_256 = Cls_TransformerDecoderLayer(input_size=14, in_channels=256, d_model=512, n_class=self.n_class - 1, patch_size=14, c_width=self.c_width)
        self.out_conv = nn.Conv2d(self.ft_chns[0], 1, kernel_size=3, padding=1)

    def forward(self, feature, cls_text_features):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x4_attn = self.CSA_256(x4, cls_text_features)
        x0_d = self.up1(x4 + x4_attn, x3)
        x1_d = self.up2(x0_d, x2)
        x2_d = self.up3(x1_d, x1)
        x3_d = self.up4(x2_d, x0)
        output = self.out_conv(x3_d)
        return (output, x4_attn)

class Decoder_img(nn.Module):

    def __init__(self, params):
        super(Decoder_img, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dataset = self.params['dataset']
        self.hw_size = self.params['hw_size']
        assert len(self.ft_chns) == 5
        self.CSA_256 = Img_TransformerDecoderLayer(input_size=14, in_channels=256, d_model=512, patch_size=self.hw_size)
        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, bilinear=True)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, bilinear=True)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, bilinear=True)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, bilinear=True)
        self.out_conv = nn.Conv2d(self.ft_chns[0], 1, kernel_size=3, padding=1)

    def forward(self, feature, img_text_feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x4_attn = self.CSA_256(x4, img_text_feature)
        x0_d = self.up1(x4 + x4_attn, x3)
        x1_d = self.up2(x0_d, x2)
        x2_d = self.up3(x1_d, x1)
        x3_d = self.up4(x2_d, x0)
        output = self.out_conv(x3_d)
        return (output, x4_attn)

class MC_clip(nn.Module):

    def __init__(self, args, in_chns, class_num, bilinear=True):
        super(MC_clip, self).__init__()
        params = {'in_chns': in_chns, 'feature_chns': [16, 32, 64, 128, 256], 'dropout': [0.05, 0.1, 0.2, 0.3, 0.5], 'class_num': class_num, 'bilinear': bilinear, 'acti_func': 'relu', 'dataset': args.dataset, 'hw_size': args.hw_size, 'c_width': args.c_width}
        self.args = args
        self.alpha = 0.9
        self.num_classes = class_num
        self.encoder = Encoder(params)
        self.decoder_cls = Decoder_cls(params)
        self.decoder_img = Decoder_img(params)
        self.projection_head_cls = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1)
        self.projection_head_img = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1)
        self.projection_head = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1)
        self.register_buffer('cls_prototypes', torch.zeros(1, 256))
        self.register_buffer('img_prototypes', torch.zeros(1, 256))
        self.register_buffer('prototypes', torch.zeros(1, 256))

    def update_prototypes(self, features, labels, is_cls):
        """
        Use EMA to update class prototypes in 2D segmentation task.
        features: The feature map from the model (size: [batch_size, channels, height, width])
        labels: The ground truth label map (size: [batch_size, height, width])
        """
        batch_size, channels, height, width = features.shape
        features = features.reshape(batch_size * height * width, channels)
        labels = labels.reshape(batch_size * height * width)
        for i in range(1):
            class_features = features[labels == 1]
            if class_features.size(0) > 0:
                class_proto = class_features.mean(dim=0)
                if is_cls:
                    self.cls_prototypes[i] = self.alpha * self.cls_prototypes[i] + (1 - self.alpha) * class_proto
                else:
                    self.img_prototypes[i] = self.alpha * self.img_prototypes[i] + (1 - self.alpha) * class_proto

    def forward(self, image, img_text=None, cls_text=None):
        feats = self.encoder(image)
        output_cls, cls_attn = self.decoder_cls(feats, cls_text)
        output_img, img_attn = self.decoder_img(feats, img_text)
        cls_attn_pro = self.projection_head_cls(cls_attn)
        img_attn_pro = self.projection_head_img(img_attn)
        feats_pro = self.projection_head(feats[-1])
        return (feats_pro, feats[-1], output_cls, cls_attn, cls_attn_pro, output_img, img_attn, img_attn_pro)
