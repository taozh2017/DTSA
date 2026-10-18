import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import generate_mask
import numpy as np


from model.model_prototype_SingleCLS_MutiGPU import MC_clip


from prediction_list import test_single_volume


def get_cut_mask(P, temperature=0.1):
    P_sharpen = torch.sigmoid(P / temperature)

    pseudo_labels = P_sharpen > 0.5

    return pseudo_labels.squeeze(1)


class PolyWarmRestartScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, max_iters, power=0.9, warm_restart_iters=5000, last_epoch=-1):
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.power = power
        self.warm_restart_iters = warm_restart_iters
        self.current_cycle_start = 0
        super(PolyWarmRestartScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch - self.current_cycle_start
        if t >= self.warm_restart_iters:
            self.current_cycle_start = self.last_epoch
            t = 0
        factor = (1 - t / self.warm_restart_iters) ** self.power
        return [self.base_lr * factor for _ in self.base_lrs]


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        self.ce_loss = nn.BCELoss(reduction='none')

        self.model = MC_clip(args=args, in_chns=args.in_channels, class_num=args.num_classes).to(args.device).train()

        if torch.device(args.device).type == "cuda" and torch.cuda.device_count() > 1:
            primary = torch.device(args.device).index or 0
            device_ids = [primary] + [index for index in range(torch.cuda.device_count()) if index != primary]
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

        self.scheduler = PolyWarmRestartScheduler(self.optimizer,
                                            base_lr=args.lr,
                                            max_iters=args.max_iterations,
                                            power=0.9,
                                            warm_restart_iters=10000
                                            )

        self.best_performance = 0.0

    @property
    def core_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, epoch):
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def get_entropy_map(self, p):
        """
        输入:
            logits: [B, 1, H, W] (未经过 Sigmoid 的原始输出)
        输出:
            ent_map: [B, 1, H, W] (数值越大表示越不确定，0.5时最大)
        """
        epsilon = 1e-6
        ent_map = - (p * torch.log(p + epsilon) + (1 - p) * torch.log(1 - p + epsilon))
        return ent_map

    def seg_loss(self, pred, mask, smooth=1e-6, is_sim=False):
        mask_one_hot = mask.unsqueeze(1).float()

        if not is_sim:
            pred_sig = torch.sigmoid(pred)
            uncertainty_weit = self.get_entropy_map(pred_sig)
        else:
            _, height, width = mask.shape
            pred = F.interpolate(pred, size=[height, width], mode='bilinear')
            pred_sig = torch.sigmoid(pred)
            uncertainty_weit = self.get_entropy_map(pred_sig)

        weit_coef = self.args.weit_coef
        weit = uncertainty_weit * weit_coef + 1

        wbce = self.ce_loss(pred_sig, mask_one_hot)
        wbce = (weit * wbce).mean()

        inter = (pred_sig * mask_one_hot * weit).sum(dim=(2, 3))
        union = (pred_sig * weit).sum(dim=(2, 3)) + (mask_one_hot * weit).sum(dim=(2, 3)) - inter
        wiou = 1 - ((inter + smooth)/(union + smooth)).mean(dim=1)

        if wiou.mean() < 0:
            print("Error", wiou.mean())

        loss = wbce + wiou.mean()
        return loss

    def cosine_similarity(self, feats, prototypes):
        dist_list = []
        for prototype in prototypes:
            dist = F.cosine_similarity(feats, prototype[..., None, None], dim=1)
            dist_list.append(dist)
        return torch.stack(dist_list, dim=1)

    def train(self, volume_batch, label_batch, image_text, class_text, logger, iter_num):
        _, _, output_cls_lab, cls_attn_lab, cls_attn_pro_lab, output_img_lab, img_attn_lab, img_attn_pro_lab = self.model(volume_batch[:self.args.labeled_bs], image_text[:self.args.labeled_bs], class_text[:self.args.labeled_bs])
        self.core_model.decoder_img.CSA_256.add_image_text(image_text[:self.args.labeled_bs])
        _, _, output_cls_unlab, cls_attn_unlab, cls_attn_pro_unlab, output_img_unlab, img_attn_unlab, img_attn_pro_unlab = self.model(volume_batch[self.args.labeled_bs:], None, class_text[self.args.labeled_bs:])

        cls_pseudo_label = get_cut_mask(output_cls_unlab).to(self.args.device)
        img_pseudo_label = get_cut_mask(output_img_unlab).to(self.args.device)

        volume_batch_unlab_weak = volume_batch[self.args.labeled_bs:]
        mask, _ = generate_mask(volume_batch_unlab_weak)
        volume_batch_unlab_strong_1 = volume_batch_unlab_weak[:self.args.labeled_bs//2] * mask + volume_batch_unlab_weak[self.args.labeled_bs//2:] * (1 - mask)
        volume_batch_unlab_strong_2 = volume_batch_unlab_weak[:self.args.labeled_bs//2] * (1 - mask) + volume_batch_unlab_weak[self.args.labeled_bs//2:] * mask
        volume_batch_unlab_strong = torch.concat([volume_batch_unlab_strong_1, volume_batch_unlab_strong_2], dim=0)

        feats_pro_unlab, feats_unlab, output_cls_unlab_strong, cls_attn_unlab_strong, cls_attn_pro_unlab_strong, output_img_unlab_strong, img_attn_unlab_strong, img_attn_pro_unlab_strong = self.model(volume_batch_unlab_strong, None, class_text[self.args.labeled_bs:])

        _, _, height, width = cls_attn_lab.shape
        mask_down = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=[height, width], mode='nearest').squeeze(0).squeeze(0).long()
        output_cls_unlab_strong_1 = output_cls_unlab_strong[:self.args.labeled_bs//2] * mask + output_cls_unlab_strong[self.args.labeled_bs//2:] * (1 - mask)
        output_cls_unlab_strong_2 = output_cls_unlab_strong[self.args.labeled_bs//2:] * mask + output_cls_unlab_strong[:self.args.labeled_bs//2] * (1 - mask)
        output_cls_unlab_strong_restore = torch.concat([output_cls_unlab_strong_1, output_cls_unlab_strong_2], dim=0)
        output_img_unlab_strong_1 = output_img_unlab_strong[:self.args.labeled_bs//2] * mask + output_img_unlab_strong[self.args.labeled_bs//2:] * (1 - mask)
        output_img_unlab_strong_2 = output_img_unlab_strong[self.args.labeled_bs//2:] * mask + output_img_unlab_strong[:self.args.labeled_bs//2] * (1 - mask)
        output_img_unlab_strong_restore = torch.concat([output_img_unlab_strong_1, output_img_unlab_strong_2], dim=0)

        cls_attn_unlab_strong_1 = cls_attn_unlab_strong[:self.args.labeled_bs//2] * mask_down + cls_attn_unlab_strong[self.args.labeled_bs//2:] * (1 - mask_down)
        cls_attn_unlab_strong_2 = cls_attn_unlab_strong[self.args.labeled_bs//2:] * mask_down + cls_attn_unlab_strong[:self.args.labeled_bs//2] * (1 - mask_down)
        cls_attn_unlab_strong_restore = torch.concat([cls_attn_unlab_strong_1, cls_attn_unlab_strong_2], dim=0)
        img_attn_unlab_strong_1 = img_attn_unlab_strong[:self.args.labeled_bs//2] * mask_down + img_attn_unlab_strong[self.args.labeled_bs//2:] * (1 - mask_down)
        img_attn_unlab_strong_2 = img_attn_unlab_strong[self.args.labeled_bs//2:] * mask_down + img_attn_unlab_strong[:self.args.labeled_bs//2] * (1 - mask_down)
        img_attn_unlab_strong_restore = torch.concat([img_attn_unlab_strong_1, img_attn_unlab_strong_2], dim=0)

        output_cls_sup_loss = self.seg_loss(output_cls_lab, label_batch[:self.args.labeled_bs].float())
        output_img_sup_loss = self.seg_loss(output_img_lab, label_batch[:self.args.labeled_bs].float())

        output_cls_unsup_loss = (self.seg_loss(output_cls_unlab_strong_restore, img_pseudo_label.float()) + self.seg_loss(output_cls_unlab, img_pseudo_label.float())) / 2
        output_img_unsup_loss = (self.seg_loss(output_img_unlab_strong_restore, cls_pseudo_label.float()) + self.seg_loss(output_img_unlab, cls_pseudo_label.float())) / 2

        text_loss = 0

        label_one_hot = label_batch[:self.args.labeled_bs].unsqueeze(1).float()
        label_one_hot = F.interpolate(label_one_hot, size=[height, width], mode='nearest')

        cur_sup_cls_attn_map = (label_one_hot * cls_attn_pro_lab).mean(dim=[-2, -1])
        cur_sup_img_attn_map = (label_one_hot * img_attn_pro_lab).mean(dim=[-2, -1])
        text_loss += F.mse_loss(cur_sup_cls_attn_map, class_text[:self.args.labeled_bs].squeeze(1)) + F.mse_loss(cur_sup_img_attn_map, image_text[:self.args.labeled_bs].squeeze(1))

        sim_lab_cls_dist = self.cosine_similarity(img_attn_lab, self.core_model.cls_prototypes.detach())
        sim_lab_img_dist = self.cosine_similarity(cls_attn_lab, self.core_model.img_prototypes.detach())

        sim_unlab_cls_dist = self.cosine_similarity(img_attn_unlab, self.core_model.cls_prototypes.detach())
        sim_unlab_img_dist = self.cosine_similarity(cls_attn_unlab, self.core_model.img_prototypes.detach())

        sim_unlab_strong_cls_dist = self.cosine_similarity(img_attn_unlab_strong_restore, self.core_model.cls_prototypes.detach())
        sim_unlab_strong_img_dist = self.cosine_similarity(cls_attn_unlab_strong_restore, self.core_model.img_prototypes.detach())

        sim_lab_label = F.interpolate(label_batch[:self.args.labeled_bs].unsqueeze(1), size=[height, width], mode='nearest').squeeze(1)

        sim_cls_unlab_label = F.interpolate(cls_pseudo_label.float().unsqueeze(1), size=[height, width], mode='nearest').squeeze(1)
        sim_img_unlab_label = F.interpolate(img_pseudo_label.float().unsqueeze(1), size=[height, width], mode='nearest').squeeze(1)

        sim_sup_loss = (self.seg_loss(sim_lab_cls_dist, sim_lab_label, is_sim=True) + self.seg_loss(sim_lab_img_dist, sim_lab_label, is_sim=True)) / 2
        sim_unsup_loss = (self.seg_loss(sim_unlab_cls_dist, sim_img_unlab_label, is_sim=True) + self.seg_loss(sim_unlab_img_dist, sim_cls_unlab_label, is_sim=True) + \
                        self.seg_loss(sim_unlab_strong_cls_dist, sim_img_unlab_label, is_sim=True) + self.seg_loss(sim_unlab_strong_img_dist, sim_cls_unlab_label, is_sim=True)) / 4

        cls_attn = torch.concat([cls_attn_lab, cls_attn_unlab], dim=0)
        img_attn = torch.concat([img_attn_lab, img_attn_unlab], dim=0)
        cls_attn_label = F.interpolate(torch.concat([label_batch[:self.args.labeled_bs], cls_pseudo_label.float()], dim=0).unsqueeze(1), size=[height, width], mode='nearest').squeeze(1)
        img_attn_label = F.interpolate(torch.concat([label_batch[:self.args.labeled_bs], img_pseudo_label.float()], dim=0).unsqueeze(1), size=[height, width], mode='nearest').squeeze(1)
        self.core_model.update_prototypes(cls_attn, cls_attn_label, is_cls=True)
        self.core_model.update_prototypes(img_attn, img_attn_label, is_cls=False)

        consistency_weight = self.get_current_consistency_weight(iter_num // max(1, int(self.args.max_iterations / self.args.consistency_rampup)))
        sup_loss = (output_cls_sup_loss + output_img_sup_loss) / 2 + sim_sup_loss
        cons_loss = (output_cls_unsup_loss + output_img_unsup_loss) / 2 + sim_unsup_loss

        loss = text_loss + sup_loss + consistency_weight * cons_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']

        logger.info('iteration %d : '
                     '  loss : %f'
                     '  sup_loss : %f'
                     '  text_loss : %f'

                     '  cons_loss : %f'
                     '  lr_ : %10f'

                     % (iter_num,
                        loss,
                        sup_loss,
                        text_loss,

                        consistency_weight * cons_loss,
                        current_lr
                        ))


    @torch.no_grad()
    def val_list(self, val_loader, snapshot_path, logger, iter_num):
        self.model.eval()

        avg_dice = 0.0

        for i_batch, sampled_batch in enumerate(val_loader):
            val_image, val_label, class_text = sampled_batch['image'].to(self.args.device), sampled_batch['label'].to(self.args.device), sampled_batch['class_text'].squeeze(1).to(self.args.device)

            metric_list = test_single_volume(self.args, val_image, val_label, class_text, self.core_model)
            metric_list = np.array(metric_list).astype("float32")
            metric_list = np.mean(metric_list, axis=0)

            avg_dice += metric_list[0]

        avg_dice = avg_dice / len(val_loader)

        logger.info('iteration %d : '
                     '  avg_dice : %f '
                     % (iter_num,  avg_dice))

        if avg_dice > self.best_performance:
            self.best_performance = avg_dice
            save_best = os.path.join(snapshot_path, 'best_model_' + str(iter_num) + '.pth')
            torch.save(self.model.state_dict(), save_best)

        self.model.train()
