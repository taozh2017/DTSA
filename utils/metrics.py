import torch
import numpy as np
from medpy import metric

def eval(y_true, y_pred, thr=0.5,):
    all_res = []
    for c in range(1):
        res = []
        c_true = y_true.to(torch.float32).squeeze(0)[c].cpu().detach().numpy()
        c_pred = (y_pred > thr).to(torch.float32).squeeze(0)[c].cpu().detach().numpy()
        if c_pred.sum() == 0:
            res = [0, 0, 0]
        else:
            res.append(metric.binary.dc(c_pred, c_true))
            res.append(metric.binary.jc(c_pred, c_true))
            res.append(metric.binary.sensitivity(c_pred, c_true))

        all_res.append(res)
    all_res = np.mean(np.array(all_res), axis=0)
    return all_res


