import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
import datetime
from sklearn.metrics import *
import json
from pathlib import Path
import datetime as dt
from model.gt import *

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def save_json(json_obj, json_path, append_if_exists=False,
              overwrite_if_exists=True, unique_fn_if_exists=False):
    """Saves a json file

    Arguments:
        json_obj: json, json object
        json_path: Path, path including the file name where the json object
            should be saved to
        append_if_exists: bool, append to the existing json file with the same
            name if it exists (keep the json structure intact)
        overwrite_if_exists: bool, xor with append, overwrites any existing
            target file
        unique_fn_if_exsists: bool, appends the current date and time to the
            file name if the target file exists already.
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)

    if overwrite_if_exists:
        append_if_exists = False
        unique_fn_if_exists = False

    if unique_fn_if_exists:
        overwrite_if_exists = False
        append_if_exists = False
        if json_path.exists():
            time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
            json_path = json_path.parents[0] / f'{str(json_path.stem)}_{time}'\
                                               f'{str(json_path.suffix)}'

    if overwrite_if_exists:
        append_if_exists = False
        with open(json_path, 'w+') as fout:
            json.dump(json_obj, fout, indent=2)
        return

    if append_if_exists:
        if json_path.exists():
            with open(json_path, 'r') as fin:
                read_file = json.load(fin)
            read_file.update(json_obj)
            with open(json_path, 'w+') as fout:
                json.dump(read_file, fout, indent=2)
            return

    with open(json_path, 'w+') as fout:
        json.dump(json_obj, fout, indent=2) 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count
# metric
def metrics(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mae = torch.abs(pred - label).type(torch.float32)
    rmse = mae ** 2
    mape = mae / (label+1e-4)
    mae = torch.mean(mae)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))
    mape = mape * mask
    mape = torch.mean(mape)    
    return rmse.cpu().numpy(), mae.cpu().numpy(),  mape.cpu().numpy()

def cl_metrics(end_task_loss_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.
    Params:
    end_task_acc_arr:       (num_run, num_task, num_task)
    task_ids:               (list or tuple) Task ids to keep track of
    return:                 (avg_end_acc, forgetting, avg_acc_task)
    """

    # compute average test accuracy and CI
    end_loss = end_task_loss_arr[-1]                         # shape: (num_run, num_task)
    avg_end_loss = np.mean(end_loss)

    # compute forgetting
    best_loss = np.min(end_task_loss_arr, axis=1)
    final_forgets = end_loss - best_loss
    avg_fgt = np.mean(final_forgets)

    return avg_end_loss, avg_fgt

def alarm_metric(pred, label, threshold):
    pred = pred.reshape(-1).cpu().numpy()
    label = label.reshape(-1).cpu().numpy()
    pred = (pred > threshold)
    label = (label > threshold)
    precision = precision_score(label, pred)
    recall = recall_score(label, pred)
    f1 = f1_score(label, pred)
    return precision, recall, f1

def weighted_focal_mse_loss(inputs, targets, activate='sigmoid', beta=7, gamma=1, weights=None):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def setup_model(params):
    if params.model == 'GT': # inputs_size = (batch_size, 28*28)
        # print('mlp model loaded')
        # return MLP(784, params.n_class, params.mlp_hidden) 
        print('GraphTrans loaded')
        return GraphTrans(params.p_length, params.D, params.K, params.bn_decay, params.device)
    if params.model == 'GMAN':
        print('GMAN loaded')
        return ResNet18(params.n_class, nf=64, bias=True)

def setup_criterion(params):
    if params.criterion == 'MSE':
        criterion = torch.nn.MSELoss()
    if params.criterion == 'MAE':
        criterion = torch.nn.L1Loss()
        # criterion = F.cross_entropy()
    return criterion

def setup_optimizer(params, model):
    if params.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                          lr = params.lr,
                          weight_decay = params.weight_decay,
                          momentum = params.momentum)
    elif params.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                           lr = params.lr,
                           weight_decay = params.weight_decay) 
    else:
        raise Exception('Wrong optimizer name')
    return optimizer      

def setup_scheduler(params, optimizer):
    if params.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = params.decay_epoch, gamma = 0.9 )
    elif params.optimizer == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR()
    elif params.optimizer == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR()
    else:
        raise Exception('Wrong scheduler name')
    return scheduler 