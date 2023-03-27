import os
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchsummary import *
from modules import *
import time
import datetime
from utils import *
import math


parser = argparse.ArgumentParser()
# data args
parser.add_argument('--dataset', type = str, default = 'tj')
parser.add_argument('--interval', type = str, default = '5m')
parser.add_argument('--train_ratio', type = float, default = 0.7)
parser.add_argument('--test_ratio', type = float, default = 0.1)
parser.add_argument('--P', type = int, default = 64)
parser.add_argument('--Q', type = int, default = 16)

# model args
parser.add_argument('--p_length', type = int, default = 2)
parser.add_argument('--D', type = int, default = 64)
parser.add_argument('--K', type = int, default = 8)
parser.add_argument('--bn_decay', type = float, default = 0.1)

# training args
parser.add_argument('--max_epochs', type = int, default = 30)
parser.add_argument('--patience', type = int, default = 9)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--LR', type = float, default = 0.01)
parser.add_argument('--decay_epoch', type = int, default = 10)
parser.add_argument('--gamma', type = float, default = 0.9)
parser.add_argument('--gpu', type = bool, default = True )
parser.add_argument('--loss', type = str, default = 'MSE')

args = parser.parse_args()

#============================
#main
#============================
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(f"Using device {device}")

log_file = 'Wind/log(tjWind)'
log = open(log_file, 'w')

args.data_file = './Wind/{}_wind_{}.csv'.format(args.dataset, args.interval)

args.p_length = p_length
log_string(log, f'Performance on {dataset}_wind by model with p_length of {args.p_length}')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std) = loadData(args)
log_string(log,'Data have been loaded totally')

model = GraphTrans(args.p_length, args.D, args.K, args.bn_decay, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                    step_size=args.decay_epoch,
                                    gamma=args.gamma)
if args.loss == 'MSE':
    loss_criterion = nn.MSELoss().to(device)
elif args.loss == 'L1':
    loss_criterion = nn.L1Loss().to(device)
log_string(log,'Model have been loaded totally')

model_file = 'result/model/GT_{}{}_p{}.pkl'.format(args.dataset, args.interval, args.p_length)

batch_size = args.batch_size
max_epochs = args.max_epochs
log_string(log, '**** training model ****')
num_val = valX.shape[0]
num_train = trainX.shape[0]
num_test = testX.shape[0]
train_num_batch = math.ceil(num_train / batch_size)
val_num_batch = math.ceil(num_val / batch_size)
test_num_batch = math.ceil(num_test / batch_size)
wait = 0
val_loss_min = float('inf')
best_model_wts = None
train_total_loss = []
val_total_loss = []
# shuffle
# model = torch.load(model_file)
for epoch in range(args.max_epochs):    
    if wait >= args.patience:
        log_string(log, f'early stop at epoch: {epoch:04d}')
        break
    permutation = torch.randperm(num_train)
    trainX = trainX[permutation]
    trainTE = trainTE[permutation]
    trainY = trainY[permutation]
    # train
    start_train = time.time()
    model.train()
    train_loss = 0
    for batch_idx in range(train_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_train, (batch_idx + 1) * batch_size)
        X = trainX[start_idx: end_idx].to(device)
        TE = trainTE[start_idx: end_idx].to(device)
        label = trainY[start_idx: end_idx].to(device)
        optimizer.zero_grad()
        pred = model(X, TE)
        pred = pred * std + mean
        loss_batch = loss_criterion(pred, label)
        train_loss += float(loss_batch) * (end_idx - start_idx)
        loss_batch.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (batch_idx+1) % 5 == 0:
            print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}')
        del X, TE, label, pred, loss_batch
    train_loss /= num_train
    train_total_loss.append(train_loss)
    end_train = time.time()

    # val loss
    start_val = time.time()
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx in range(val_num_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(num_val, (batch_idx + 1) * batch_size)
            X = valX[start_idx: end_idx].to(device)
            TE = valTE[start_idx: end_idx].to(device)
            label = valY[start_idx: end_idx].to(device)
            pred = model(X, TE)
            pred = pred * std + mean
            loss_batch = loss_criterion(pred, label)
            val_loss += loss_batch * (end_idx - start_idx)
            del X, TE, label, pred, loss_batch
    val_loss /= num_val
    val_total_loss.append(val_loss)
    end_val = time.time()
    log_string(
        log,
        '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
        max_epochs, end_train - start_train, end_val - start_val))
    log_string(
        log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
    if val_loss <= val_loss_min:
        log_string(
            log,
            f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {model_file}')
        wait = 0
        val_loss_min = val_loss
        best_model_wts = model.state_dict()
        torch.save(model, model_file)
    else:
        wait += 1
    scheduler.step()
model.load_state_dict(best_model_wts)
torch.save(model, model_file)
log_string(log, f'Training and validation are completed, and model has been stored as {model_file}')



model = torch.load(model_file)
threshold = 15
if torch.cuda.is_available():
    torch.cuda.empty_cache()
log_file = 'Wind/log(bjWind)'
log = open(log_file, 'w')
model = torch.load(model_file)
num_test = testX.shape[0]
test_num_batch = math.ceil(num_test / batch_size)
testPred = []
start_test = time.time()
for batch_idx in range(test_num_batch):
    start_idx = batch_idx * batch_size
    end_idx = min(num_test, (batch_idx + 1) * batch_size)
    X = testX[start_idx: end_idx].to(device)
    TE = testTE[start_idx: end_idx].to(device)
    pred_batch = model(X, TE)
    testPred.append(pred_batch.detach().clone())
    del X, TE, pred_batch
testPred = torch.cat(testPred, axis=0)
testPred = testPred* std + mean
interval = [3,6,12]
# max_length = int(math.log(args.Q,2))
for i in range(3):
    if i!=0:
        start_idx = pow(2,i+1)
    else:
        start_idx = 0
    end_idx = pow(2,i+2)
    pred = testPred[:,start_idx:end_idx]
    label = testY[:,start_idx:end_idx]
    test_mae, test_rmse, test_mape = metric(pred, label.to(device))
    total_MAE.append(test_mae)
    total_RMSE.append(test_rmse)
    total_MAPE.append(test_mape)
    log_string(log, f'From step {start_idx} to step {end_idx}')
    log_string(log, f' Prediction performance:   MAE: {test_mae}, RMSE: {test_rmse}, MAPE: {test_mape}')
test_mae, test_rmse, test_mape = metric(testPred, testY.to(device))
log_string(log, 'During all steps')
log_string(log, f' Prediction performance:   MAE: {test_mae}, RMSE: {test_rmse}, MAPE: {test_mape}')
precision, recall, f1 = alarm_metric(testPred,testY,threshold)
log_string(log,  f' Alarm performance:   precision score: {precision}, recall score: {recall}, F1 score: {f1}')
