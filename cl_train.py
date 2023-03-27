import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import pickle
import argparse
import json
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
from torchsummary import *
from model.gt import *
import time
import datetime
from utils.data import *
from utils.utils import *
from engine import *


parser = argparse.ArgumentParser()
# data hp
parser.add_argument('--dataset', type = str, default = 'tj')
parser.add_argument('--rate', type = str, default = '5m')
parser.add_argument('--train_ratio', type = float, default = 0.7)
parser.add_argument('--test_ratio', type = float, default = 0.1)
parser.add_argument('--P', type = int, default = 64)
parser.add_argument('--Q', type = int, default = 16)
parser.add_argument('--interval', type = int, default = 1)

# model hp
parser.add_argument('--model', type = str, default = 'GT', choices=['GT', 'GMAN'])
parser.add_argument('--criterion', type = str, default = 'MSE', choices=['MSE', 'MAE'])
parser.add_argument('--optimizer', type = str, default = 'Adam', choices=['Adam', 'SGD'])
parser.add_argument('--scheduler', type = str, default = 'step', choices=['step', 'cos', 'exp'])

parser.add_argument('--p_length', type = int, default = 4)
parser.add_argument('--D', type = int, default = 64)
parser.add_argument('--K', type = int, default = 8)
parser.add_argument('--bn_decay', type = float, default = 0.1)

# training hp
parser.add_argument('--max_epochs', type = int, default = 12)
parser.add_argument('--patience', type = int, default = 8)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--weight_decay', type = float, default = 0.001) 
parser.add_argument('--lr_decay', action = 'store_true')
parser.add_argument('--decay_epoch', type = int, default = 10)
parser.add_argument('--gamma', type = float, default = 0.9)
parser.add_argument('--gpu', type = bool, default = True )
parser.add_argument('--pretrain', action = 'store_true')

# continual learning hp
parser.add_argument('--num_task', type = int, default = 36)
parser.add_argument('--cl_type', type = str, default = 'er', choices=['retrain', 'ft', 'er'])
parser.add_argument('--update', type = str, default = 'loss_lh', choices=['random', 'loss_l', 'loss_h', 'loss_lh'])
parser.add_argument('--mem_size', type = int, default = 500)


args = parser.parse_args()
args.device = 'cuda:0'
print(f"Using device {args.device}")

log_file = 'data/Wind/log(bjWind)'
log = open(log_file, 'w')

'''Load Data'''
def DataLoad(args):
    data_file = 'data/Wind/{}_wind_{}.csv'.format(args.dataset,args.rate)
    df = pd.read_csv(data_file,index_col=0)
    train_list, val_list, test_list  = loadData_cl(df,args)
    log_string(log,'Data have been loaded totally')

    return (train_list, val_list, test_list)


def cl_train(args, data):
    '''Save Folder'''
    model_save_path = f'result/CL/{args.dataset}_{args.rate}/model/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    result_save_path = f'result/CL/{args.dataset}_{args.rate}/result/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    model_file = model_save_path + f'{args.model}_{args.p_length}_{args.cl_type}_{args.update}_{args.num_task}.pth'
    result_file = result_save_path + f'{args.model}_{args.p_length}_{args.cl_type}_{args.update}_{args.num_task}.json'
    
    train_list, val_list, test_list = data

    '''Task Loop'''
    num_task = args.num_task
    RMSE, MAE, MAPE = np.zeros((num_task,num_task)), np.zeros((num_task,num_task)), np.zeros((num_task,num_task))
    if args.cl_type == 'ft':
        engine = BasicLearner(args)
    elif args.cl_type == 'er':
        engine = ERLearner(args)

    log_string(log, '**** start training model ****')
    for i in range(num_task):
        start_task = time.time()
        log_string(log, f'Task {i+1} is handled')
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
            # train 
            start_train = time.time()
            train_loss = engine.train(train_list[i])
            train_total_loss.append(train_loss)
            end_train = time.time()

            # evaluation
            start_val = time.time()
            val_loss = engine.eval(val_list[i])
            val_total_loss.append(val_loss)
            end_val = time.time()

            log_string(
                log,
                '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
                (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
                args.max_epochs, end_train - start_train, end_val - start_val))
            log_string(
                log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
            if val_loss <= val_loss_min:
                log_string(
                    log,
                    f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {model_file}')
                wait = 0
                val_loss_min = val_loss
                best_model_wts = engine.model.state_dict()
                torch.save(best_model_wts, model_file)
            else:
                wait += 1
            
            if epoch and epoch % args.decay_epoch ==0:
                if args.lr_decay:
                    engine.scheduler.step()

        engine.model.load_state_dict(best_model_wts)
        log_string(log, f'Training and validation are completed, and model has been stored as {model_file}')

        ## test
        start_test = time.time()
        for j in range(num_task):
            rmse, mae, mape = engine.test(test_list[j])
            RMSE[i,j], MAE[i,j], MAPE[i,j] = rmse, mae, mape
        end_test = time.time()
        end_task = time.time()
        log_string(log,f'Task {i+1} finished. training time: {end_test-start_test:.1f}s, total time: {end_task-start_task:.1f}s')
        if args.cl_type == 'er':
            engine.buffer_update(train_list[i])

    results = [RMSE.tolist(),MAE.tolist(),MAPE.tolist()]
    save_json(results, result_file)
 

def retrain(args, data):
    '''Save Folder'''
    model_save_path = f'result/CL/{args.dataset}_{args.rate}/model/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    result_save_path = f'result/CL/{args.dataset}_{args.rate}/result/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    model_file = model_save_path + f'{args.model}_{args.p_length}_{args.cl_type}_{args.update}.pth'
    result_file = result_save_path + f'{args.model}_{args.p_length}_{args.cl_type}_{args.update}.json'
    
    train_list, val_list, test_list = data
    train_data, val_data, test_data = mergeData(train_list, val_list, test_list)
    engine = BasicLearner(args)
    
    '''retrain'''
    wait = 0
    val_loss_min = float('inf')
    best_model_wts = None
    train_total_loss = []
    val_total_loss = []
    for epoch in range(args.max_epochs):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        # train 
        start_train = time.time()   
        loss_train = engine.train(train_data)
        # train_loss = AverageMeter()
        # for i in range(len(train_list)):
        #     loss_train = engine.train(train_list[i])
        #     train_loss.update(loss_train, len(train_list[i][0]))
        #     loss_train = train_loss.avg()
        train_total_loss.append(loss_train)
        end_train = time.time()
       
        # eval
        start_val = time.time()
        # loss_val = engine.train(train_data)
        val_loss = AverageMeter()
        for i in range(len(val_list)):
            loss_val = engine.train(val_list[i])
            val_loss.update(loss_train, len(val_list[i][0]))
            loss_val = val_loss.avg()
        val_total_loss.append(loss_val)
        end_val = time.time()
        
        log_string(log, '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
            args.max_epochs, end_train - start_train, end_val - start_val))
        log_string(
            log, f'train loss: {loss_train:.4f}, val_loss: {loss_val:.4f}')
        if loss_val <= val_loss_min:
            log_string(
            log, f'val loss decrease from {val_loss_min:.4f} to {loss_val:.4f}, saving model to {model_file}')
            wait = 0
            val_loss_min = loss_val
            best_model_wts = engine.model.state_dict()
            torch.save(best_model_wts, model_file)
        else:
            wait += 1
        
        if epoch and epoch % args.decay_epoch ==0:
            if args.lr_decay:
                engine.scheduler.step()

    engine.model.load_state_dict(best_model_wts)
    log_string(log, f'Training and validation are completed, and model has been stored as {model_file}')
    
    '''test'''
    RMSE, MAE, MAPE =[], [], []
    start_test = time.time()
    # RMSE, MAE, MAPE = engine.test(test_data)
    for j in range(len(test_list)):
        rmse, mae, mape = engine.test(test_list[j])
        RMSE.append(rmse.tolist())
        MAE.append(mae.tolist())
        MAPE.append(mape.tolist())
    end_test = time.time()
    log_string(log,f'Total training time: {end_test-start_test:.1f}s')
    log_string(log,f'RMSE: {RMSE}, MAE: {MAE}, MAPE: {MAPE}')
    results = [RMSE, MAE, MAPE]
    save_json(results, result_file)


if __name__ =='__main__': 
    t1 = time.time()
    data = DataLoad(args)
    if args.cl_type =='retrain':
        retrain(args,data)
    else:
        cl_train(args, data)
    t2 = time.time()
    log_string(log, f'Finish Training, total time {t2-t2:4f}')


    ## test
    # train_list, val_list, test_list = data
    # engine = BasicLearner(args)
    # model_save_path = f'result/CL/{args.dataset}_{args.rate}/model/'
    # model_file = model_save_path + f'{args.model}_{args.p_length}_{args.cl_type}_{args.update}_{args.num_task}.pth'
    # engine.model.load_state_dict(torch.load(model_file))
    # RMSE, MAE, MAPE = [], [], []
    # for j in range(args.num_task):
    #     rmse, mae, mape = engine.test(test_list[j])
    #     RMSE.append(rmse)
    #     MAE.append(mae)
    #     MAPE.append(mape)
    # print(np.mean(RMSE),np.mean(MAE),np.mean(MAPE) )