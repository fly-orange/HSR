import numpy as np
import pandas as pd
import torch
from utils.utils import *


def seq2instance(data, P,Q, interval):
    num_step, dims = data.shape
    num_sample = int(( num_step - P -Q + 1 )/interval)
    x = torch.zeros(num_sample, P, dims)
    y = torch.zeros(num_sample,Q, dims)
    for i in range(num_sample):
        x[i] = data[i*interval : i*interval + P]
        y[i] = data[i*interval + P : i*interval + P +Q]
    return x, y #(B, T, N)

def load_mp(df, args):
    # Traffic
    Traffic = torch.from_numpy(df.values)
    # train/val/test 
    num_step = df.shape[0]
    # mean = df.mean().values
    # std = df.std().values

    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]  #(T, D)
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    # X, Y 
    trainX, trainY = seq2instance(train, args.P, args.Q, args.interval) # (B, T, N, D)
    valX, valY = seq2instance(val,args.P, args.Q, args.interval)
    testX, testY = seq2instance(test,args.P, args.Q, args.Q)
  
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 2, 2))  #(B, T, N, D)
    valX = valX.reshape((valX.shape[0], valX.shape[1], 2, 2))
    testX = testX.reshape((testX.shape[0], testX.shape[1], 2, 2))

    trainY = trainY.reshape((trainY.shape[0], trainY.shape[1], 2, 2))[...,0]
    valY = valY.reshape((valY.shape[0], valY.shape[1], 2, 2))[...,0]
    testY = testY.reshape((testY.shape[0], testY.shape[1], 2, 2))[...,0]

    # normalization 
    mean, std = torch.mean(trainX[...,0]), torch.std(trainX[...,0])
    mean_d, std_d = torch.mean(trainX[...,1]), torch.std(trainX[...,1])

    trainX[...,0] = (trainX[...,0] - mean) / std  #(B, T, D)
    valX[...,0] = (valX[...,0] - mean) / std
    testX[...,0] = (testX[...,0] - mean) / std
    
    trainX[...,1] = (trainX[...,1] - mean_d) / std_d  #(B, T, D)
    valX[...,1] = (valX[...,1] - mean_d) / std_d
    testX[...,1] = (testX[...,1] - mean_d) / std_d


    # temporal embedding 
    time = pd.DatetimeIndex(df.index)
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
    #     timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
    #                 // time.freq.delta.total_seconds()
    #     timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    timeofday = (time.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))        
    time = torch.cat((dayofweek, timeofday), -1)
    # train/val/test
    train = time[: train_steps]
    val = time[train_steps : train_steps + val_steps]
    test = time[-test_steps :]
    # shape = (num_sample, ['P'] +['Q'], 2)
    trainTE = seq2instance(train,args.P, args.Q, args.interval)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, args.P, args.Q,args.interval)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, args.P,args.Q, args.Q)
    testTE = torch.cat(testTE, 1).type(torch.int32)
    
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
             mean, std)


def loadData(df,args):
    # Traffic
    Traffic = torch.from_numpy(df.values)
    # train/val/test 
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    # X, Y 
    trainX, trainY = seq2instance(train, args.P, args.Q, args.interval)
    valX, valY = seq2instance(val,args.P, args.Q, args.interval)
    testX, testY = seq2instance(test,args.P, args.Q, args.Q)
    # normalization
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std
    
    # temporal embedding 
    time = pd.DatetimeIndex(df.index)
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
#     timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
#                 // time.freq.delta.total_seconds()
#     timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    timeofday = (time.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))        
    time = torch.cat((dayofweek, timeofday), -1)
    # train/val/test
    train = time[: train_steps]
    val = time[train_steps : train_steps + val_steps]
    test = time[-test_steps :]
    # shape = (num_sample, ['P'] +['Q'], 2)
    trainTE = seq2instance(train,args.P, args.Q, args.interval)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, args.P, args.Q,args.interval)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, args.P,args.Q, args.Q)
    testTE = torch.cat(testTE, 1).type(torch.int32)
    
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
             mean, std)

def loadData_cl(total_df,args):
    # Traffic
    # train/val/test
    train_list = []
    val_list = []
    test_list = []
    total_len = len(total_df)
    task_len = int(total_len/args.num_task)
    for i in range(args.num_task):
        df = total_df.iloc[task_len*i:task_len*i+task_len]
        trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std = loadData(df, args)
        train_list.append([trainX, trainTE, trainY, mean, std])
        val_list.append([valX, valTE, valY, mean, std])
        test_list.append([testX, testTE, testY, mean, std])
    
    return train_list, val_list, test_list

def mergeData(train_list, val_list, test_list):
    trainX = torch.cat([X*std+mean for (X,TE,Y,mean,std) in train_list], axis =0)
    valX = torch.cat([X*std+mean for (X,TE,Y,mean,std) in val_list], axis =0)
    testX = torch.cat([X*std+mean for (X,TE,Y,mean,std) in test_list], axis =0)
    trainTE = torch.cat([TE for (X,TE,Y,mean,std) in train_list], axis =0)
    valTE = torch.cat([TE for (X,TE,Y,mean,std) in val_list], axis =0)
    testTE = torch.cat([TE for (X,TE,Y,mean,std) in test_list], axis =0)
    trainY = torch.cat([Y for (X,TE,Y,mean,std) in train_list], axis =0)
    valY = torch.cat([Y for (X,TE,Y,mean,std) in val_list], axis =0)
    testY = torch.cat([Y for (X,TE,Y,mean,std) in test_list], axis =0)
    
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    train_data = (trainX, trainTE, trainY, mean, std)
    val_data = (valX, valTE, valY, mean, std)
    test_data = (testX, testTE, testY, mean, std)

    return train_data, val_data, test_data


def load_Data(args):
    # Traffic
    df = pd.read_csv(args['data_file'],index_col=0)
    Traffic = torch.from_numpy(df.values)
    # train/val/test 
    num_step = df.shape[0]
    train_steps = round(args['train_ratio'] * num_step)
    test_steps = round(args['test_ratio'] * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    # X, Y 
    trainX, trainY = seq2instance(train, args['P'], args['Q'],1)
    valX, valY = seq2instance(val,args['P'], args['Q'],1)
    testX, testY = seq2instance(test,args['P'], args['Q'], 1)
    # normalization
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std
    
    # temporal embedding 
    time = pd.DatetimeIndex(df.index)
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
#     timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
#                 // time.freq.delta.total_seconds()
#     timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    timeofday = (time.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))        
    time = torch.cat((dayofweek, timeofday), -1)
    # train/val/test
    train = time[: train_steps]
    val = time[train_steps : train_steps + val_steps]
    test = time[-test_steps :]
    # shape = (num_sample, ['P'] +['Q'], 2)
    trainTE = seq2instance(train,args['P'], args['Q'],1)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, args['P'], args['Q'],1)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, args['P'],args['Q'],1)
    testTE = torch.cat(testTE, 1).type(torch.int32)
    
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
             mean, std)