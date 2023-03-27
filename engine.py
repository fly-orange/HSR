'''Train a model'''
'''Control Training, Evaluation and Testing'''
import logging
import numpy as np
import math
import torch
import torch.nn.functional as F
import copy
from utils.utils import *
from utils.data import *

class BasicLearner():
    def __init__(self, params):
        self.model = setup_model(params).to(params.device)
        if params.pretrain:
            self.model.load_state_dict(torch.load(params.pretrain_file))
            logging.info(f'model has been loaded from {pretrain_file} ')
        self.criterion = setup_criterion(params)
        self.optimizer = setup_optimizer(params, self.model)
        self.scheduler = setup_scheduler(params, self.optimizer)

        self.params = params

    def train(self, trainloader):
        ## load data
        trainX, trainTE, trainY, mean, std = trainloader
        num_train = trainX.shape[0]
        train_num_batch = math.ceil(num_train / self.params.batch_size)
        permutation = torch.randperm(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        
        self.model.train()
        loss_train = AverageMeter()

        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * self.params.batch_size
            end_idx = min(num_train, (batch_idx + 1) * self.params.batch_size)
            X = trainX[start_idx: end_idx].to(self.params.device)
            TE = trainTE[start_idx: end_idx].to(self.params.device)
            label = trainY[start_idx: end_idx].to(self.params.device)
            self.optimizer.zero_grad()
            pred = self.model(X, TE)
            pred = pred * std + mean
            loss_batch = self.criterion(pred, label)
            loss_train.update(float(loss_batch), (end_idx - start_idx))
            loss_batch.backward()
            self.optimizer.step()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # if (batch_idx+1) % 5 == 0:
            #     print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}')
            del X, TE, label, pred, loss_batch

        return loss_train.avg()


    def eval(self, dataloader):
        evalX, evalTE, evalY, mean, std = dataloader
        num_eval = evalX.shape[0]
        eval_num_batch = math.ceil(num_eval / self.params.batch_size)
        self.model.eval()
        loss_eval = AverageMeter()

        with torch.no_grad():
            for batch_idx in range(eval_num_batch):
                start_idx = batch_idx * self.params.batch_size
                end_idx = min(num_eval, (batch_idx + 1) * self.params.batch_size)
                X = evalX[start_idx: end_idx].to(self.params.device)
                TE = evalTE[start_idx: end_idx].to(self.params.device)
                label = evalY[start_idx: end_idx].to(self.params.device)
                pred = self.model(X, TE)
                pred = pred * std + mean
                loss_batch = self.criterion(pred, label)
                loss_eval.update(float(loss_batch), (end_idx - start_idx))
                del X, TE, label, pred, loss_batch

        return loss_eval.avg()
    
    ## tester
    def test(self, dataloader):
        
        testX, testTE, testY, mean, std = dataloader
        num_test = testX.shape[0]
        test_num_batch = math.ceil(num_test / self.params.batch_size)
        self.model.eval()
        testPred = []
        start_test = time.time()
        self.model.eval()
        with torch.no_grad():
            for batch_idx in range(test_num_batch):
                start_idx = batch_idx * self.params.batch_size
                end_idx = min(num_test, (batch_idx + 1) * self.params.batch_size)
                X = testX[start_idx: end_idx].to(self.params.device)
                TE = testTE[start_idx: end_idx].to(self.params.device)
                pred_batch = self.model(X, TE)
                testPred.append(pred_batch.detach().cpu())
                del X, TE, pred_batch
        testPred = torch.cat(testPred, axis=0)
        testPred = testPred* std + mean

        rmse, mae, mape = metrics(testPred, testY)
        logging.info('                MAE\t\tRMSE\t\tMAPE')
        logging.info('test             %.2f\t\t%.2f\t\t%.2f%%' %
                (mae, rmse, mape * 100))
        
        return rmse, mae, mape



'''Continual Learner'''
class ERLearner(BasicLearner):
    def __init__(self, params):
        super(ERLearner,self).__init__(params)
        self.mem_size = params.mem_size
        # self.mem_iter = params.mem_iter
        # self.buffer = Buffer(self.model, params)
        self.buffer = [] # Dataset

    def train(self, trainloader):
        
        trainX, trainTE, trainY, mean, std = trainloader
        num_train = trainX.shape[0]
        train_num_batch = math.ceil(num_train / self.params.batch_size)
        permutation = torch.randperm(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        
        self.model.train()
        loss_train = AverageMeter()
        
        for batch_idx in range(train_num_batch):
            ## train data
            start_idx = batch_idx * self.params.batch_size
            end_idx = min(num_train, (batch_idx + 1) * self.params.batch_size)
            X = trainX[start_idx: end_idx].to(self.params.device)
            TE = trainTE[start_idx: end_idx].to(self.params.device)
            label = trainY[start_idx: end_idx].to(self.params.device)
            self.optimizer.zero_grad()
            pred = self.model(X, TE)
            pred = pred * std + mean
            loss_batch = self.criterion(pred, label)
            loss_train.update(float(loss_batch), (end_idx - start_idx))
            loss_batch.backward()

            ## batch data
            if len(self.buffer):
                # mem_size = int(self.params.batch_size/len(self.buffer))
                # for i in range(len(self.buffer)):
                i = np.random.randint(len(self.buffer))
                bufferX, bufferTE, bufferY, buffer_mean, buffer_std = self.buffer[i]
                index = np.random.choice(len(bufferX), self.params.batch_size, replace=False)
                memX = bufferX[index].to(self.params.device)
                memTE = bufferTE[index].to(self.params.device)
                mem_label = bufferY[index].to(self.params.device)
                mem_pred = self.model(memX, memTE)
                mem_pred = mem_pred * buffer_std + buffer_mean
                loss_mem_batch = self.criterion(mem_pred, mem_label)
                loss_mem_batch.backward()
            # else: 
            #     loss_batch.backward()

            self.optimizer.step()
            del X, TE, label, pred, loss_batch

        return loss_train.avg()

    def buffer_update(self, trainloader):
        trainX, trainTE, trainY, mean, std = trainloader
        if self.params.update == 'random':
            index = np.random.choice(len(trainX), self.params.mem_size, replace=False)
            bufferX = trainX[index]
            bufferTE = trainTE[index]
            bufferY = trainY[index]

        if self.params.update in ['loss_h', 'loss_l', 'loss_lh']:
            trainX, trainTE, trainY, mean, std = trainloader
            num_train = trainX.shape[0]
            train_num_batch = math.ceil(num_train / self.params.batch_size)
            loss_criterion = torch.nn.MSELoss(reduction= 'none')
            loss_all = []
            self.model.eval()
            with torch.no_grad():
                for batch_idx in range(train_num_batch):
                    start_idx = batch_idx * self.params.batch_size
                    end_idx = min(num_train, (batch_idx + 1) * self.params.batch_size)
                    X = trainX[start_idx:end_idx].to(self.params.device)
                    TE = trainTE[start_idx:end_idx].to(self.params.device)
                    label = trainY[start_idx:end_idx].to(self.params.device)
                    pred = self.model(X, TE)
                    pred = pred * std + mean
                    loss_batch = loss_criterion(pred, label).mean(dim=-1).mean(dim=-1)
                    loss_all.append(loss_batch.detach().cpu())
            
            loss_all = torch.cat(loss_all,axis = 0)
            all_index = torch.argsort(loss_all)
            if self.params.update == 'loss_l':
                index = all_index[:self.params.mem_size].numpy()
            elif self.params.update == 'loss_h':
                index = all_index[-self.params.mem_size:].numpy()
            elif self.params.update == 'loss_lh':
                index = np.concatenate((all_index[:self.params.mem_size//2].numpy(),all_index[-self.params.mem_size//2:].numpy()),axis=0)
            bufferX = trainX[index]
            bufferTE = trainTE[index]
            bufferY = trainY[index]
        
        self.buffer.append([bufferX, bufferTE, bufferY, mean, std])

