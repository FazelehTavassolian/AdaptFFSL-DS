import copy
from abc import ABC
from collections import OrderedDict
import time
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Optimizer
from optim import mk_optim
from optim.fedOptim import *

# Loss
from loss import mk_loss


class LocalUpdate():
    def __init__(self, name, args, model, device, core, freq, bw, **kwargs):
        self.model = copy.deepcopy(model)
        #print('init local')
        #print(next(self.model.parameters()).is_cuda)
        self.args = args
        self.dataset = kwargs['dataset']
        self.device = device
        self.name = name
        self.train_samples = len(self.dataset)
        self.train_time_cost = {'num_rounds': 0.0, 'total_cost': 0.0, 'norm_cost': 0.0}
        self.cores = core
        self.freq = freq
        self.bw = bw

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.args.batch
        return DataLoader(self.dataset, batch_size, drop_last=True, shuffle=True)

    @staticmethod
    def clone_model(model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def set_parameter(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    @staticmethod
    def update_parameter(model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def train(self, epochs, **kwargs):
        str_rnd = kwargs['str_rnd']
        train_loader = self.load_train_data()
        self.model.train()
        start_time = time.time()
        ep_loss = []
        ep_acc = []
        

        for epoch in range(epochs):
            tr_loss = 0.0
            tr_acc = 0.0
            step = 0
            loader_iter = tqdm(train_loader, desc=self.name)
            str_ep = f"[{epoch + 1}/{self.args.epochs}]"
            for i, data in enumerate(loader_iter):
                str_st = f"[{i + 1}/{len(train_loader)}]"
                data = tuple(fr.to(self.device) for fr in data)
                y = data[-1].view(-1)
                y = F.one_hot(y, 5)
                y = y.float()

                self.optimizer.zero_grad()
                #print('\n', '*' * 20, next(self.model.parameters()).is_cuda, data.is_cuda, '\n', '*' * 20)
                output = self.model(data)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                tr_loss += loss.item()
                if kwargs['cnf']['model']['name'] == 'relationNet':
                    batch_acc = metrics.accuracy_score(np.argmax(y.detach().cpu().numpy(), axis=-1),
                                                       np.argmax(output.detach().cpu().numpy(), axis=-1))
                else:
                    batch_acc = metrics.accuracy_score(np.argmax(y.detach().cpu().numpy(), axis=-1),
                                                       np.argmax(output.detach().cpu().numpy(), axis=-1))
                tr_acc += batch_acc
                str_postfix = f'Round: {str_rnd} Epoch: {str_ep} Step: {str_st} Batch Acc: {round(batch_acc, 3)} Acc: {round(tr_acc / (i + 1), 3)} Batch Loss: {round(loss.item(), 3)} Loss: {round(tr_loss / (i + 1), 3)}'
                show = OrderedDict({f'[Train]': str_postfix})
                loader_iter.set_postfix(show)
                step = i
            ep_loss.append(tr_loss / (step + 1))
            ep_acc.append(tr_acc / (step + 1))
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['norm_cost'] = self.train_time_cost['total_cost'] / self.train_time_cost['num_rounds']

        return {
            'loss': ep_loss,
            'accuracy': ep_acc
        }


class LocalUpdateAvg(LocalUpdate, ABC):
    def __init__(self, name, args, model, device, core, freq, bw, **kwargs):
        super().__init__(name, args, model, device, core, freq, bw, **kwargs)
        self.loss = mk_loss(self.args)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)


class LocalUpdateProx(LocalUpdate, ABC):
    def __init__(self, name, args, model, device, core, freq, bw, **kwargs):
        super().__init__(name, args, model, device, core, freq, bw, **kwargs)
        self.global_params = copy.deepcopy(list(self.model.parameters()))
        self.loss = mk_loss(self.args)
        self.optimizer = PerturbedGradientDescent(self.model.parameters(), lr=self.args.lr, mu=self.args.opt_prox_mu)

    def train(self, epochs, **kwargs):
        str_rnd = kwargs['str_rnd']
        train_loader = self.load_train_data()
        self.model.train()
        start_time = time.time()
        ep_loss = []
        ep_acc = []
        self.model.to(self.device)
        for epoch in range(epochs):
            tr_loss = 0.0
            tr_acc = 0.0
            step = 0
            loader_iter = tqdm(train_loader, desc=self.name)
            str_ep = f"[{epoch + 1}/{self.args.epochs}]"
            for i, data in enumerate(loader_iter):
                str_st = f"[{i + 1}/{len(train_loader)}]"
                data = tuple(fr.to(self.device) for fr in data)
                y = data[-1].view(-1)
                # if kwargs['cnf']['model']['name'] == 'relationNet':
                y = F.one_hot(y, 5)
                y = y.float()
                self.optimizer.zero_grad()

                #print('\n', '*' * 20, next(self.model.parameters()).is_cuda, '\n', '*' * 20)
                output = self.model(data)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step(self.global_params, self.device)
                tr_loss += loss.item()
                if kwargs['cnf']['model']['name'] == 'relationNet':
                    batch_acc = metrics.accuracy_score(np.argmax(y.detach().cpu().numpy(), axis=-1),
                                                       np.argmax(output.detach().cpu().numpy(), axis=-1))
                else:
                    batch_acc = metrics.accuracy_score(np.argmax(y.detach().cpu().numpy(), axis=-1),
                                                       np.argmax(output.detach().cpu().numpy(), axis=-1))
                tr_acc += batch_acc
                str_postfix = f'Round: {str_rnd} Epoch: {str_ep} Step: {str_st} Batch Acc: {round(batch_acc, 3)} Acc: {round(tr_acc / (i + 1), 3)} Batch Loss: {round(loss.item(), 3)} Loss: {round(tr_loss / (i + 1), 3)}'
                show = OrderedDict({f'[Train]': str_postfix})
                loader_iter.set_postfix(show)
                step = i
            ep_loss.append(tr_loss / (step + 1))
            ep_acc.append(tr_acc / (step + 1))
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['norm_cost'] = self.train_time_cost['total_cost'] / self.train_time_cost['num_rounds']

        return {
            'loss': ep_loss,
            'accuracy': ep_acc
        }

    def set_parameter(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()


class LocalUpdatePerAvg(LocalUpdateProx, ABC):
    def __init__(self, name, args, model, device, **kwargs):
        super().__init__(name, args, model, device, **kwargs)
        self.loss = mk_loss(self.args)
        self.beta = self.args.lr
        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.args.lr)

    def train(self, epochs, **kwargs):
        str_rnd = kwargs['str_rnd']
        train_loader = self.load_train_data(batch_size=self.args.batch * 2)
        self.model.train()
        start_time = time.time()
        ep_loss = []
        ep_acc = []
        for epoch in range(epochs):
            tr_loss = 0.0
            tr_acc = 0.0
            step = 0
            loader_iter = tqdm(train_loader, desc=self.name)
            str_ep = f"[{epoch + 1}/{self.args.epochs}]"
            for i, data in enumerate(loader_iter):
                str_st = f"[{i + 1}/{len(train_loader)}]"
                temp_model = copy.deepcopy(list(self.model.parameters()))

                # Step 1
                data_prime = tuple(fr[:self.args.batch].to(self.device) for fr in data)
                y = data_prime[-1].view(-1)
                # if kwargs['cnf']['model']['name'] == 'relationNet':
                y = F.one_hot(y, 5)
                y = y.float()
                self.optimizer.zero_grad()
                output = self.model(data_prime)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

                # Step 2
                data_prime = tuple(fr[self.args.batch:].to(self.device) for fr in data)
                y = data_prime[-1].view(-1)
                if kwargs['cnf']['model']['name'] == 'relationNet':
                    y = F.one_hot(y, 5)
                    y = y.float()
                self.optimizer.zero_grad()
                output = self.model(data_prime)
                loss = self.loss(output, y)
                loss.backward()

                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                self.optimizer.step(self.beta)

                tr_loss += loss.item()
                if kwargs['cnf']['model']['name'] == 'relationNet':
                    batch_acc = metrics.accuracy_score(np.argmax(y.detach().cpu().numpy(), axis=-1),
                                                       np.argmax(output.detach().cpu().numpy(), axis=-1))
                else:
                    batch_acc = metrics.accuracy_score(np.argmax(y.detach().cpu().numpy(), axis=-1),
                                                       np.argmax(output.detach().cpu().numpy(), axis=-1))
                tr_acc += batch_acc
                str_postfix = f'Round: {str_rnd} Epoch: {str_ep} Step: {str_st} Batch Acc: {round(batch_acc, 3)} Acc: {round(tr_acc / (i + 1), 3)} Batch Loss: {round(loss.item(), 3)} Loss: {round(tr_loss / (i + 1), 3)}'
                show = OrderedDict({f'[Train]': str_postfix})
                loader_iter.set_postfix(show)
                step = i
            ep_loss.append(tr_loss / (step + 1))
            ep_acc.append(tr_acc / (step + 1))
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['norm_cost'] = self.train_time_cost['total_cost'] / self.train_time_cost['num_rounds']

        return {
            'loss': ep_loss,
            'accuracy': ep_acc
        }


LocalUpdater = {
    'FedAvg': LocalUpdateAvg,
    'FedProx': LocalUpdateProx,
    'FedPerAvg': LocalUpdatePerAvg
}
