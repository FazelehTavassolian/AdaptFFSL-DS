from pathlib import Path
from threading import Barrier
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import gc
import torch
import pickle

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, RandomSampler

# Settings
from settings import HAS_GPU, DEVICE

# Strategy
from strategy import STRATEGIES


class Client:
    """
    Client class to train the nodes weights
    """

    def __init__(self, loader_path, optim, model, loss_fn, barrier: Barrier, end_barrier: Barrier, save_path: Path,
                 name: str,
                 epochs: int, device, lr: float):
        """
        :param loader_path: Dataset path
        :param optim: optimizer function
        :param model: model object
        :param loss_fn: loss function
        :param barrier: (concurrent)
        :param end_barrier: (concurrent)
        :param save_path: output path
        :param name: client`s name
        :param epochs: number of epochs
        :param device: target device
        :param lr: leaning rate
        """
        self.save_path = save_path.joinpath(name)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.model_snapshot = self.save_path.joinpath('snapshot.pth')

        self.epochs = epochs
        self.loader_path = loader_path

        self.barrier = barrier
        self.end_barrier = end_barrier

        self.name = name

        self.model = model
        self.device = DEVICE if HAS_GPU and device == 'gpu' else 'cpu'
        self.model = self.model.to(self.device)

        self.optim = optim([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.loss = loss_fn()

        self.snapshot()
        self.unload()

    def read_loader(self):
        """
        read dataset
        :return: dataset
        """
        with open(str(self.loader_path), 'rb') as f:
            return pickle.load(f)

    def snapshot(self) -> None:
        """
        save client`s state
        :return: None
        """
        obj = {
            'net': self.model,
            'optim': self.optim
        }
        torch.save(obj, self.model_snapshot)

    def to(self, device) -> None:
        """
        transfer model on specific device
        :param device:
        :return: None
        """
        self.model = self.model.to(device)

    def revert(self) -> None:
        """
        return to client`s target device
        :return:
        """
        self.model = self.model.to(self.device)

    @property
    def state_dict(self):
        """
        get model`s weights
        :return:
        """
        return self.model.state_dict()

    def load_state_dict(self, st) -> None:
        """
        Load model`s weights
        :param st:
        :return:
        """
        self.model.load_state_dict(st)

    def train_step(self, epoch, loader, **kwargs):
        """
        Train on one epoch
        """
        gc.collect()
        torch.cuda.empty_cache()

        list_loss = []
        list_acc = []
        for idx, batch in enumerate(loader):
            supportData, supportLabel, x, y = batch

            supportData = supportData.to(self.device)
            supportLabel = supportLabel.to(self.device)
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward
            output = self.model(supportData, supportLabel, x)
            output = output.view(x.shape[0] * x.shape[1], -1)

            y = y.view(-1)

            loss = self.loss(output, y)

            if kwargs['fl_strategy'] == 'FedProx':
                print('ffffff')
                loss += kwargs['fl_strategy_fn'](glob_model=kwargs['glob_model'],
                                                 client_model=self.model,
                                                 mu=kwargs['args'].stg_mu)

            # Backward
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            list_loss.append(loss.item())
            list_acc.append(accuracy_score(y.cpu().detach().numpy(), np.argmax(output.cpu().detach().numpy(), axis=1)))

        return {
            'accuracy': np.array(list_acc).mean(),
            'loss': np.array(list_loss).mean()
        }

    def train_tnt_step(self, epoch, loader, **kwargs):
        """
                Train on one epoch
                """
        gc.collect()
        torch.cuda.empty_cache()

        list_loss = []
        list_acc = []
        for idx, batch in enumerate(loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward
            output = self.model(x)
            loss = self.loss(output, y)

            if kwargs['fl_strategy'] == 'FedProx':
                print('ffffff')
                loss += kwargs['fl_strategy_fn'].forward(glob_model=kwargs['glob_model'],
                                                         client_model=self.model,
                                                         mu=kwargs['args'].stg_mu)

            # Backward
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            list_loss.append(loss.item())
            list_acc.append(accuracy_score(y.cpu().detach().numpy(), np.argmax(output.cpu().detach().numpy(), axis=1)))

        return {
            'accuracy': np.array(list_acc).mean(),
            'loss': np.array(list_loss).mean()
        }

    def train(self, rnd: int, **kwargs):
        save_path = self.save_path.joinpath(f'Round-{str(rnd).zfill(4)}')
        save_path.mkdir(parents=True, exist_ok=True)

        if kwargs.get('thread'):
            self.barrier.wait()

        # Train scheduler
        scheduler = ExponentialLR(self.optim, gamma=0.95)
        e_loss = []
        e_acc = []

        if kwargs.get('has_loader'):
            loader = self.read_loader()
        else:
            train_ds = self.read_loader()
            sampler = RandomSampler(train_ds)

            generator = torch.Generator()
            generator.manual_seed(kwargs['seed'])
            loader = DataLoader(train_ds,
                                batch_size=kwargs['batch_size'],
                                drop_last=True,
                                pin_memory=True,
                                sampler=sampler,
                                generator=generator)
        for epoch in tqdm(range(self.epochs), desc=self.name):
            self.model.train(True)

            if kwargs.get('train_fn') == 'tnt':
                results = self.train_tnt_step(epoch, loader, **kwargs)
            else:
                results = self.train_step(epoch, loader, **kwargs)
            scheduler.step()
            e_loss.append(results.get('loss'))
            e_acc.append(results.get('accuracy'))
            gc.collect()
        plt.rcParams.update({'font.size': 140})
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(200, 80))
        ax_loss.plot(list(range(self.epochs)), e_loss, color=colors[0], label="Train", linewidth=7.0)
        ax_acc.plot(list(range(self.epochs)), e_acc, color=colors[0], label="Train", linewidth=7.0)
        ax_loss.grid(False)
        ax_loss.legend()
        ax_loss.set_title("Loss")
        ax_acc.grid(False)
        ax_acc.set_title("accuracy")
        ax_acc.legend()
        fig.savefig(save_path.joinpath('result.jpg'))
        plt.clf()
        self.snapshot()

        if kwargs.get('thread'):
            self.end_barrier.wait()

    def run(self, args, rnd: int, thread: bool, batch_size: int, seed: int, has_loader=False, train_fn=None,
            fl_strategy=None,
            glob_model=None):
        """
        Start Train loop
        :return:
        """
        self.load()
        print(f'[{self.name}] is starting on round {rnd}', flush=True)
        self.train(rnd,
                   thread=thread,
                   batch_size=batch_size,
                   seed=seed,
                   has_loader=has_loader,
                   train_fn=train_fn,
                   fl_strategy=fl_strategy,
                   fl_strategy_fn=STRATEGIES.get(fl_strategy, None)() if STRATEGIES.get(fl_strategy,
                                                                                        None) is not None else None,
                   glob_model=glob_model,
                   args=args)
        self.unload()

    def load(self) -> None:
        """
        load model state
        :return:
        """
        obj = torch.load(self.model_snapshot)
        self.model = obj['net']
        self.optim = obj['optim']
        self.model = self.model.to(self.device)

    def unload(self) -> None:
        """
        remove model state from memory
        :return:
        """
        del self.model
        del self.optim
