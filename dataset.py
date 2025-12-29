from typing import Tuple
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn import preprocessing
import PIL.Image as Image
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision


def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    return x


def print_split(idcs, labels):
    # Tool for print divided dataset
    n_labels = np.max(labels) + 1
    print("[*] Data split:")
    splits = []
    show = '{:<10}'.format('Name')
    for idx in range(n_labels):
        show += '{:<4}'.format(idx + 1)
    show += '{:<8}'.format('Total')
    print(show)

    for i, idccs in enumerate(idcs):
        split = np.sum(np.array(labels)[idccs].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
        splits += [split]

        show = ""
        for s in split:
            show += '{:<4}'.format(s)
        if len(idcs) < 30 or i < 10 or i > len(idcs) - 10:
            print("Client {:<1}: {:<4} {:<8}".format(i + 1, show, np.sum(split)), flush=True)
        elif i == len(idcs) - 10:
            print(".  " * 10 + "\n" + ".  " * 10 + "\n" + ".  " * 10)

    show = '{:<10}'.format('Total')
    for idx in np.stack(splits, axis=0).sum(axis=0):
        show += '{:<4}'.format(idx)
    print(show)
    print()


def split_dirichlet(labels, n_clients, alpha, double_stochstic=True, seed=0):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

    np.random.seed(seed)

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    n_classes = np.max(labels) + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    if double_stochstic:
        label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels) == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    print_split(client_idcs, labels)

    return client_idcs


class Subset:
    """
        Store sub-dataset
    """
    def __init__(self, ds, target_indexes):
        self.data = []
        self.targets = []

        for idx in target_indexes:
            dt, y = ds[idx]
            self.data.append(dt)
            self.targets.append(y)


class EpisodeDataset(Dataset):
    """
        Create Train Episodes
    """

    def __init__(self, subset: Subset, configs: dict, transform: transforms.Compose, input_shape: Tuple[int, int]):
        """
        Episode constructor
        :param subset:
        :param configs:
        :param transform: transformation function
        :param input_shape: input image shape
        """

        self.shape = input_shape
        self.transform = transform
        self.configs = configs

        self.data = subset.data
        self.targets = subset.targets
        self.tensorSupport = torch.FloatTensor(configs['param']['n_classes'] * configs['param']['n_support'], 3,
                                               *input_shape)
        self.labelSupport = torch.LongTensor(configs['param']['n_classes'] * configs['param']['n_support'])

        self.tensorQuery = torch.FloatTensor(configs['param']['n_classes'] * configs['param']['n_query'], 3,
                                             *input_shape)
        self.labelQuery = torch.LongTensor(configs['param']['n_classes'] * configs['param']['n_query'])

        self.imgTensor = torch.FloatTensor(3, *input_shape)

        for i in range(configs['param']['n_classes']):
            self.labelSupport[i * configs['param']['n_support']: (i + 1) * configs['param']['n_support']] = i
            self.labelQuery[i * configs['param']['n_query']: (i + 1) * configs['param']['n_query']] = i

    def __len__(self):
        return self.configs['param']['n_episodes']

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        cnt = Counter(self.targets)
        labels = list(cnt.keys())
        labels_cnt = cnt.values()
        label_cnt_idx = \
            np.where(
                np.array(list(labels_cnt)) >= (self.configs['param']['n_support'] + self.configs['param']['n_query']))[
                0]
        labels = [labels[idx] for idx in label_cnt_idx]

        clss = np.random.choice(labels, self.configs['param']['n_classes'], replace=False)
        for i, cls in enumerate(clss):
            cls_idx = np.where(np.array(self.targets) == cls)[0]
            target_dt = [self.data[idx] for idx in cls_idx]

            selected_img = np.random.choice(target_dt,
                                            self.configs['param']['n_support'] + self.configs['param']['n_query'],
                                            replace=False)
            # Generate support set
            for j in range(self.configs['param']['n_support']):
                img = Image.open(selected_img[j]).convert('RGB')
                self.tensorSupport[i * self.configs['param']['n_support'] + j] = self.imgTensor.copy_(
                    self.transform(img))

            # Generate query set
            for j in range(self.configs['param']['n_query']):
                img = Image.open(selected_img[j + self.configs['param']['n_support']]).convert('RGB')
                self.tensorQuery[i * self.configs['param']['n_query'] + j] = self.imgTensor.copy_(self.transform(img))

        perm_support = torch.randperm(self.configs['param']['n_classes'] * self.configs['param']['n_support'])
        perm_query = torch.randperm(self.configs['param']['n_classes'] * self.configs['param']['n_query'])

        return (self.tensorSupport[perm_support],
                self.labelSupport[perm_support],
                self.tensorQuery[perm_query],
                self.labelQuery[perm_query])


class ValEpisodeDataset(Dataset):
    """
    Same as EpisodeDataset
    """
    def __init__(self, img_root: Path, configs: dict, transform: transforms.Compose, input_shape: Tuple[int, int],
                 ep_json_path):
        self.shape = input_shape
        self.transform = transform
        self.configs = configs

        self.img_root = img_root

        with open(str(ep_json_path), 'r') as f:
            self.episodeInfo = json.load(f)

        self.tensorSupport = torch.FloatTensor(configs['param']['n_classes'] * configs['param']['n_support'], 3,
                                               *input_shape)
        self.labelSupport = torch.LongTensor(configs['param']['n_classes'] * configs['param']['n_support'])

        self.tensorQuery = torch.FloatTensor(configs['param']['n_classes'] * configs['param']['n_query'], 3,
                                             *input_shape)
        self.labelQuery = torch.LongTensor(configs['param']['n_classes'] * configs['param']['n_query'])

        self.imgTensor = torch.FloatTensor(3, *input_shape)
        for i in range(configs['param']['n_classes']):
            self.labelSupport[i * configs['param']['n_support']: (i + 1) * configs['param']['n_support']] = i
            self.labelQuery[i * configs['param']['n_query']: (i + 1) * configs['param']['n_query']] = i

    def __len__(self):
        return self.configs['param']['n_episodes']

    def __getitem__(self, idx):
        for i in range(self.configs['param']['n_classes']):
            for j in range(self.configs['param']['n_support']):
                img = Image.open(self.img_root.joinpath(self.episodeInfo[idx]['Support'][i][j])).convert('RGB')
                self.tensorSupport[i * self.configs['param']['n_support'] + j] = self.imgTensor.copy_(
                    self.transform(img))

            for j in range(self.configs['param']['n_query']):
                img = Image.open(self.img_root.joinpath(self.episodeInfo[idx]['Query'][i][j])).convert('RGB')
                self.tensorQuery[i * self.configs['param']['n_query'] + j] = self.imgTensor.copy_(self.transform(img))

        return (self.tensorSupport,
                self.labelSupport,
                self.tensorQuery,
                self.labelQuery)


class CifarDataset:
    def __init__(self, img_dir: Path):
        self.img_dir = img_dir
        self.n_class = len(list(self.img_dir.glob('*')))
        self.class_name = [p.stem for p in self.img_dir.glob('*')]
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(self.class_name)
        self.data = []
        self.targets = []

        for cl in self.img_dir.glob('*/*'):
            self.data.append(cl)
            self.targets.append(self.encoder.transform([cl.parent.stem])[0])

        idx = np.arange(len(self.data))
        np.random.shuffle(idx)

        # rebase
        self.data = [self.data[i] for i in idx]
        self.targets = [self.targets[i] for i in idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def cifar_setting(nSupport, root: Path, img_size=32):
    """
    Return dataset setting
    :param int nSupport: number of support examples
    """
    mean = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
    std = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    valTransform = transforms.Compose([  # lambda x: np.asarray(x),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize])
    inputW, inputH, nbCls = img_size, img_size, 64

    trainDir = 'cifar-fs/train/'
    valDir = 'cifar-fs/val/'
    testDir = 'cifar-fs/test/'
    episodeJson = 'cifar-fs/val1000Episode_5_way_1_shot.json' if nSupport == 1 \
        else 'cifar-fs/val1000Episode_5_way_5_shot.json'
    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, root.joinpath(episodeJson), nbCls, \
           CifarDataset(root.joinpath(trainDir))


def mini_image_net_setting(nSupport, root: Path, img_size=32):
    """
    Return dataset setting
    :param int nSupport: number of support examples
    """
    mean = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    std = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([  # transforms.RandomCrop(img_size, padding=8),
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        # lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize
    ])

    valTransform = transforms.Compose([  # transforms.CenterCrop(80),
        transforms.Resize((img_size, img_size)),
        # lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize])

    inputW, inputH, nbCls = img_size, img_size, 64

    trainDir = 'Mini-ImageNet/train/'
    valDir = 'Mini-ImageNet/val/'
    testDir = 'Mini-ImageNet/test/'
    episodeJson = 'Mini-ImageNet/val1000Episode_5_way_1_shot.json' if nSupport == 1 \
        else 'Mini-ImageNet/val1000Episode_5_way_5_shot.json'

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, root.joinpath(
        episodeJson), nbCls, CifarDataset(root.joinpath(trainDir))


def get_cifar(num_classes=100, dataset_dir="./Data",resize=224):
    """
        Get Cifar 10 or 100 dataset
    """

    if num_classes == 10:
        print("[*] Loading CIFAR10...")
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        print("[*] Loading CIFAR100...")
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Resize(size=resize),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=resize),
        normalize,
    ])

    train_data = dataset(root=dataset_dir, train=True,
                         download=True, transform=train_transform)

    test_data = dataset(root=dataset_dir, train=False,
                        download=True,
                        transform=test_transform)

    return train_data, test_data


# Defined Dataset
DATASET = {
    'cifar': cifar_setting,
    'miniImageNet': mini_image_net_setting
}
