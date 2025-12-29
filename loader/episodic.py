from typing import Union, Tuple
from collections import Counter
from pathlib import Path
import json

import numpy as np
import PIL.Image as Image

# Torch
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

EpisodeFmt = Tuple[Tensor, Tensor, Tensor, Tensor]


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
    Episodic Dataset
    """

    def __init__(self, subset: Subset, transform: Union[Compose, None], input_shape: Tuple[int, int], **kwargs) -> None:
        self.spatial = input_shape
        self.transform = transform
        self.n_classes = kwargs['n_classes']
        self.n_support = kwargs['n_support']
        self.n_query = kwargs['n_query']
        self.n_episodes = kwargs['n_episodes']

        self.data = subset.data
        self.target = subset.targets

        # Support Tensors
        self.tensorS = torch.FloatTensor(self.n_classes * self.n_support, 3, *self.spatial)
        self.labelS = torch.LongTensor(self.n_classes * self.n_support)

        # Query Tensor
        self.tensorQ = torch.FloatTensor(self.n_classes * self.n_query, 3, *self.spatial)
        self.labelQ = torch.LongTensor(self.n_classes * self.n_query)

        # Image Tensor
        self.tensorImg = torch.FloatTensor(3, *self.spatial)

        for i in range(self.n_classes):
            self.labelS[i * self.n_support:(i + 1) * self.n_support] = i
            self.labelQ[i * self.n_query:(i + 1) * self.n_query] = i

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx) -> EpisodeFmt:
        cnt = Counter(self.target)
        labels = list(cnt.keys())
        labels_cnt = cnt.values()
        label_cnt_idx = np.where(np.array(list(labels_cnt)) >= (self.n_query + self.n_support))[0]

        labels = [labels[idx] for idx in label_cnt_idx]
        # labels = 5
        # print(labels)
        # print(self.n_classes)
        # print(self.n_query)
        class_idx = np.random.choice(labels, self.n_classes, replace=True)
        for i, cls in enumerate(class_idx):
            target_idx = np.where(np.array(self.target) == cls)[0]
            dt = [self.data[idx] for idx in target_idx]
            select_dt = np.random.choice(dt, self.n_query + self.n_support, replace=False)

            # Read Support set
            for j in range(self.n_support):
                img = Image.open(select_dt[j]).convert('RGB')
                self.tensorS[i * self.n_support + j] = self.tensorImg.copy_(self.transform(img))

            # Read Query set
            for j in range(self.n_query):
                img = Image.open(select_dt[j + self.n_support]).convert('RGB')
                self.tensorQ[i * self.n_query + j] = self.tensorImg.copy_(self.transform(img))

        perm_support = torch.randperm(self.n_classes * self.n_support)
        perm_query = torch.randperm(self.n_classes * self.n_query)

        return self.tensorS[perm_support], self.labelS[perm_support], self.tensorQ[perm_query], self.labelQ[perm_query]


class ValEpisodeDataset(Dataset):
    """
    Validation Episode Dataset
    """

    def __init__(self, img_root: Path, ep_json_path: Path, transform: Union[Compose, None],
                 input_shape: Tuple[int, int], **kwargs) -> None:
        self.img_root = img_root
        self.spatial = input_shape
        self.transform = transform
        self.n_classes = kwargs['n_classes']
        self.n_support = kwargs['n_support']
        self.n_query = kwargs['n_query']
        self.n_episodes = kwargs['n_episodes']

        # Read Episode Json
        with open(str(ep_json_path), 'r') as f:
            self.episodeInfo = json.load(f)

        # Support Tensors
        self.tensorS = torch.FloatTensor(self.n_classes * self.n_support, 3, *self.spatial)
        self.labelS = torch.LongTensor(self.n_classes * self.n_support)

        # Query Tensor
        self.tensorQ = torch.FloatTensor(self.n_classes * self.n_query, 3, *self.spatial)
        self.labelQ = torch.LongTensor(self.n_classes * self.n_query)

        # Image Tensor
        self.tensorImg = torch.FloatTensor(3, *self.spatial)

        for i in range(self.n_classes):
            self.labelS[i * self.n_support:(i + 1) * self.n_support] = i
            self.labelQ[i * self.n_query:(i + 1) * self.n_query] = i

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        for i in range(self.n_classes):
            for j in range(self.n_support):
                img = Image.open(self.img_root.joinpath(self.episodeInfo[idx]['Support'][i][j])).convert('RGB')
                self.tensorS[i * self.n_support + j] = self.tensorImg.copy_(self.transform(img))

            for j in range(self.n_query):
                img = Image.open(self.img_root.joinpath(self.episodeInfo[idx]['Query'][i][j])).convert('RGB')
                self.tensorQ[i * self.n_query + j] = self.tensorImg.copy_(self.transform(img))

        return self.tensorS, self.labelS, self.tensorQ, self.labelQ



