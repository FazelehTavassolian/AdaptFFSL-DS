# Common
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

# Torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Dataset
from .base import DatasetFmt
from .episodic import EpisodeDataset, ValEpisodeDataset, Subset
from .utils import split_dirichlet, iid_separate

from aug import mk_train_transform


class OmniglotDataset(Dataset):
    def __init__(self, img_dir: Path) -> None:
        self.img_dir = img_dir
        self.n_class = len(list(self.img_dir.glob('*')))
        self.class_name = [p.stem for p in self.img_dir.glob('*')]
        self.encoder = LabelEncoder()
        self.encoder.fit(self.class_name)
        self.data = []
        self.targets = []

        for cl in self.img_dir.glob('*/*'):
            self.data.append(cl)
            self.targets.append(self.encoder.transform([cl.parent.stem])[0])

        idx = np.arange(len(self.data))
        np.random.shuffle(idx)

        # Rebase
        self.data = [self.data[i] for i in idx]
        self.targets = [self.targets[i] for i in idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_episodic_omniglot(args,fs, **kwargs) -> DatasetFmt:
    # Make Augmentation
    t_transform = mk_train_transform(args)
    t_transform += [
        transforms.ToTensor()
    ]

    t_transform = transforms.Compose(t_transform)

    v_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()])

    ds_root = Path(kwargs['root'])

    # Create Validation
    val_ep_json = 'val1000Episode_5_way_1_shot.json' if kwargs['n_support'] == 1 else 'val1000Episode_5_way_5_shot.json'
    valid_ds = ValEpisodeDataset(img_root=ds_root.joinpath('val'),
                                 ep_json_path=ds_root.joinpath(val_ep_json),
                                 transform=v_transform,
                                 input_shape=(args.img_size, args.img_size),
                                 **kwargs)

    # Separate Dataset for each client
    train_ds = OmniglotDataset(img_dir=ds_root.joinpath('train'))
    if args.iid:
        client_idx = iid_separate(train_ds.targets, args.n_clients, balance=args.balance, seed=args.seed,
                                  verbose=args.verbose,fs = fs)

    else:
        client_idx = split_dirichlet(train_ds.targets, args.n_clients, seed=args.seed, verbose=args.verbose,
                                     alpha=args.alpha,fs=fs)

    clients_subset = [Subset(train_ds, targets) for targets in client_idx]
    clients_episodes = [
        EpisodeDataset(subset, transform=t_transform, input_shape=(args.img_size, args.img_size), **kwargs) for subset
        in clients_subset]

    return clients_episodes, valid_ds, valid_ds
