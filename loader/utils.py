import numpy as np

import torch


def make_double_stochastic(x):
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


def print_split(idcs, labels, fs):
    # Tool for print divided dataset
    n_labels = np.max(labels) + 1
    print("[*] Data split:", file=fs)
    splits = []
    show = '{:<10}'.format('Name')
    for idx in range(n_labels):
        show += '{:<4}'.format(idx + 1)
    show += '{:<8}'.format('Total')
    print(show, file=fs)

    for i, idccs in enumerate(idcs):
        split = np.sum(np.array(labels)[idccs].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
        splits += [split]

        show = ""
        for s in split:
            show += '{:<4}'.format(s)
        if len(idcs) < 30 or i < 10 or i > len(idcs) - 10:
            print("Client {:<1}: {:<4} {:<8}".format(i + 1, show, np.sum(split)), flush=True, file=fs)
        elif i == len(idcs) - 10:
            print(".  " * 10 + "\n" + ".  " * 10 + "\n" + ".  " * 10, file=fs)

    show = '{:<10}'.format('Total')
    for idx in np.stack(splits, axis=0).sum(axis=0):
        show += '{:<4}'.format(idx)
    print(show, file=fs)
    print(file=fs)


def split_dirichlet(labels, n_clients, fs, alpha, double_stochastic=True, seed=0, verbose: bool = True):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

    np.random.seed(seed)

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    n_classes = np.max(labels) + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    if double_stochastic:
        label_distribution = make_double_stochastic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels) == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    if verbose:
        print_split(client_idcs, labels, fs)

    return client_idcs


def iid_separate(labels, n_clients, fs, balance: bool = False, seed: int = 0, verbose: bool = True):
    """
    IID data separation
    """
    n_classes = np.max(labels) + 1
    labels = np.array(labels)
    idx = np.array(range(len(labels)))
    idx_for_each_class = []
    for i in range(n_classes):
        idx_for_each_class.append(idx[labels == i])

    client_idx = [[] for _ in range(n_clients)]
    for i in range(n_classes):
        np.random.seed(seed + i)
        num_all_samples = len(idx_for_each_class[i])
        num_per = num_all_samples / n_clients
        if balance:
            num_samples = [int(num_per) for _ in range(n_clients)]

        else:
            num_samples = np.random.randint(max(num_per / 10, 1), num_per, n_clients).tolist()

        idx = 0
        for client, num_samples in zip(range(n_clients), num_samples):
            client_idx[client] += np.array(idx_for_each_class[i])[idx:idx + num_samples].tolist()
            idx += num_samples

    if verbose:
        print_split(client_idx, labels, fs)

    return client_idx
