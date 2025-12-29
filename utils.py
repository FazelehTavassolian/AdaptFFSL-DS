import yaml
import sys
import settings
import torch
import copy
import numpy as np
from pathlib import Path
from importlib import import_module
from sklearn.metrics import accuracy_score


def read_configs(yml_path: str, show: bool = True) -> dict:
    """
    Read and parse the configuration file
    :param yml_path:
    :param show: log the details
    :return: configuration dictionary
    """

    with open(yml_path, "r") as file:
        parsed_data = yaml.load(file, Loader=yaml.FullLoader)

    configs = {'clients': []}

    for key, value in parsed_data.items():
        if key.startswith('client'):
            configs['clients'].append(value if settings.HAS_GPU else 'cpu')
        else:
            configs[key] = value

    if show:
        print('[*] Hyper parameters')
        print("\t{:<8} {:<15} {:<10}".format('#', 'Type', 'Value'))
        for idx, (key, value) in enumerate(configs['param'].items()):
            print("\t{:<8} {:<15} {:<10}".format(idx + 1, key.capitalize(), value))

        print('[*] Clients')
        print("\t{:<8} {:<15} {:<10}".format('#', 'Name', 'Device'))
        for idx, cl in enumerate(configs['clients']):
            print("\t{:<8} {:<15} {:<10}".format(idx + 1, f'Client {idx + 1}', cl.get('device')))

    return configs


def fed_avg(w):
    """
    fed average operation
    :param w: weights
    :return: weights
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def evaluate(loader, model, configs):
    """
    Evaluation function
    :param loader:
    :param model:
    :param configs:
    :return:
    """
    server_model = model.eval()
    with torch.no_grad():
        test_correct = 0
        total = 0
        for idx, batch in enumerate(loader):
            supportData, supportLabel, x, y = batch
            if settings.HAS_GPU and configs['server']['device'] == 'gpu':
                supportData = supportData.to(settings.DEVICE)
                supportLabel = supportLabel.to(settings.DEVICE)
                x = x.to(settings.DEVICE)
                y = y.to(settings.DEVICE)
            output = server_model(supportData, supportLabel, x)
            output = output.view(x.shape[0] * x.shape[1], -1)
            y = y.view(-1)
            total += y.size(0)
            predictions = torch.max(output, 1)
            test_correct += np.sum(predictions[1].cpu().numpy() == y.cpu().numpy())

    return {
        'accuracy': float(test_correct) / total
    }


def tnt_evaluation(loader, model, configs):
    server_model = model.eval()
    with torch.no_grad():
        test_correct = 0
        total = 0
        for idx, batch in enumerate(loader):
            x, y = batch
            if settings.HAS_GPU and configs['server']['device'] == 'gpu':
                x = x.to(settings.DEVICE)
                y = y.to(settings.DEVICE)
            output = server_model(x)
            total += y.size(0)
            predictions = np.argmax(output.cpu().detach().numpy(), axis=1)
            test_correct += accuracy_score(y.cpu().detach().numpy(), predictions)

    return {
        'accuracy': float(test_correct) / total
    }


def read_py_config(filename: Path):
    absolute = filename.absolute()
    assert absolute.suffix.endswith('.py')
    config_dir = filename.parent
    sys.path.insert(0, str(config_dir))
    mod = import_module(filename.stem)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }

    return cfg_dict['runtime']
