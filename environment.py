from argparse import Namespace
from loader import *
from torch.nn import Module
import torch

# Models
from torchvision.models.resnet import resnet50, resnet18
from models.protonet import ProtoNet
from models.rn import RelationNet
from models.rn import Net as EncoderNet
from models.tnt import tnt_s_patch16_224
from models.alexnet import alexnet
from models.mobilenet import mobilenet_v2
from models.fedavgcnn import FedAvgCNN


def get_dataset_environment(args: Namespace, cnf: dict, fs) -> DatasetFmt:
    """
    Get run environment
    """
    model_name = cnf['model']['name']
    ds_name = cnf['dataset']['name']
    if model_name == 'protonet' or model_name == 'relationNet':
        if ds_name == 'cifar100':
            return get_episodic_ci_far100(args, fs, **cnf['dataset'])
        elif ds_name == 'mini-imagenet':
            return get_episodic_mini_imagenet(args, fs, **cnf['dataset'])

        elif ds_name == 'omniglot':
            return get_episodic_omniglot(args, fs, **cnf['dataset'])
        else:
            raise ValueError(f'{ds_name} is not supported for {model_name} architecture!')

    else:
        raise ValueError('Invalid Config setting!')


def get_model_environment(args: Namespace, cnf: dict) -> Module:
    """
    Get models
    """
    model_name = cnf['model']['name']
    if model_name == 'protonet':
        backbone_name = cnf['model']['backbone']
        if backbone_name == 'resnet50':
            backbone = resnet50(pretrained=True)
            backbone.fc = torch.nn.Identity()
            return ProtoNet(backbone=backbone)
        elif backbone_name == 'resnet18':
            backbone = resnet18(pretrained=True)
            backbone.fc = torch.nn.Identity()
            return ProtoNet(backbone=backbone)

        elif backbone_name == 'alexnet':
            backbone = alexnet(pretrained=True)
            backbone.fc = torch.nn.Identity()
            return ProtoNet(backbone=backbone)
        elif backbone_name == 'mobilenet':
            backbone = mobilenet_v2(pretrained=True)
            backbone.fc = torch.nn.Identity()
            return ProtoNet(backbone=backbone)

        elif backbone_name=='fedavgcnn':
            backbone = FedAvgCNN(dim=18496)
            return ProtoNet(backbone=backbone)
        elif backbone_name == 'tnt+patch16+224':
            pretrained = False if cnf['model'].get('pretrained', None) is None else True
            pretrained_url = cnf['model'].get('pretrained', None)
            if args.verbose:
                print(f'Load Pretrain {pretrained_url}')
            backbone = tnt_s_patch16_224(pretrained=pretrained, pretrain_url=cnf['model'].get('pretrained', None))
            return ProtoNet(backbone=backbone)
        else:
            raise ValueError(f"Invalid Backbone {backbone_name} for {model_name}")
    elif model_name == 'relationNet':
        backbone = EncoderNet(num_in_channel=3, num_filter=64)
        return RelationNet(num_in_channel=64*2, num_filter=64, num_fc1=5 * 5 * 64, num_fc2=8, drop_prob=0.0,
                           backbone=backbone)
    else:
        raise ValueError(f'Invalid Model {model_name}!')
