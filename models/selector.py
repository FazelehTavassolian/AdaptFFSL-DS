from torchvision.models.resnet import resnet50, resnet18
import torch
from .protonet import ProtoNet
from .tnt import tnt_s_patch16_224
from .alexnet import alexnet
from .mobilenet import mobilenet_v2


def get_model(arch: str, **kwargs):
    """
    Define models architecture
    :param arch:
    :return:
    """
    if arch == 'protonet+resnet50':
        backbone = resnet50(pretrained=True)
        backbone.fc = torch.nn.Identity()
        return ProtoNet(backbone=backbone)

    elif arch == 'protonet+resnet18':
        backbone = resnet18(pretrained=True)
        backbone.fc = torch.nn.Identity()
        return ProtoNet(backbone=backbone)

    elif arch == 'protonet+alexnet':
        backbone = alexnet(pretrained=True)
        backbone.fc = torch.nn.Identity()
        return ProtoNet(backbone=backbone)

    elif arch == 'protonet+mobilenet':
        backbone = mobilenet_v2(pretrained=True)
        backbone.fc = torch.nn.Identity()
        return ProtoNet(backbone=backbone)

    elif arch == 'tnt+patch16+224':
        return tnt_s_patch16_224(pretrained=True, **kwargs)


    else:
        return None
