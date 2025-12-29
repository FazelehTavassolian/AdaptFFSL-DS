import torch

HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if HAS_GPU else 'cpu')
