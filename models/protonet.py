import torch
from torch import nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # Backbone
        self.backbone = backbone

        # Bias and Scale
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

    def cos_classifier(self, w, f):
        f = F.normalize(f, p=2, dim=f.dim() - 1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim() - 1, eps=1e-12)
        cls_scores = f @ w.transpose(1, 2)  # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    def forward(self, data):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        supp_x, supp_y, x, _ = data
        num_classes = supp_y.max() + 1  # NOTE: assume B==1

        B, nSupp, C, H, W = supp_x.shape
        supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))
        supp_f = supp_f.view(B, nSupp, -1)

        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2)  # B, nC, nSupp

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True)  # NOTE: may div 0 if some classes got 0 images

        feat = self.backbone.forward(x.view(-1, C, H, W))
        feat = feat.view(B, x.shape[1], -1)  # B, nQry, d
        output = self.cos_classifier(prototypes, feat)  # B, nQry, nC

        return output.view(x.shape[0] * x.shape[1], -1)
