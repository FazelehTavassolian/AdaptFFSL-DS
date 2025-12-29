import torch
import torch.nn as nn
import torch.nn.functional as F

'''
define networks
'''


class Net(nn.Module):
    def __init__(self, num_in_channel, num_filter):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_in_channel, num_filter, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filter)
        self.conv4 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filter)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 40x40
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 20x20
        x = F.relu(self.bn3(self.conv3(x)))  # 20x20
        x = F.relu(self.bn4(self.conv4(x)))  # bx64x20x20
        return x


class RelationNet(nn.Module):
    def __init__(self, num_in_channel, num_filter, num_fc1, num_fc2, drop_prob, backbone):
        super(RelationNet, self).__init__()
        self.conv1 = nn.Conv2d(num_in_channel, num_filter, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.fc1 = nn.Linear(num_fc1, num_fc2)
        self.fc2 = nn.Linear(num_fc2, 1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.drop_prob = drop_prob
        self.backbone = backbone

    def forward(self, data):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        supp_x, supp_y, x, _ = data
        B, nSupp, C, H, W = supp_x.shape

        supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))
        query_f = self.backbone.forward(x.view(-1, *x.shape[2:]))
        supp_f = supp_f.view(B, supp_x.shape[1], -1)
        query_f = query_f.view(B, -1)
        query_f = torch.stack([query_f] * 5, 1)

        sup_qu_f = torch.concat([supp_f, query_f], dim=-1)
        sq_size = sup_qu_f.shape
        sup_qu_f = sup_qu_f.view(sq_size[0] * sq_size[1], 128, 20, 20)

        x = self.pool(F.relu(self.bn1(self.conv1(sup_qu_f))))  # 10x10
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 5x5
        x = x.view(x.size()[0], -1)  # 6400
        x = F.relu(self.fc1(x))  # 8
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = F.sigmoid(self.fc2(x))  # 1

        x = x.view(B,nSupp)

        return x
