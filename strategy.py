from abc import ABC, abstractmethod
import copy
import torch


class Strategy(ABC):

    def forward(self, **kwargs):
        NotImplementedError('This method should be implemented!')


class FedAvg(Strategy):
    def __init__(self):
        pass

    def forward(self, **kwargs):
        weights = kwargs.get('weights')
        w_avg = copy.deepcopy(weights[0])
        for k in w_avg.keys():
            for i in range(1, len(weights)):
                w_avg[k] += weights[i][k]
            w_avg[k] = torch.div(w_avg[k], len(weights))
        return w_avg


class FedProx(Strategy):
    def __init__(self):
        pass

    def forward(self, **kwargs):
        glob_model = kwargs.get('glob_model')
        client_model = kwargs.get('client_model')

        proximal_term = 0.0
        for w, w_t in zip(client_model.parameters(), glob_model.parameters()):
            proximal_term += (w - w_t).norm(2)

        return proximal_term * (kwargs['mu'] / 2)


# class PerFedAvg(Strategy):
#     def __init__(self):
#         pass
#
#     def forward(self, **kwargs):
#         sec_order = kwargs.get('second_order', False)
#         client_model = kwargs.get('client_model')
#         x_input = kwargs.get('x_input')
#
#         if sec_order:
#             pass
#
#         else:
#


STRATEGIES = {
    'FedProx': FedProx,
}
