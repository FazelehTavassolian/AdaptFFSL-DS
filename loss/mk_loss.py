from torch import nn

RegisterLoss = {
    'MSE': nn.MSELoss,
    'BCE': nn.BCELoss,
    'BCELogit': nn.BCEWithLogitsLoss,
    'CE': nn.CrossEntropyLoss
}


def mk_loss(args, **kwargs):
    if args.loss == 'MSE':
        return RegisterLoss[args.loss]()
    elif args.loss == 'CE':
        return RegisterLoss[args.loss]()
