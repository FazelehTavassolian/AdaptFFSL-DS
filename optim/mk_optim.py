from torch import optim

RegisterOptim = {
    'Adam': optim.Adam,
    'SGD': optim.SGD,
}


def mk_optim(args, params):
    if args.optim == 'Adam':
        return optim.Adam(params=params, lr=args.lr, weight_decay=args.opt_weight_decay)
    elif args.optim == 'SGD':
        return optim.SGD(params=params, lr=args.lr, weight_decay=args.opt_weight_decay, momentum=args.opt_momentum)
