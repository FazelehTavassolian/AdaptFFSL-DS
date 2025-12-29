# Contain Episode learning
runtime = {
    'model': {
        'name': 'protonet',
         'backbone': 'resnet18',
        # 'backbone': 'resnet50',
        # 'backbone': 'tnt+patch16+224',
        # 'backbone': 'alexnet',
        # 'backbone': 'mobilenet',
        #'backbone': 'fedavgcnn',
        'checkpoint': None,
        #'checkpoint':'/home/basu/project/lightffsl-sc-dqn//experimental/Run1_cifar5salpha0_5ResfedproxffslDqn50round50client',
        ## 'pretrained': '/home/basu/Project/federated-meta-learning/Data/models/tnt_s_patch16_224.pth'

    },
    'dataset': {
        'name': 'cifar100',
        'root': '/home/basu/project/lightffsl-sc-dqn/Data/cifar-fs',#'/home/basu/project/lightffsl-sc/Data/cifar-fs',
        'n_support': 1,
        'n_query': 5,
        'n_classes': 5,
        'n_episodes': 100
    }
}

