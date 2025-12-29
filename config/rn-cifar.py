# Contain Episode learning
runtime = {
    'model': {
        'name': 'relationNet',
        'backbone': 'resnet18',
        # 'backbone': 'resnet50',
        # 'backbone': 'tnt+patch16+224',
        # 'checkpoint': '/home/jericho/Project/federated-meta-learning/Data/cpt.latest.pt',

    },
    'dataset': {
        'name': 'cifar100',
        'root': '/home/jericho/Project/federated-meta-learning/Data/cifar-fs',
        'n_support': 5,
        'n_query': 1,
        'n_classes': 1,
        'n_episodes': 80
    }
}
