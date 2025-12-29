# Contain Episode learning
runtime = {
    'model': {
        'name': 'protonet',
        # 'backbone': 'resnet18',
        # 'backbone': 'resnet50',
        # 'backbone': 'tnt+patch16+224',
        # 'backbone': 'alexnet',
        # 'backbone': 'mobilenet',
        'backbone': 'fedavgcnn',
        'checkpoint': None,
        # 'pretrained': '/home/jericho/Project/federated-meta-learning/Data/models/tnt_s_patch16_224.pth'

    },
    'dataset': {
        'name': 'mini-imagenet',
        'root': '/home/jericho/Project/federated-meta-learning/Data/mini-imagenet',
        'n_support': 1,
        'n_query': 5,
        'n_classes': 5,
        'n_episodes': 80
    }
}
