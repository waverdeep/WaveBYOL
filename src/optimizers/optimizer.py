import torch.optim as optimizer
from adamp import AdamP


def get_optimizer(model_parameter, config):
    optimizer_name = config['optimizer_name']
    if optimizer_name == 'Adam':
        return optimizer.Adam(params=model_parameter,
                              lr=config['learning_rate'],
                              weight_decay=config['weight_decay'],
                              eps=config['eps'],
                              betas=config['betas'])
    elif optimizer_name == 'SGD':
        return optimizer.SGD(params=model_parameter,
                             lr=config['learning_rate'],
                             # momentum=config['momentum'],
                             # dampening=config['dampening'],
                             weight_decay=config['weight_decay'],)
                             # nesterov=config['nesterov'])
    elif optimizer_name == 'AdamP':
        return AdamP(model_parameter,
                     lr=config['learning_rate'],
                     betas=config['betas'],
                     weight_decay=config['weight_decay'],
                     eps=config['eps'])




