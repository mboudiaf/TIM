import torch.optim
from sacred import Ingredient
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR

optim_ingredient = Ingredient('optim')


@optim_ingredient.config
def config():
    gamma = 0.1
    lr = 0.1
    lr_stepsize = 30
    nesterov = False
    weight_decay = 1e-4
    optimizer_name = 'SGD'
    scheduler = 'multi_step'


@optim_ingredient.capture
def get_scheduler(epochs, num_batches, optimizer, gamma, lr_stepsize, scheduler):

    SCHEDULER = {'step': StepLR(optimizer, lr_stepsize, gamma),
                 'multi_step': MultiStepLR(optimizer, milestones=[int(.5 * epochs), int(.75 * epochs)],
                                           gamma=gamma),
                 'cosine': CosineAnnealingLR(optimizer, num_batches * epochs, eta_min=1e-9),
                 None: None}
    print(epochs)
    return SCHEDULER[scheduler]


@optim_ingredient.capture
def get_optimizer(module, optimizer_name, nesterov, lr, weight_decay):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=lr, momentum=0.9,
                                        weight_decay=weight_decay, nesterov=nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)}
    return OPTIMIZER[optimizer_name]