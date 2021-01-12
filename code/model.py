import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.optim
import torch.nn as nn

import config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, config.n_hidden_layer_size),
                         nn.BatchNorm1d(config.n_hidden_layer_size),
                         nn.ELU(),
                         nn.Linear(config.n_hidden_layer_size,
                                   config.n_hidden_layer_size),
                         nn.BatchNorm1d(config.n_hidden_layer_size),
                         nn.ELU(),
                         nn.Linear(config.n_hidden_layer_size,  dims_out))


class subnet(nn.Module):
    def __init__(self, dims_in, dims_out):
        super().__init__()

        self.lstm = nn.LSTM(dims_in, config.hidden_size,
                            config.rnn_layers, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2*config.hidden_size, dims_out)

    def forward(self, x):
        x = self.lstm(x)[0]
        x = self.lin(x)
        return x


def subnet_func(dims_in, dims_out):
    return subnet(dims_in, dims_out)


def generate_cINN_old():
    cond = Ff.ConditionNode(config.n_cond_features, name='condition')
    nodes = [Ff.InputNode(config.n_x_features, name='input')]

    for k in range(config.n_blocks):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet_fc,
                                 'clamp': config.clamp},
                             conditions=cond,
                             name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed': k},
                             name=F'permute_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    model = Ff.ReversibleGraphNet(nodes + [cond])
    model.to(device)

    params_trainable = list(
        filter(lambda p: p.requires_grad, model.parameters()))
    for p in params_trainable:
        p.data = config.init_scale * torch.randn_like(p).to(device)

    gamma = config.gamma
    optim = torch.optim.AdamW(params_trainable, lr=config.lr)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=1, gamma=gamma)

    return model, optim, weight_scheduler


def generate_rcINN_old():
    cond = Ff.ConditionNode(config.n_cond_features, name='condition')
    nodes = [Ff.InputNode(config.n_x_features, name='input')]

    for k in range(config.n_blocks):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet_func,
                                 'clamp': config.clamp},
                             conditions=cond,
                             name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed': k},
                             name=F'permute_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    model = Ff.ReversibleGraphNet(nodes + [cond])
    model.to(device)

    params_trainable = list(
        filter(lambda p: p.requires_grad, model.parameters()))
    for p in params_trainable:
        p.data = config.init_scale * torch.randn_like(p).to(device)

    gamma = config.gamma
    optim = torch.optim.AdamW(params_trainable, lr=config.lr)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=1, gamma=gamma)

    return model, optim, weight_scheduler


def save(name, optim, model):
    torch.save({'opt': optim.state_dict(),
                'net': model.state_dict()}, name)


def load(name, optim, model):
    state_dicts = torch.load(name)
    model.load_state_dict(state_dicts['net'])
    try:
        optim.load_state_dict(state_dicts['opt'])
    except ValueError:
        print('Cannot load optimizer for some reason or other')
