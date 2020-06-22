import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.optim
import torch.nn as nn

import config as c

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 2*c_in), nn.LeakyReLU(),
                         nn.Linear(2*c_in,  c_out))


nodes = [Ff.InputNode(c.ndim_x+c.ndim_pad_x, name='input')]

for i in range(c.n_blocks):
    nodes.append(Ff.Node(nodes[-1],
                         Fm.GLOWCouplingBlock,
                         {'subnet_constructor':subnet_fc,
                          'clamp':c.clamp},
                         name=F'coupling_{i}'))
    nodes.append(Ff.Node(nodes[-1],
                         Fm.PermuteRandom,
                         {'seed':i},
                         name=F'permute_{i}'))

nodes.append(Ff.OutputNode(nodes[-1], name='output'))
model = Ff.ReversibleGraphNet(nodes)
model.to(device)

params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
for p in params_trainable:
    p.data = c.init_scale * torch.randn(p.data.shape).to(device)

#gamma = (c.final_decay)**(1./c.n_epochs)
optim = torch.optim.Adam(params_trainable, lr=c.lr_init, betas=c.adam_betas, 
                         eps=1e-6, weight_decay=c.l2_weight_reg)
#weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)


def save(name):
    torch.save({'opt':optim.state_dict(),
                'net':model.state_dict()}, name)

def load(name):
    state_dicts = torch.load(name)
    model.load_state_dict(state_dicts['net'])
    try:
        optim.load_state_dict(state_dicts['opt'])
    except ValueError:
        print('Cannot load optimizer for some reason or other')
