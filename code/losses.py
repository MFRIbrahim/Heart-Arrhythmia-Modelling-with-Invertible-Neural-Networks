import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def l2_loss(input, target):
    #l2 = torch.mean((input - target)**2)
    #assert(not np.any(np.isnan(np.array(l2.cpu().detach()))))
    
    return torch.mean((input - target)**2)

def MMD_multiscale(x, y, widths_exponents):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for C,a in widths_exponents:
        XX += C**a * ((C + dxx) / a)**-a
        YY += C**a * ((C + dyy) / a)**-a
        XY += C**a * ((C + dxy) / a)**-a

    return torch.mean(XX + YY - 2.*XY)