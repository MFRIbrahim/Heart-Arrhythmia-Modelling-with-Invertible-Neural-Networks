import torch
import torch.nn as nn
import numpy as np

import config as c

import datagen
import model as Model
import losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(i_epoch, loader):
    Model.model.train()
    l_tot = 0
    batch_idx = 0
    
    #loss_factor = min(1., 2. * 0.002**(1. - (float(i_epoch) / c.n_epochs)))
    
    for x, y in loader:
        batch_idx += 1
        if batch_idx > c.n_its_per_epoch:
            break
        
        x, y = x.to(device), y.to(device)
        
        y_clean = y.clone()
        pad_x = c.zero_noise * torch.randn(c.batch_size, c.ndim_pad_x, device=device)
        pad_zy = c.zero_noise * torch.randn(c.batch_size, c.ndim_pad_zy, device=device)
        
        x += c.x_noise * torch.randn(c.batch_size, c.ndim_x, device=device)
        x = torch.cat((x, pad_x), dim=1)
        y += c.y_noise * torch.randn(c.batch_size, c.ndim_y, device=device)
        z = torch.randn(c.batch_size, c.ndim_z, device=device)
        zy = torch.cat((z, pad_zy, y),dim=1)
        
        Model.optim.zero_grad
        
        output = Model.model(x)
        zy_short = torch.cat((zy[:, :c.ndim_z], zy[:, -c.ndim_y:]), dim=1)
        l = c.lambd_l2 * losses.l2_loss(output[:, c.ndim_z:], zy[:, c.ndim_z:])
        output_block_grad = torch.cat((output[:, :c.ndim_z], 
                                       output[:, -c.ndim_y:].data), dim=1)
        l += c.lambd_mmd_forw * losses.MMD_multiscale(output_block_grad, zy_short, 
                                                      c.mmd_forw_kernels)
        l_tot += l.data.item()
        
        l.backward()
        
        pad_zy = c.zero_noise * torch.randn(c.batch_size, c.ndim_pad_zy, device=device)
        
        y = y_clean + c.y_noise * torch.randn(c.batch_size, c.ndim_y, device=device)
        z = torch.randn(c.batch_size, c.ndim_z, device=device)
        zy_rev = torch.cat((z, pad_zy, y), dim=1)
        
        output_rev_rand = Model.model(zy_rev, rev=True)
        l_rev = c.lambd_mmd_back * losses.MMD_multiscale(output_rev_rand[:, :c.ndim_x], 
                                                                       x[:, :c.ndim_x], 
                                                                       c.mmd_back_kernels)
        l_tot += l_rev.data.item()
        
        l_rev.backward()
        
        for p in Model.model.parameters():
            p.grad.data.clamp_(-15, 15)
        
        Model.optim.step()    
        
    return l_tot/batch_idx
        

def test(i_epoch, loader):
    # use generate_dataloader to make an independent test set
    return #performance in the test so as to decide whether to keep as best net or not

def main():
    train_loader = datagen.generate_dataloader(c.block_type, c.n_samples, c.batch_size)
    
    for i_epoch in range(c.n_epochs):
        loss_mean = train(i_epoch, train_loader)
        print(i_epoch, loss_mean)
        #test(i_epoch)
        #Model.weight_scheduler.step()



if __name__ == "__main__":
    main()