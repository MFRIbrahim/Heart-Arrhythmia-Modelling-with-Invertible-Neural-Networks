#------Data------#

ndim_x = 1402
ndim_y = 24
ndim_z = 24
ndim_pad_x = 598
ndim_pad_zy = 1952
x_noise = 1e-2
y_noise = 1e-2
zero_noise = 1e-3 
norm_noise = 1e-6

#------Model------#

n_blocks = 4
clamp = 2.0
init_scale = 1e-2
hidden_layer_sizes = None


#------Losses------#

lambd_l2 = 1
lambd_mmd_forw = 300
lambd_mmd_back = 400
mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]

#------Training------#

n_epochs = 1000
n_its_per_epoch = 1
block_type = "1"
n_samples = 8 #2**3
batch_size = 8
lr_init = 1.0e-4
final_decay = 0.02
adam_betas = (0.9, 0.95)
l2_weight_reg   = 1e-5







