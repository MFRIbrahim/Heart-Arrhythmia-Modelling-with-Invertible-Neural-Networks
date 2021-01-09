

# ------Model------ #

n_x_features = 352
n_cond_features = 24
n_blocks = 5
clamp = 2.0
init_scale = 0.01

# for cINN

n_hidden_layer_size = 1024

# for rcINN

rnn_layers = 2
hidden_size = 64

# ------Training------ #

n_epochs = 100
n_iterations = 200
batch_size = 512
lr = 0.001
gamma = 0.95
