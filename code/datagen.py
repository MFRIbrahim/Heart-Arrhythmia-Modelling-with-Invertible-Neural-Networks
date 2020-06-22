import numpy as np
import torch
import torch.utils.data
import random

from mavb.ratios import *
from mavb.forward import simulate_type_1
from mavb.forward import simulate_type_2a
from mavb.forward import simulate_type_2b
from mavb.forward import simulate_type_2c
from mavb.forward import simulate_type_3

import config as c

def generate_block_pattern(block_type):
    n_Rwaves = np.random.randint(6,26)
    
    if block_type == "1":
        block_pattern_2 = [np.random.randint(1,8)]
        block_pattern_2_sum = block_pattern_2[0]
        while block_pattern_2_sum < n_Rwaves:
            current_block_ratio = block_pattern_2[-1]
            block_pattern_2.append(np.random.randint(
                np.amax([1,current_block_ratio-3]),
                np.amin([8,current_block_ratio+3])))
            block_pattern_2_sum += block_pattern_2[-1]
        return block_pattern_2
    
    if block_type == "2a":
        block_pattern_2 = [np.random.randint(1,8)]
        block_pattern_2_sum = block_pattern_2[0]
        while block_pattern_2_sum < n_Rwaves:
            current_block_ratio = block_pattern_2[-1]
            block_pattern_2.append(np.random.randint(
                np.amax([1,current_block_ratio-3]),
                np.amin([8,current_block_ratio+3])))
            block_pattern_2_sum += block_pattern_2[-1]
        block_pattern_1 = [1 for x in range(block_pattern_2_sum+len(block_pattern_2))]
        
        return block_pattern_1, block_pattern_2
    
    if block_type == "2b":
        block_pattern_3 = [1 for x in range(n_Rwaves)]
        block_pattern_2 = [np.random.randint(1,8)]
        block_pattern_2_sum = block_pattern_2[0]
        while block_pattern_2_sum < 2*n_Rwaves:
            current_block_ratio = block_pattern_2[-1]
            block_pattern_2.append(np.random.randint(
                np.amax([1,current_block_ratio-3]),
                np.amin([8,current_block_ratio+3])))
            block_pattern_2_sum += block_pattern_2[-1]
        return block_pattern_2, block_pattern_3
    
    if block_type == "2c":
        block_pattern_2 = [np.random.randint(0,2)]
        block_pattern_2_sum = 1
        while block_pattern_2_sum < n_Rwaves:
            block_pattern_2.append(np.random.randint(0,2))
            block_pattern_2_sum += 1
        block_pattern_1 = [np.random.randint(1,3)]
        block_pattern_1_sum = block_pattern_1[0]
        while block_pattern_1_sum < np.sum(block_pattern_2)+len(block_pattern_2):
            block_pattern_1.append(np.random.randint(1,3))
            block_pattern_1_sum += block_pattern_1[-1]
        return block_pattern_1, block_pattern_2
    
    if block_type == "3":
        block_pattern_3 = [np.random.randint(0,2)]
        block_pattern_3_sum = 1
        while block_pattern_3_sum < n_Rwaves:
            block_pattern_3.append(np.random.randint(0,2))
            block_pattern_3_sum += 1
        block_pattern_2 = [np.random.randint(0,2)]
        block_pattern_2_sum = 1
        while block_pattern_2_sum < np.sum(block_pattern_3)+len(block_pattern_3):
            block_pattern_2.append(np.random.randint(0,2))
            block_pattern_2_sum += 1
        block_pattern_1 = [np.random.randint(1,3)]
        block_pattern_1_sum = block_pattern_1[0]
        while block_pattern_1_sum < np.sum(block_pattern_2)+len(block_pattern_2):
            block_pattern_1.append(np.random.randint(1,3))
            block_pattern_1_sum += block_pattern_1[-1]
        return block_pattern_1, block_pattern_2, block_pattern_3
            
def block_pattern_to_one_hot(block_pattern):
    block_pattern = np.array(block_pattern)
    one_hot = np.zeros((block_pattern.size, 8))
    one_hot[np.arange(block_pattern.size),block_pattern] = 1
    return one_hot

def generate_dataloader(block_type, n_samples, batch_size):
    x = []
    y = []
    mix = False
    if block_type == "mix": 
        mix = True
    
    for i in range(n_samples):
        if mix:
            block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length+1)
        x_i = np.zeros(1402) # max blocks when 2:1 for all levels and using one hot plus constants
        y_i = np.zeros(24) # max intervals = max R waves - 1
        
        if block_type == "1":
            block_pattern = generate_block_pattern(block_type)
            intervals = simulate_type_1(block_pattern, atrial_cycle_length,
                                        conduction_constant)
            one_hot = block_pattern_to_one_hot(block_pattern)
            one_hot = one_hot.flatten()
            x_i[0:800:8] = 1
            x_i[800:800+one_hot.shape[0]] = one_hot
            x_i[800+one_hot.shape[0]:1400:8] = 1
            x_i[1400:1402] = [atrial_cycle_length, conduction_constant]
            if intervals.shape[0] > 24:
                intervals = intervals[0:24]
            y_i[0:intervals.shape[0]] = intervals
        
        if block_type == "2a":
            block_pattern = generate_block_pattern(block_type)
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                         conduction_constant)
            one_hot_1 = block_pattern_to_one_hot(block_pattern[0])
            one_hot_2 = block_pattern_to_one_hot(block_pattern[1])
            one_hot_1 = one_hot_1.flatten()
            one_hot_2 = one_hot_2.flatten()
            x_i[0:one_hot_1.shape[0]] = one_hot_1
            x_i[one_hot_1.shape[0]:800:8] = 1
            x_i[800:800+one_hot_2.shape[0]] = one_hot_2
            x_i[800+one_hot_2.shape[0]:1400:8] = 1
            x_i[1400:1402] = [atrial_cycle_length, conduction_constant]
            if intervals.shape[0] > 24:
                intervals = intervals[0:24]
            y_i[0:intervals.shape[0]] = intervals
            
        if block_type == "2b":
            block_pattern = generate_block_pattern(block_type)
            intervals = simulate_type_2b(block_pattern[0], atrial_cycle_length,
                                         conduction_constant)
            one_hot_1 = block_pattern_to_one_hot(block_pattern[0])
            one_hot_2 = block_pattern_to_one_hot(block_pattern[1])
            one_hot_1 = one_hot_1.flatten()
            one_hot_2 = one_hot_2.flatten()
            x_i[0:800:8] = 1
            x_i[800:800+one_hot_1.shape[0]] = one_hot_1
            x_i[800+one_hot_1.shape[0]:1200:8] = 1
            x_i[1200:1200+one_hot_2.shape[0]] = one_hot_2
            x_i[1200+one_hot_2.shape[0]:1400:8] = 1
            x_i[1400:1402] = [atrial_cycle_length, conduction_constant]
            if intervals.shape[0] > 24:
                intervals = intervals[0:24]
            y_i[0:intervals.shape[0]] = intervals
            
        if block_type == "2c":
            block_pattern = generate_block_pattern(block_type)
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1],
                                         atrial_cycle_length,
                                         conduction_constant)
            one_hot_1 = block_pattern_to_one_hot(block_pattern[0])
            one_hot_2 = block_pattern_to_one_hot(block_pattern[1])
            one_hot_1 = one_hot_1.flatten()
            one_hot_2 = one_hot_2.flatten()
            x_i[0:one_hot_1.shape[0]] = one_hot_1
            x_i[one_hot_1.shape[0]:800:8] = 1
            x_i[800:800+one_hot_2.shape[0]] = one_hot_2
            x_i[800+one_hot_2.shape[0]:1400:8] = 1
            x_i[1400:1402] = [atrial_cycle_length, conduction_constant]
            if intervals.shape[0] > 24:
                intervals = intervals[0:24]
            y_i[0:intervals.shape[0]] = intervals
            
        if block_type == "3":
            block_pattern = generate_block_pattern(block_type)
            intervals = simulate_type_3(block_pattern[0], block_pattern[1],
                                        block_pattern[2], atrial_cycle_length,
                                        conduction_constant)
            one_hot_1 = block_pattern_to_one_hot(block_pattern[0])
            one_hot_2 = block_pattern_to_one_hot(block_pattern[1])
            one_hot_3 = block_pattern_to_one_hot(block_pattern[2])
            one_hot_1 = one_hot_1.flatten()
            one_hot_2 = one_hot_2.flatten()
            one_hot_3 = one_hot_3.flatten()
            x_i[0:one_hot_1.shape[0]] = one_hot_1
            x_i[one_hot_1.shape[0]:800:8] = 1
            x_i[800:800+one_hot_2.shape[0]] = one_hot_2
            x_i[800+one_hot_2.shape[0]:1200:8] = 1
            x_i[1200:1200+one_hot_3.shape[0]] = one_hot_3
            x_i[1200+one_hot_3.shape[0]:1400:8] = 1
            x_i[1400:1402] = [atrial_cycle_length, conduction_constant]
            if intervals.shape[0] > 24:
                intervals = intervals[0:24]
            y_i[0:intervals.shape[0]] = intervals
            
            
        
        x.append(x_i)
        y.append(y_i)
    
    
    x = torch.tensor(x, dtype=torch.float32)
    x += c.norm_noise * torch.randn(x.shape[0], c.ndim_x)
    x = (x-torch.mean(x, dim=0))/torch.std(x, dim=0)
    
    y = torch.tensor(y, dtype=torch.float32)
    y += c.norm_noise * torch.randn(y.shape[0], c.ndim_y)
    y = (y-torch.mean(y, dim=0))/torch.std(y, dim=0)
    
    
    
    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))
    
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y), 
                                       batch_size=batch_size, shuffle=True)




    



        
            
