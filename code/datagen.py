import numpy as np
import torch
import torch.utils.data
import random
import copy


from mavb.forward import simulate_type_1
from mavb.forward import simulate_type_2a
from mavb.forward import simulate_type_2b
from mavb.forward import simulate_type_2c
from mavb.forward import simulate_type_3



def block_pattern_to_one_hot(block_pattern, length):
    block_pattern = np.array(block_pattern)
    one_hot = np.zeros((block_pattern.size, length))
    one_hot[np.arange(block_pattern.size),block_pattern] = 1
    return one_hot


def find_nearest(array, value):
    array = np.array(array)
    idx = np.argmin((np.abs(array - value)), axis=0)
    return idx


def y_to_cond(matching, seq_len, y):
    cond = np.zeros((2,seq_len))
    for i in range(len(matching)-1):
        start = matching[i]
        end = matching[i+1]
        cond[1,start] = y[i]
        if end-start > 1:
            cond[0,start+1:end+1] = y[i]
        else:
            cond[0,end] = y[i]
        if i == 0:
            cond[0,start] = y[i]
            cond[1,start] = 0
    return cond



def check_block_pattern_splitter(block_pattern, n_Rwaves, btype):
    res_bp = []
    res_type = []
    block_pattern_res = copy.deepcopy(block_pattern)
    
    if btype == "1" or btype == "2a":
        differences = []
        for i in range(len(block_pattern_res[1]) - 1):
            differences.append(abs(block_pattern_res[1][i]-block_pattern_res[1][i+1]))
        if len(differences) == 0:
            differences = [0]
        if max(differences) <= 3:
            block_sum = sum(block_pattern_res[1])
            if block_sum >= n_Rwaves and block_sum - block_pattern_res[1][-1] < n_Rwaves:
                res_bp.append(copy.deepcopy([block_pattern_res[1]]))
                res_type.append("1") if btype == "1" else res_type.append("2a")
    
    if btype == "2b":
        differences = []
        for i in range(len(block_pattern_res[1]) - 1):
            differences.append(abs(block_pattern_res[1][i]-block_pattern_res[1][i+1]))
        if len(differences) == 0:
            differences = [0]
        if max(differences) <= 3:
            block_sum = sum(block_pattern_res[1])
            if block_sum >= (2 * n_Rwaves - 1) and block_sum - block_pattern_res[1][-1] <= (2 * n_Rwaves - 1):
                res_bp.append(copy.deepcopy([block_pattern_res[1]]))
                res_type.append("2b")
    
    if btype == "2c":
        block_sum_0 = sum(block_pattern_res[0])
        block_sum_1 = len(block_pattern_res[1])
        if block_sum_1 == n_Rwaves:
            if block_pattern_res[1][-1] == 0:
                if block_sum_0 == n_Rwaves + sum(block_pattern_res[1]):
                    res_bp.append(copy.deepcopy([block_pattern_res[0], block_pattern_res[1]]))
                    res_type.append("2c")
            if block_pattern_res[1][-1] == 1:
                if (block_sum_0 == n_Rwaves + sum(block_pattern_res[1]) or 
                    block_sum_0 == n_Rwaves + sum(block_pattern_res[1]) - 1):
                    res_bp.append(copy.deepcopy([block_pattern_res[0], block_pattern_res[1]]))
                    res_type.append("2c")
    
    if btype == "3":
        block_sum_0 = sum(block_pattern_res[0])
        block_sum_1 = len(block_pattern_res[1])
        block_sum_2 = len(block_pattern_res[2])
        if block_sum_2 == n_Rwaves: 
            if block_pattern_res[2][-1] == 0:
                if block_sum_1 == n_Rwaves + sum(block_pattern_res[2]):
                    if block_pattern_res[1][-1] == 0:
                        if block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
                    if block_pattern_res[1][-1] == 1:
                        if (block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]) or
                            block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]) - 1):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
            if block_pattern_res[2][-1] == 1:
                if (block_sum_1 == n_Rwaves + sum(block_pattern_res[2]) or 
                    block_sum_1 == n_Rwaves + sum(block_pattern_res[2]) - 1):
                    if block_pattern_res[1][-1] == 0 and block_sum_1 == n_Rwaves + sum(block_pattern_res[2]):
                        if block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
                    if block_pattern_res[1][-1] == 0 and block_sum_1 == n_Rwaves + sum(block_pattern_res[2]) - 1:
                         if block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) - 1 + sum(block_pattern_res[1]):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
                    if block_pattern_res[1][-1] == 1 and block_sum_1 == n_Rwaves + sum(block_pattern_res[2]):
                        if (block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]) or
                            block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]) - 1):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
                    if block_pattern_res[1][-1] == 1 and block_sum_1 == n_Rwaves + sum(block_pattern_res[2]) - 1:
                        if (block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) - 1 + sum(block_pattern_res[1]) or
                            block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) - 1 + sum(block_pattern_res[1]) - 1):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
      
    
    return res_bp, res_type
    


def check_block_pattern_alt(block_pattern, n_Rwaves):
    res_bp = []
    res_type = []
    block_pattern_res = copy.deepcopy(block_pattern)
    
    
    if (len(block_pattern_res[0]) != 0 and len(block_pattern_res[2]) != 0 and len(np.where(np.array(block_pattern_res[0])==0)[0]) == len(block_pattern_res[0]) and 
        len(np.where(np.array(block_pattern_res[2])==0)[0]) == len(block_pattern_res[2]) and 
        np.any(block_pattern_res[1]) and min(block_pattern_res[1]) >= 1  and max(block_pattern_res[1]) <= 7):
        differences = []
        for i in range(len(block_pattern_res[1]) - 1):
            differences.append(abs(block_pattern_res[1][i]-block_pattern_res[1][i+1]))
        if len(differences) == 0:
            differences = [0]
        if max(differences) <= 3:
            block_sum_0 = len(block_pattern_res[0])
            block_sum_1 = sum(block_pattern_res[1])
            block_sum_2 = len(block_pattern_res[2])
            if (((block_sum_0 == n_Rwaves + len(block_pattern_res[1]) - 1 and block_sum_1 >= n_Rwaves) or 
                 (block_sum_0 == n_Rwaves + len(block_pattern_res[1]) and block_sum_1 == n_Rwaves)) and
                (block_sum_1 - block_pattern_res[1][-1]) < n_Rwaves and 
                block_sum_2 == n_Rwaves):
                res_bp.append(copy.deepcopy([block_pattern_res[0], block_pattern_res[1], block_pattern_res[2]]))
                res_type.append("1")
                
            
    if (len(block_pattern_res[0]) != 0 and len(block_pattern_res[2]) != 0 and len(np.where(np.array(block_pattern_res[2])==0)[0]) == len(block_pattern_res[2]) and 
        len(np.where(np.array(block_pattern_res[0])==1)[0]) == len(block_pattern_res[0]) and 
        np.any(block_pattern_res[1]) and min(block_pattern_res[1]) >= 1  and max(block_pattern_res[1]) <= 7):
        differences = []
        for i in range(len(block_pattern_res[1]) - 1):
            differences.append(abs(block_pattern_res[1][i]-block_pattern_res[1][i+1]))
        if len(differences) == 0:
            differences = [0]
        if max(differences) <= 3:
            block_sum_0 = sum(block_pattern_res[0])
            block_sum_1 = sum(block_pattern_res[1])
            block_sum_2 = len(block_pattern_res[2])
            if (((block_sum_0 == n_Rwaves + len(block_pattern_res[1]) - 1 and block_sum_1 >= n_Rwaves) or 
                 (block_sum_0 == n_Rwaves + len(block_pattern_res[1]) and block_sum_1 == n_Rwaves)) and
                (block_sum_1 - block_pattern_res[1][-1]) < n_Rwaves and 
                block_sum_2 == n_Rwaves):
                res_bp.append(copy.deepcopy([block_pattern_res[0], block_pattern_res[1], block_pattern_res[2]]))
                res_type.append("2a")

    if (len(block_pattern_res[0]) != 0 and len(block_pattern_res[2]) != 0 and len(np.where(np.array(block_pattern_res[0])==0)[0]) == len(block_pattern_res[0]) and 
        len(np.where(np.array(block_pattern_res[2])==1)[0]) == len(block_pattern_res[2]) and 
        np.any(block_pattern_res[1]) and min(block_pattern_res[1]) >= 1  and max(block_pattern_res[1]) <= 7):
        differences = []
        for i in range(len(block_pattern_res[1]) - 1):
            differences.append(abs(block_pattern_res[1][i]-block_pattern_res[1][i+1]))
        if len(differences) == 0:
            differences = [0]
        if max(differences) <= 3:
            block_sum_0 = len(block_pattern_res[0])
            block_sum_1 = sum(block_pattern_res[1])
            block_sum_2 = sum(block_pattern_res[2])
            if block_sum_2 == n_Rwaves:
                if block_sum_1 == 2 * n_Rwaves - 1:
                    if (block_sum_0 == 2 * n_Rwaves - 1 + len(block_pattern_res[1]) or
                        block_sum_0 == 2 * n_Rwaves - 1 + len(block_pattern_res[1]) - 1):
                        res_bp.append(copy.deepcopy([block_pattern_res[0], block_pattern_res[1], block_pattern_res[2]]))
                        res_type.append("2b")
                if block_sum_1 == 2 * n_Rwaves:
                    if (block_sum_0 == 2 * n_Rwaves + len(block_pattern_res[1]) or
                        block_sum_0 == 2 * n_Rwaves + len(block_pattern_res[1]) - 1 or
                        block_sum_0 == 2 * n_Rwaves - 1 + len(block_pattern_res[1]) - 1):
                        res_bp.append(copy.deepcopy([block_pattern_res[0], block_pattern_res[1], block_pattern_res[2]]))
                        res_type.append("2b")
                if block_sum_1 > 2 * n_Rwaves:
                    if (block_sum_1 - block_pattern_res[1][-1] == 2 * n_Rwaves - 1):
                        if block_sum_0 == 2 * n_Rwaves - 1 + len(block_pattern_res[1]):
                            res_bp.append(copy.deepcopy([block_pattern_res[0], block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("2b")                                 
                    if block_sum_1 - block_pattern_res[1][-1] < 2 * n_Rwaves - 1:
                        if (block_sum_0 == 2 * n_Rwaves + len(block_pattern_res[1]) - 1 or
                            block_sum_0 == 2 * n_Rwaves - 1 + len(block_pattern_res[1]) - 1):
                            res_bp.append(copy.deepcopy([block_pattern_res[0], block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("2b") 
            
                
    
    if (len(block_pattern_res[2]) != 0 and len(np.where(np.array(block_pattern_res[2])==0)[0]) == len(block_pattern_res[2]) and 
        np.any(block_pattern_res[0]) and len(block_pattern_res[1]) != 0 and
        min(block_pattern_res[0]) >= 1 and max(block_pattern_res[0]) <= 2 and min(block_pattern_res[1]) >= 0 and max(block_pattern_res[1]) <= 1):
        block_sum_0 = sum(block_pattern_res[0])
        block_sum_1 = len(block_pattern_res[1])
        block_sum_2 = len(block_pattern_res[2])
        if block_sum_1 == n_Rwaves and block_sum_2 == n_Rwaves:
            if block_pattern_res[1][-1] == 0:
                if block_sum_0 == n_Rwaves + sum(block_pattern_res[1]):
                    res_bp.append(copy.deepcopy([block_pattern_res[0], block_pattern_res[1], block_pattern_res[2]]))
                    res_type.append("2c")
            if block_pattern_res[1][-1] == 1:
                if (block_sum_0 == n_Rwaves + sum(block_pattern_res[1]) or
                    block_sum_0 == n_Rwaves + sum(block_pattern_res[1]) - 1):
                    res_bp.append(copy.deepcopy([block_pattern_res[0], block_pattern_res[1], block_pattern_res[2]]))
                    res_type.append("2c")
    
    if (np.any(block_pattern_res[0]) and len(block_pattern_res[1]) != 0 and len(block_pattern_res[2]) != 0 and
        min(block_pattern_res[0]) >= 1 and max(block_pattern_res[0]) <= 2 and min(block_pattern_res[1]) >= 0 and max(block_pattern_res[1]) <= 1 and
        min(block_pattern_res[2]) >= 0 and max(block_pattern_res[2]) <= 1):
        block_sum_0 = sum(block_pattern_res[0])
        block_sum_1 = len(block_pattern_res[1])
        block_sum_2 = len(block_pattern_res[2])
        if block_sum_2 == n_Rwaves: 
            if block_pattern_res[2][-1] == 0:
                if block_sum_1 == n_Rwaves + sum(block_pattern_res[2]):
                    if block_pattern_res[1][-1] == 0:
                        if block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
                    if block_pattern_res[1][-1] == 1:
                        if (block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]) or
                            block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]) - 1):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
            if block_pattern_res[2][-1] == 1:
                if (block_sum_1 == n_Rwaves + sum(block_pattern_res[2]) or 
                    block_sum_1 == n_Rwaves + sum(block_pattern_res[2]) - 1):
                    if block_pattern_res[1][-1] == 0 and block_sum_1 == n_Rwaves + sum(block_pattern_res[2]):
                        if block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
                    if block_pattern_res[1][-1] == 0 and block_sum_1 == n_Rwaves + sum(block_pattern_res[2]) - 1:
                         if block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) - 1 + sum(block_pattern_res[1]):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
                    if block_pattern_res[1][-1] == 1 and block_sum_1 == n_Rwaves + sum(block_pattern_res[2]):
                        if (block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]) or
                            block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) + sum(block_pattern_res[1]) - 1):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
                    if block_pattern_res[1][-1] == 1 and block_sum_1 == n_Rwaves + sum(block_pattern_res[2]) - 1:
                        if (block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) - 1 + sum(block_pattern_res[1]) or
                            block_sum_0 == n_Rwaves + sum(block_pattern_res[2]) - 1 + sum(block_pattern_res[1]) - 1):
                            res_bp.append(copy.deepcopy([block_pattern_res[0],block_pattern_res[1], block_pattern_res[2]]))
                            res_type.append("3")
        
    return res_bp, res_type


def seq_to_block_pattern(x):
    block_pattern = [[],[],[]]
    idx = 0
    
    status_1 = [x[idx][0] + 1, 1 if x[idx][0] == 0 else x[idx][0]]
    status_2 = [x[idx][1] + 1, 1 if x[idx][1] == 0 else x[idx][1]]
    status_3 = [x[idx][2] + 1, 1 if x[idx][2] == 0 else x[idx][2]]
    
    block_pattern[0].append(x[idx][0])
    block_pattern[1].append(x[idx][1])
    block_pattern[2].append(x[idx][2])
    
    while True:
        if idx == len(x) - 1:
            break
        
        update_1 = False
        update_2 = False
        update_3 = False
        
        change_1_0 = max(0, status_1[0] - 1)
        change_1_1 = max(0, status_1[1] - 1)
        change_2_0 = max(0, status_2[0] - 1 if status_1[1] != 0 else status_2[0])
        change_2_1 = max(0, status_2[1] - 1 if status_1[1] != 0 else status_2[1])
        change_3_0 = max(0, status_3[0] - 1 if status_2[1] != 0 else status_3[0])
        change_3_1 = max(0, status_3[1] - 1 if status_2[1] != 0 else status_3[1])
        
        status_1[0] = change_1_0
        status_1[1] = change_1_1
        status_2[0] = change_2_0
        status_2[1] = change_2_1
        status_3[0] = change_3_0
        status_3[1] = change_3_1
        
        idx += 1
        
        if status_1 == [0,0]:
            update_1 = True
        if status_2 == [0,0] and (status_1[1] != 0 or update_1):
            update_2 = True
        if status_3 == [0,0] and (status_1[1] != 0 or update_1) and (status_2[1] != 0 or update_2):
            update_3 = True
        
        if update_1:
            block_pattern[0].append(x[idx][0])
            status_1 = [x[idx][0] + 1, 1 if x[idx][0] == 0 else x[idx][0]]
        if update_2:
            block_pattern[1].append(x[idx][1])
            status_2 = [x[idx][1] + 1, 1 if x[idx][1] == 0 else x[idx][1]]
        if update_3:
            block_pattern[2].append(x[idx][2])
            status_3 = [x[idx][2] + 1, 1 if x[idx][2] == 0 else x[idx][2]] 
        
    return block_pattern


def block_pattern_to_seq(block_pattern):
    x = []
    idx_1 = 0
    idx_2 = 0
    idx_3 = 0
    matching = [0]
    matching_counter = 0
    
    status_1 = [block_pattern[0][idx_1] + 1, 1 if block_pattern[0][idx_1] == 0 else block_pattern[0][idx_1]]
    status_2 = [block_pattern[1][idx_2] + 1, 1 if block_pattern[1][idx_2] == 0 else block_pattern[1][idx_2]]
    status_3 = [block_pattern[2][idx_3] + 1, 1 if block_pattern[2][idx_3] == 0 else block_pattern[2][idx_3]]
    
    while True:
        x.append([block_pattern[0][idx_1], block_pattern[1][idx_2], block_pattern[2][idx_3]])
        
        if idx_3 == len(block_pattern[2]) - 1:
            break
        
        matching_counter += 1
        
        update_1 = False
        update_2 = False
        update_3 = False
        
        change_1_0 = max(0, status_1[0] - 1)
        change_1_1 = max(0, status_1[1] - 1)
        change_2_0 = max(0, status_2[0] - 1 if status_1[1] != 0 else status_2[0])
        change_2_1 = max(0, status_2[1] - 1 if status_1[1] != 0 else status_2[1])
        change_3_0 = max(0, status_3[0] - 1 if status_2[1] != 0 else status_3[0])
        change_3_1 = max(0, status_3[1] - 1 if status_2[1] != 0 else status_3[1])
        
        status_1[0] = change_1_0
        status_1[1] = change_1_1
        status_2[0] = change_2_0
        status_2[1] = change_2_1
        status_3[0] = change_3_0
        status_3[1] = change_3_1
        
        if status_1 == [0,0]:
            idx_1 += 1
            update_1 = True
        if status_2 == [0,0] and (status_1[1] != 0 or update_1):
            idx_2 += 1
            update_2 = True
        if status_3 == [0,0] and (status_1[1] != 0 or update_1) and (status_2[1] != 0 or update_2):
            idx_3 += 1
            update_3 = True
            matching.append(matching_counter)
        
        if update_1:
            status_1 = [block_pattern[0][idx_1] + 1, 1 if block_pattern[0][idx_1] == 0 else block_pattern[0][idx_1]]
        if update_2:
            status_2 = [block_pattern[1][idx_2] + 1, 1 if block_pattern[1][idx_2] == 0 else block_pattern[1][idx_2]]
        if update_3:
            status_3 = [block_pattern[2][idx_3] + 1, 1 if block_pattern[2][idx_3] == 0 else block_pattern[2][idx_3]]
        
    return x, matching


def signals_to_bp(signals, n_Rwaves):
    limit = np.where(np.array(signals[0]) == 1)[0][-1] + 1
    candidate1 = []
    candidate2 = []
    candidate3 = []

    counter = 0
    sub1 = []
    for i in range(limit):
        if signals[0][i] == 1:
            counter += 1
            if i == limit - 1:
                sub1.append(counter)
        if signals[0][i] == 0:
            sub1.append(counter)
            counter = 0
    if max(sub1) <= 7 and min(sub1) >= 1:
        candidate1.append(sub1)
    sub2 = []
    for i in range(limit):
        if i == limit - 1:
            sub2.append(1)
            break
        if signals[0][i] == 1 and signals[0][i+1] == 0:
            sub2.append(1)
        if signals[0][i] == 1 and signals[0][i+1] == 1:
            sub2.append(0)
    if sub2 not in candidate1:
        candidate1.append(sub2)
    sub3 = copy.deepcopy(sub2)
    sub3[-1] = 0
    if sub3 not in candidate1:
        candidate1.append(sub3)
    
    idx_1 = np.where(np.array(signals[0]) == 1)[0]
    counter = 0
    sub1 = []
    vary = False
    for i in range(len(idx_1)):
        if signals[1][idx_1[i]] == 1:
            counter += 1
            if i == len(idx_1) - 1:
                sub1.append(counter)
                vary = True      
        if signals[1][idx_1[i]] == 0:
            sub1.append(counter)
            counter = 0
    if not vary:
        if max(sub1) <= 7 and min(sub1) >= 1:
            candidate2.append(sub1)
    if vary:
        if len(sub1) > 1 and max(sub1) <= 7 and min(sub1) >= 1:
            low_limit = np.amax([1, sub1[-2] - 3])
            up_limit = np.amin([7, sub1[-2] + 3])
            valid_range = np.linspace(low_limit, up_limit, up_limit - low_limit + 1, dtype='int16')
            for val in valid_range:
                if val >= sub1[-1]:
                    sub_alt = copy.deepcopy(sub1[:-1])
                    sub_alt += [val]
                    candidate2.append(sub_alt)
        if len(sub1) == 1 and max(sub1) <= 7 and min(sub1) >= 1:
            low_limit = sub1[0]
            up_limit = 7
            valid_range = np.linspace(low_limit, up_limit, up_limit - low_limit + 1, dtype='int16')
            for val in valid_range:
                sub_alt = copy.deepcopy(sub1[:-1])
                sub_alt += [val]
                candidate2.append(sub_alt)
            
    sub2 = []
    alt = True
    for i in range(len(idx_1)):
        if i == len(idx_1) - 1 and signals[1][idx_1[i]] == 1:
            sub2.append(1)
            break
        if i == len(idx_1) - 1 and signals[1][idx_1[i]] == 0:
            alt = False
            break
        if signals[1][idx_1[i]] == 1 and signals[1][idx_1[i+1]] == 0:
            sub2.append(1)
        if signals[1][idx_1[i]] == 1 and signals[1][idx_1[i+1]] == 1:
            sub2.append(0)
    if sub2 not in candidate2:
        candidate2.append(sub2)
    if alt:
        sub3 = copy.deepcopy(sub2)
        sub3[-1] = 0
        if sub3 not in candidate2:
            candidate2.append(sub3)
    
    idx_2 = np.where(np.array(signals[1]) == 1)[0]
    sub2 = []
    alt = True
    for i in range(len(idx_2)):
        if i == len(idx_2) - 1 and signals[2][idx_2[i]] == 1:
            sub2.append(1)
            break
        if i == len(idx_2) - 1 and signals[2][idx_2[i]] == 0:
            alt = False
            break
        if signals[2][idx_2[i]] == 1 and signals[2][idx_2[i+1]] == 0:
            sub2.append(1)
        if signals[2][idx_2[i]] == 1 and signals[2][idx_2[i+1]] == 1:
            sub2.append(0)
    if sub2 not in candidate3:
        candidate3.append(sub2)
    if alt:
        sub3 = copy.deepcopy(sub2)
        sub3[-1] = 0
        if sub3 not in candidate3:
            candidate3.append(sub3)
    
    res = []
    
    for i in range(len(candidate1)):
        for j in range(len(candidate2)):
            for k in range(len(candidate3)):
                bp, bp_type = check_block_pattern_alt([candidate1[i], candidate2[j], candidate3[k]], n_Rwaves)
                if len(bp) != 0:
                    res.append((bp, bp_type))
    
    return res
    
    
        

def correct_bp(bp, bp_type, n_Rwaves):
    bp_res = copy.deepcopy(bp)
    if bp_type == "1":
        if sum(bp_res[1]) > n_Rwaves:
            bp_res[1][-1] -= abs(sum(bp_res[1]) - n_Rwaves)
    if bp_type == "2a":
        if sum(bp_res[1]) > n_Rwaves:
            bp_res[1][-1] -= abs(sum(bp_res[1]) - n_Rwaves)
    if bp_type == "2b":
        if sum(bp_res[1]) == 2 * n_Rwaves and len(bp_res[0]) == 2 * n_Rwaves - 1 + len(bp_res[1]) - 1:
            bp_res[1][-1] -= 1
            return bp_res
        if sum(bp_res[1]) > 2 * n_Rwaves:
            if sum(bp_res[1]) - bp_res[1][-1] < 2 * n_Rwaves - 1:
                if len(bp_res[0]) == 2 * n_Rwaves + len(bp_res[1]) - 1:
                    bp_res[1][-1] -= abs(sum(bp_res[1]) - 2 * n_Rwaves)
                    return bp_res
                if len(bp_res[0]) == 2 * n_Rwaves - 1 + len(bp_res[1]) - 1:
                    bp_res[1][-1] -= abs(sum(bp_res[1]) - (2 * n_Rwaves - 1))
                    return bp_res
            if sum(bp_res[1]) - bp_res[1][-1] == 2 * n_Rwaves - 1:
                bp_res[1][-1] -= abs(sum(bp_res[1]) - 2 * n_Rwaves)
    return bp_res


def bp_to_signals(bp, bp_type, n_Rwaves, fill=True):
    if bp_type == "1":
        lvl1 = []
        for b in bp[0]:
            lvl1 += [1]
        lvl2 = []
        for b in bp[1]:
            lvl2 += [1 for i in range(b)] + [0]
        lvl3 = []
        for b in bp[2]:
            lvl3 += [1]
        idx = np.where(np.array(lvl2) == 0)[0]
        for idx_i in idx:
            lvl3.insert(idx_i, 0)
    
    if bp_type == "2a":
        lvl1 = []
        for b in bp[0]:
            lvl1 += [1 for i in range(b)] + [0]
        lvl2 = []
        for b in bp[1]:
            lvl2 += [1 for i in range(b)] + [0]
        idx = np.where(np.array(lvl1) == 0)[0]
        for idx_i in idx:
            lvl2.insert(idx_i, 0)
        lvl3 = []
        for b in bp[2]:
            lvl3 += [1]
        idx = np.where(np.array(lvl2) == 0)[0]
        for idx_i in idx:
            lvl3.insert(idx_i, 0)
        
    if bp_type == "2b":
        lvl1 = []
        for b in bp[0]:
            lvl1 += [1]
        lvl2 = []
        for b in bp[1]:
            lvl2 += [1 for i in range(b)] + [0]
        lvl3 = []
        for b in bp[2]:
            lvl3 += [1 for i in range(b)] + [0]
        idx = np.where(np.array(lvl2) == 0)[0]
        for idx_i in idx:
            lvl3.insert(idx_i, 0)
    
    if bp_type == "2c":
        lvl1 = []
        for b in bp[0]:
            lvl1 += [1 for i in range(b)] + [0]
        lvl2 = []
        for b in bp[1]:
            if b == 0:
                lvl2 += [1]
            else:
                lvl2 += [1 for i in range(b)] + [0]
        idx = np.where(np.array(lvl1) == 0)[0]
        for idx_i in idx:
            lvl2.insert(idx_i, 0)
        lvl3 = []
        for b in bp[2]:
            lvl3 += [1]
        idx = np.where(np.array(lvl2) == 0)[0]
        for idx_i in idx:
            lvl3.insert(idx_i, 0)
        
    if bp_type == "3":
        lvl1 = []
        for b in bp[0]:
            lvl1 += [1 for i in range(b)] + [0]
        lvl2 = []
        for b in bp[1]:
            if b == 0:
                lvl2 += [1]
            else:
                lvl2 += [1 for i in range(b)] + [0]
        idx = np.where(np.array(lvl1) == 0)[0]
        for idx_i in idx:
            lvl2.insert(idx_i, 0)
        lvl3 = []
        for b in bp[2]:
            if b == 0:
                lvl3 += [1]
            else:
                lvl3 += [1 for i in range(b)] + [0]
        idx = np.where(np.array(lvl2) == 0)[0]
        for idx_i in idx:
            lvl3.insert(idx_i, 0)
    
    if fill:
        lvl1 += [0 for i in range(200 - len(lvl1))]
        lvl2 += [0 for i in range(200 - len(lvl2))]
        lvl3 += [0 for i in range(200 - len(lvl3))]
    
    else:
        lvl1 = lvl1[:np.where(np.array(lvl1) == 1)[0][-1] + 1]
        lvl2 = lvl2[:len(lvl1)]
        lvl3 = lvl3[:len(lvl1)]
    
    return [lvl1, lvl2, lvl3]


def generate_block_pattern_alt(block_type, n_Rwaves):
    
    if block_type == "1":
        block_pattern_2 = [np.random.randint(1,8)]
        block_pattern_2_sum = block_pattern_2[0]
        while block_pattern_2_sum < n_Rwaves:
            current_block_ratio = block_pattern_2[-1]
            block_pattern_2.append(np.random.randint(
                np.amax([1,current_block_ratio-3]),
                np.amin([8,current_block_ratio+4])))
            block_pattern_2_sum += block_pattern_2[-1]
        if block_pattern_2_sum == n_Rwaves:
            block_pattern_1 = random.choice([[0 for x in range(n_Rwaves + len(block_pattern_2))],
                                             [0 for x in range(n_Rwaves + len(block_pattern_2) - 1)]])
        if block_pattern_2_sum > n_Rwaves:
            block_pattern_1 = [0 for x in range(n_Rwaves + len(block_pattern_2) - 1)]
        block_pattern_3 = [0 for x in range(n_Rwaves)]
        return [block_pattern_1, block_pattern_2, block_pattern_3]
    
    if block_type == "2a":
        block_pattern_2 = [np.random.randint(1,8)]
        block_pattern_2_sum = block_pattern_2[0]
        while block_pattern_2_sum < n_Rwaves:
            current_block_ratio = block_pattern_2[-1]
            block_pattern_2.append(np.random.randint(
                np.amax([1,current_block_ratio-3]),
                np.amin([8,current_block_ratio+4])))
            block_pattern_2_sum += block_pattern_2[-1]
        if block_pattern_2_sum == n_Rwaves:
            block_pattern_1 = random.choice([[1 for x in range(n_Rwaves + len(block_pattern_2))],
                                             [1 for x in range(n_Rwaves + len(block_pattern_2) - 1)]])
        if block_pattern_2_sum > n_Rwaves:
            block_pattern_1 = [1 for x in range(n_Rwaves + len(block_pattern_2) - 1)]
        block_pattern_3 = [0 for x in range(n_Rwaves)]
        return [block_pattern_1, block_pattern_2, block_pattern_3]
 
    
    if block_type == "2b":
        while True:
            bp_2_choice = random.choice([1,2])
            block_pattern_2 = [np.random.randint(1,8)]
            block_pattern_2_sum = block_pattern_2[0]
            while block_pattern_2_sum < 2 * n_Rwaves - 1:
                current_block_ratio = block_pattern_2[-1]
                block_pattern_2.append(np.random.randint(
                    np.amax([1,current_block_ratio-3]),
                    np.amin([8,current_block_ratio+4])))
                block_pattern_2_sum += block_pattern_2[-1]
            if block_pattern_2_sum == 2 * n_Rwaves - 1 and bp_2_choice == 1:
                block_pattern_1 = random.choice([[0 for x in range(2 * n_Rwaves - 1 + len(block_pattern_2))],
                                                 [0 for x in range(2 * n_Rwaves - 1 + len(block_pattern_2) - 1)]])
                break
            if block_pattern_2_sum == 2 * n_Rwaves and bp_2_choice == 2:
                block_pattern_1 = random.choice([[0 for x in range(2 * n_Rwaves + len(block_pattern_2))],
                                                 [0 for x in range(2 * n_Rwaves + len(block_pattern_2) - 1)],
                                                 [0 for x in range(2 * n_Rwaves - 1 + len(block_pattern_2) - 1)]])
                break
            if block_pattern_2_sum > 2 * n_Rwaves and bp_2_choice == 2:
                if block_pattern_2_sum - block_pattern_2[-1] == 2 * n_Rwaves - 1:
                    block_pattern_1 = [0 for x in range(2 * n_Rwaves - 1 + len(block_pattern_2))]
                    break
                if block_pattern_2_sum - block_pattern_2[-1] < 2 * n_Rwaves - 1:
                    block_pattern_1 = random.choice([[0 for x in range(2 * n_Rwaves - 1 + len(block_pattern_2) - 1)],
                                                     [0 for x in range(2 * n_Rwaves + len(block_pattern_2) - 1)]])
                    break
        block_pattern_3 = [1 for x in range(n_Rwaves)]
        return [block_pattern_1, block_pattern_2, block_pattern_3]
    
    if block_type == "2c":
        while True:
            block_pattern_2 = [np.random.randint(0,2)]
            block_pattern_2_sum = 1
            while block_pattern_2_sum != n_Rwaves:
                block_pattern_2.append(np.random.randint(0,2))
                block_pattern_2_sum += 1
            bp_choice_1 = random.choice([1,2])
            block_pattern_1 = [np.random.randint(1,3)]
            block_pattern_1_sum = block_pattern_1[0]
            repeat = False
            while True:
                if block_pattern_2[-1] == 0:
                    if block_pattern_1_sum == n_Rwaves + sum(block_pattern_2):
                        break
                    if block_pattern_1_sum > n_Rwaves + sum(block_pattern_2):
                        repeat = True
                        break
                if block_pattern_2[-1] == 1:
                    if block_pattern_1_sum == n_Rwaves + sum(block_pattern_2) and bp_choice_1 == 1:
                        break
                    if block_pattern_1_sum > n_Rwaves + sum(block_pattern_2) and bp_choice_1 == 1:
                        repeat = True
                        break
                    if block_pattern_1_sum == n_Rwaves + sum(block_pattern_2) - 1 and bp_choice_1 == 2:
                        break
                    if block_pattern_1_sum > n_Rwaves + sum(block_pattern_2) - 1 and bp_choice_1 == 2:
                        repeat = True
                        break
                block_pattern_1.append(np.random.randint(1,3))
                block_pattern_1_sum += block_pattern_1[-1]
            if not repeat:
                break     
        block_pattern_3 = [0 for x in range(n_Rwaves)]
        return [block_pattern_1, block_pattern_2, block_pattern_3]
    
    if block_type == "3":
        while True:
            block_pattern_3 = [np.random.randint(0,2)]
            block_pattern_3_sum = 1
            while block_pattern_3_sum != n_Rwaves:
                block_pattern_3.append(np.random.randint(0,2))
                block_pattern_3_sum += 1
            bp_choice_2 = random.choice([1,2])
            block_pattern_2 = [np.random.randint(0,2)]
            block_pattern_2_sum = 1
            while True:
                if block_pattern_3[-1] == 0:
                    if block_pattern_2_sum == n_Rwaves + sum(block_pattern_3):
                        break
                if block_pattern_3[-1] == 1:
                    if block_pattern_2_sum == n_Rwaves + sum(block_pattern_3) and bp_choice_2 == 1:
                        break
                    if block_pattern_2_sum == n_Rwaves + sum(block_pattern_3) - 1 and bp_choice_2 == 2:
                        break
                block_pattern_2.append(np.random.randint(0,2))
                block_pattern_2_sum += 1
            bp_choice_1 = random.choice([1,2])
            block_pattern_1 = [np.random.randint(1,3)]
            block_pattern_1_sum = block_pattern_1[0]
            repeat = False
            while True:
                if block_pattern_2[-1] == 0:
                    if block_pattern_1_sum == block_pattern_2_sum + sum(block_pattern_2):
                        break
                    if block_pattern_1_sum > block_pattern_2_sum + sum(block_pattern_2):
                        repeat = True
                        break
                if block_pattern_2[-1] == 1:
                    if block_pattern_1_sum == block_pattern_2_sum + sum(block_pattern_2) and bp_choice_1 == 1:
                        break
                    if block_pattern_1_sum > block_pattern_2_sum + sum(block_pattern_2) and bp_choice_1 == 1:
                        repeat = True
                        break
                    if block_pattern_1_sum == block_pattern_2_sum + sum(block_pattern_2) - 1 and bp_choice_1 == 2:
                        break
                    if block_pattern_1_sum > block_pattern_2_sum + sum(block_pattern_2) - 1 and bp_choice_1 == 2:
                        repeat = True
                        break
                block_pattern_1.append(np.random.randint(1,3))
                block_pattern_1_sum += block_pattern_1[-1]
            if not repeat:
                break
        return [block_pattern_1, block_pattern_2, block_pattern_3]



def get_signals_sequence_batch(batch_size, test=False, btype=0):
    x = []
    y = []
    
    for i in range(batch_size):
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        #block_type = btype
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)
        block_pattern_extra = copy.deepcopy(block_pattern)
        
        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)
        
        block_pattern = correct_bp(block_pattern, block_type, n_Rwaves)
        signals = np.array(bp_to_signals(block_pattern, block_type, n_Rwaves, fill=False))
        x_i = np.zeros(194)
        x_i[signals.shape[1]-6] = 1
        x.append(x_i)
        
        y_i = np.zeros(24)
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        y.append(y_i)
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    x += 0.1 * torch.randn(x.shape[0], x.shape[1])
    
    y_mean = np.loadtxt('y_mean_est.csv')
    y_std = np.loadtxt('y_std_est.csv')
    
    if not test:
        y = (y - y_mean) / y_std

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))

    if test:
        return y, intervals, n_Rwaves, atrial_cycle_length, conduction_constant, block_pattern_extra, block_type
    else:
        return x, y
    

def get_signals_matching_batch(batch_size):
    x = []
    y = []

    for i in range(batch_size):
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)
        
        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)
        
        block_pattern = correct_bp(block_pattern, block_type, n_Rwaves)
        signals = np.array(bp_to_signals(block_pattern, block_type, n_Rwaves, fill=False))
        idx3 = np.where(signals[2] == 1)[0]
        matching_amount = []
        for i in range(len(idx3)-1):
            matching_amount.append(idx3[i+1] - idx3[i] + 1)
        matching_amount = np.array(matching_amount)
        matching_amount -= 2
        one_hot = block_pattern_to_one_hot(matching_amount)
        one_hot = one_hot.flatten()
        x_i = np.zeros(192)
        x_i[:one_hot.shape[0]] = one_hot
        x.append(x_i)
        
        y_i = np.zeros(24)
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        y.append(y_i)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    x += 0.1 * torch.randn(x.shape[0], x.shape[1])
    
    y_mean = np.loadtxt('y_mean_est.csv')
    y_std = np.loadtxt('y_std_est.csv')

    y = (y - y_mean) / y_std

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))
    
    return x, y


def get_signals_recurrent_matching_batch(batch_size):
    x = []
    x_shapes = []
    cond = []
    for i in range(batch_size):
        y_i = np.zeros(24)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)

        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)
        
        block_pattern = correct_bp(block_pattern, block_type, n_Rwaves)
        signals = np.array(bp_to_signals(block_pattern, block_type, n_Rwaves, fill=False))
        constants = np.array([atrial_cycle_length, conduction_constant])
        constants = np.stack([constants]*signals.shape[1], axis=1)
        x_i = np.concatenate([signals, constants], axis=0)
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        
        idx3 = np.where(signals[2] == 1)[0]
        cond_i = y_to_cond(idx3, signals.shape[1], y_i)
        
        idx, = np.where(np.array(x_shapes)==x_i.shape[1])
        if len(idx) != 0:
            x[idx[0]].append(x_i)
            cond[idx[0]].append(cond_i)
        else:
            x.append([x_i])
            x_shapes.append(x_i.shape[1])
            cond.append([cond_i])
    
    aa_mean = np.loadtxt("aa_mean_est.csv")
    aa_std = np.loadtxt("aa_std_est.csv")
    cc_mean = np.loadtxt("cc_mean_est.csv")
    cc_std = np.loadtxt("cc_std_est.csv")
    cond_mean = np.loadtxt("cond_signals_mean_est.csv")
    cond_std = np.loadtxt("cond_signals_std_est.csv")
    
    for i in range(len(x)):
        x[i] = torch.tensor(x[i], dtype=torch.float32)
        x[i][:, 0:-2, :] += 0.1 * torch.randn(x[i].shape[0], x[i].shape[1]-2, x[i].shape[2])
        x[i][:, -2, :] = (x[i][:, -2, :] - aa_mean) / aa_std
        x[i][:, -1, :] = (x[i][:, -1, :] - cc_mean) / cc_std
        x[i] = torch.tensor(x[i], dtype=torch.float32)
    
        cond[i] = torch.tensor(cond[i], dtype=torch.float32)
        cond[i][:, 0, :] = (cond[i][:, 0, :] - cond_mean[0]) / cond_std[0]
        cond[i][:, 1, :] = (cond[i][:, 1, :] - cond_mean[1]) / cond_std[1]
        cond[i] = torch.tensor(cond[i], dtype=torch.float32)
    
        assert(not np.any(np.isnan(np.array(x[i]))))
        assert(not np.any(np.isnan(np.array(cond[i]))))
    
    return x, cond


    
def get_signals_recurrent_batch(batch_size):
    x = []
    x_shapes = []
    y = []
    for i in range(batch_size):
        y_i = np.zeros(24)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)

        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)
        
        block_pattern = correct_bp(block_pattern, block_type, n_Rwaves)
        signals = np.array(bp_to_signals(block_pattern, block_type, n_Rwaves, fill=False))
        constants = np.array([atrial_cycle_length, conduction_constant])
        constants = np.stack([constants]*signals.shape[1], axis=1)
        x_i = np.concatenate([signals, constants], axis=0)
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        
        idx, = np.where(np.array(x_shapes)==x_i.shape[1])
        if len(idx) != 0:
            x[idx[0]].append(x_i)
            y[idx[0]].append(y_i)
        else:
            x.append([x_i])
            x_shapes.append(x_i.shape[1])
            y.append([y_i])
    
    aa_mean = np.loadtxt("aa_mean_est.csv")
    aa_std = np.loadtxt("aa_std_est.csv")
    cc_mean = np.loadtxt("cc_mean_est.csv")
    cc_std = np.loadtxt("cc_std_est.csv")
    y_mean = np.loadtxt("y_mean_est.csv")
    y_std = np.loadtxt("y_std_est.csv")
    
    for i in range(len(x)):
        x[i] = torch.tensor(x[i], dtype=torch.float32)
        x[i][:, 0:-2, :] += 0.1 * torch.randn(x[i].shape[0], x[i].shape[1]-2, x[i].shape[2])
        x[i][:, -2, :] = (x[i][:, -2, :] - aa_mean) / aa_std
        x[i][:, -1, :] = (x[i][:, -1, :] - cc_mean) / cc_std
        x[i] = torch.tensor(x[i], dtype=torch.float32)
    
        y[i] = (y[i] - y_mean) / y_std
        y[i] = np.stack([y[i]]*x[i].shape[2], axis=2)
        y[i] = torch.tensor(y[i], dtype=torch.float32)
    
        assert(not np.any(np.isnan(np.array(x[i]))))
        assert(not np.any(np.isnan(np.array(y[i]))))
    
    return x, y
    
    
    

def get_signals_batch(batch_size, test=False, btype=0):
    x = []
    y = []
    for i in range(batch_size):
        x_i = np.zeros(602)
        y_i = np.zeros(24)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        #block_type = btype
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)
        block_pattern_extra = copy.deepcopy(block_pattern)

        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)
        
        block_pattern = correct_bp(block_pattern, block_type, n_Rwaves)
        signals = np.array(bp_to_signals(block_pattern, block_type, n_Rwaves, fill=True))
        x_i[0:200] = signals[0]
        x_i[200:400] = signals[1]
        x_i[400:600] = signals[2]
        x_i[600] = atrial_cycle_length
        x_i[601] = conduction_constant
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        x.append(x_i)
        y.append(y_i)
        

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32) 
    
    x[:, 0:-2] += 0.1 * torch.randn(x.shape[0], x.shape[1]-2)
    
    aa_mean = np.loadtxt("aa_mean_est.csv")
    aa_std = np.loadtxt("aa_std_est.csv")
    cc_mean = np.loadtxt("cc_mean_est.csv")
    cc_std = np.loadtxt("cc_std_est.csv")
    y_mean = np.loadtxt("y_mean_est.csv")
    y_std = np.loadtxt("y_std_est.csv")
    
    if not test:
        y = (y - y_mean) / y_std
    x[:, -2] = (x[:, -2] - aa_mean) / aa_std
    x[:, -1] = (x[:, -1] - cc_mean) / cc_std
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    
    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))

    
    if test:
        return y, intervals, n_Rwaves, atrial_cycle_length, conduction_constant, block_pattern_extra, block_type
    else:
        return x, y

        

def get_approx_stats(n):
    aa = []
    cc = []
    y = []
    y_single = []
    cond_array = [[], []]
    cond_signals_array = [[], []]
    
    for i in range(n):
        if i%10000 == 0:
            print(i)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        aa.append(atrial_cycle_length)
        cc.append(conduction_constant)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)
        
        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)
        
        y_i = np.zeros(24)
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        y.append(y_i)
        y_single.extend(intervals[:(n_Rwaves-1)])
        
        seq, matching = block_pattern_to_seq(block_pattern)
        cond_i = y_to_cond(matching, len(seq), y_i)
        cond_array[0].extend(cond_i[0, :])
        cond_array[1].extend(cond_i[1, :]) 
        
        signals = np.array(bp_to_signals(block_pattern, block_type, n_Rwaves, fill=False))
        idx3 = np.where(signals[2] == 1)[0]
        cond_signals = y_to_cond(idx3, signals.shape[1], y_i)
        cond_signals_array[0].extend(cond_signals[0, :])
        cond_signals_array[1].extend(cond_signals[1, :])

    aa = torch.tensor(aa, dtype=torch.float32)
    cc = torch.tensor(cc, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    y_single = torch.tensor(y, dtype=torch.float32)
    cond_array[0] = torch.tensor(cond_array[0], dtype=torch.float32)
    cond_array[1] = torch.tensor(cond_array[1], dtype=torch.float32)
    cond_signals_array[0] = torch.tensor(cond_signals_array[0], dtype=torch.float32)
    cond_signals_array[1] = torch.tensor(cond_signals_array[1], dtype=torch.float32)
    
    aa_mean = torch.mean(aa)
    aa_std = torch.std(aa)
    cc_mean = torch.mean(cc)
    cc_std = torch.std(cc)
    y_mean = torch.mean(y, axis=0)
    y_std = torch.std(y, axis=0)
    y_single_mean = torch.mean(y_single)
    y_single_std = torch.std(y_single)
    cond_mean = [torch.mean(cond_array[0]), torch.mean(cond_array[1])]
    cond_std = [torch.std(cond_array[0]), torch.std(cond_array[1])]
    cond_signals_mean = [torch.mean(cond_signals_array[0]), torch.mean(cond_signals_array[1])]
    cond_signals_std = [torch.std(cond_signals_array[0]), torch.std(cond_signals_array[1])]
  
  
    np.savetxt("aa_mean_est.csv", np.array([aa_mean]))
    np.savetxt("aa_std_est.csv", np.array([aa_std]))
    np.savetxt("cc_mean_est.csv", np.array([cc_mean]))
    np.savetxt("cc_std_est.csv", np.array([cc_std]))
    np.savetxt("y_mean_est.csv", y_mean)
    np.savetxt("y_std_est.csv", y_std)
    np.savetxt("y_single_mean_est.csv", np.array([y_single_mean]))
    np.savetxt("y_single_std_est.csv", np.array([y_single_std]))
    np.savetxt('cond_mean_est.csv', cond_mean)
    np.savetxt('cond_std_est.csv', cond_std)
    np.savetxt('cond_signals_mean_est.csv', cond_signals_mean)
    np.savetxt('cond_signals_std_est.csv', cond_signals_std)


def generate_rcINN_batch_old(batch_size):
    x = []
    x_shapes = []
    y = []
    
    for i in range(batch_size):
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)
        
        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)
        
        y_i = np.zeros(24)
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        
        seq, matching = block_pattern_to_seq(block_pattern)
        one_hot = []
        for seq_i in seq:
            time_step = list(np.concatenate(block_pattern_to_one_hot(seq_i)))
            time_step += [atrial_cycle_length, conduction_constant]
            one_hot.append(time_step)
        x_i = np.array(one_hot).T  
        
        idx, = np.where(np.array(x_shapes)==x_i.shape[1])
        if len(idx) != 0:
            x[idx[0]].append(x_i)
            y[idx[0]].append(y_i)
        else:
            x.append([x_i])
            x_shapes.append(x_i.shape[1])
            y.append([y_i])
    
    aa_mean = np.loadtxt("aa_mean_est.csv")
    aa_std = np.loadtxt("aa_std_est.csv")
    cc_mean = np.loadtxt("cc_mean_est.csv")
    cc_std = np.loadtxt("cc_std_est.csv")
    y_mean = np.loadtxt("y_mean_est.csv")
    y_std = np.loadtxt("y_std_est.csv")
    
    for i in range(len(x)):
        x[i] = torch.tensor(x[i], dtype=torch.float32)
        x[i][:, 0:-2, :] += 0.1 * torch.randn(x[i].shape[0], x[i].shape[1]-2, x[i].shape[2])
        x[i][:, -2, :] = (x[i][:, -2, :] - aa_mean) / aa_std
        x[i][:, -1, :] = (x[i][:, -1, :] - cc_mean) / cc_std
        x[i] = torch.tensor(x[i], dtype=torch.float32)
    
        y[i] = (y[i] - y_mean) / y_std
        y[i] = np.stack([y[i]]*x[i].shape[2], axis=2)
        y[i] = torch.tensor(y[i], dtype=torch.float32)
    
        assert(not np.any(np.isnan(np.array(x[i]))))
        assert(not np.any(np.isnan(np.array(y[i]))))
    
    return x, y


def generate_rcINN_matching_batch_old(batch_size):
    x = []
    x_shapes = []
    cond = []
    
    for i in range(batch_size):
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)
        
        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)
        
        y_i = np.zeros(24)
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        
        seq, matching = block_pattern_to_seq(block_pattern)
        one_hot = []
        for seq_i in seq:
            time_step = list(np.concatenate(block_pattern_to_one_hot(seq_i)))
            time_step += [atrial_cycle_length, conduction_constant]
            one_hot.append(time_step)
        x_i = np.array(one_hot).T  
        cond_i = y_to_cond(matching, len(seq), y_i)
        
        idx, = np.where(np.array(x_shapes)==x_i.shape[1])
        if len(idx) != 0:
            x[idx[0]].append(x_i)
            cond[idx[0]].append(cond_i)
        else:
            x.append([x_i])
            x_shapes.append(x_i.shape[1])
            cond.append([cond_i])
    
    aa_mean = np.loadtxt("aa_mean_est.csv")
    aa_std = np.loadtxt("aa_std_est.csv")
    cc_mean = np.loadtxt("cc_mean_est.csv")
    cc_std = np.loadtxt("cc_std_est.csv")
    cond_mean = np.loadtxt("cond_mean_est.csv")
    cond_std = np.loadtxt("cond_std_est.csv")
  
    for i in range(len(x)):
        x[i] = torch.tensor(x[i], dtype=torch.float32)
        x[i][:, 0:-2, :] += 0.1 * torch.randn(x[i].shape[0], x[i].shape[1]-2, x[i].shape[2])
        x[i][:, -2, :] = (x[i][:, -2, :] - aa_mean) / aa_std
        x[i][:, -1, :] = (x[i][:, -1, :] - cc_mean) / cc_std
        x[i] = torch.tensor(x[i], dtype=torch.float32)
        
        cond[i] = torch.tensor(cond[i], dtype=torch.float32)
        cond[i][:, 0, :] = (cond[i][:, 0, :] - cond_mean[0]) / cond_std[0]
        cond[i][:, 1, :] = (cond[i][:, 1, :] - cond_mean[1]) / cond_std[1]
        cond[i] = torch.tensor(cond[i], dtype=torch.float32)

        assert(not np.any(np.isnan(np.array(x[i]))))
        assert(not np.any(np.isnan(np.array(cond[i]))))   
    
    return x, cond
   

def get_splitter_stats(n, btype):
    aa = []
    cc = []
    y = []
    for i in range(n):
        if i%10000 == 0:
            print(i)
        y_i = np.zeros(24)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = btype
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)

        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)      

        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        
        aa.append(atrial_cycle_length)
        cc.append(conduction_constant)
        y.append(y_i)
    
    aa = torch.tensor(aa, dtype=torch.float32)
    cc = torch.tensor(cc, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    aa_mean = torch.mean(aa)
    aa_std = torch.std(aa)
    cc_mean = torch.mean(cc)
    cc_std = torch.std(cc)
    y_mean = torch.mean(y, axis=0)
    y_std = torch.std(y, axis=0)
    
    np.savetxt(f"aa_mean_splitter{btype}_est.csv", np.array([aa_mean]))
    np.savetxt(f"aa_std_splitter{btype}_est.csv", np.array([aa_std]))
    np.savetxt(f"cc_mean_splitter{btype}_est.csv", np.array([cc_mean]))
    np.savetxt(f"cc_std_splitter{btype}_est.csv", np.array([cc_std]))
    np.savetxt(f"y_mean_splitter{btype}_est.csv", y_mean)
    np.savetxt(f"y_std_splitter{btype}_est.csv", y_std)
    
    
    
def get_random_y(btype):
    n_Rwaves = np.random.randint(6,26)
    atrial_cycle_length = np.random.randint(188,401)
    conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
    block_type = random.choice(["1", "2a", "2b", "2c", "3"])
    #block_type = btype
    block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)
        
    if block_type == "1":
        intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
    if block_type == "2a":
        intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
    if block_type == "2b":
        intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                conduction_constant)
    if block_type == "2c":
        intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                conduction_constant)
    if block_type == "3":
        intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                atrial_cycle_length, conduction_constant)
    
    y_i = np.zeros(24)
    y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
    
    return y_i, intervals, n_Rwaves, atrial_cycle_length, conduction_constant, block_pattern, block_type
    


def generate_cINN_splitter3(batch_size):
    x = []
    y = []
    for i in range(batch_size):
        x_i = np.zeros(352)
        y_i = np.zeros(24)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = "3"
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)

        intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2], atrial_cycle_length,
                                        conduction_constant)      
        
        block_pattern[0] = np.array(block_pattern[0])
        block_pattern[0] -= 1
        one_hot_1 = block_pattern_to_one_hot(block_pattern[0], 2)
        one_hot_1 = one_hot_1.flatten()
        one_hot_2 = block_pattern_to_one_hot(block_pattern[1], 2)
        one_hot_2 = one_hot_2.flatten()
        one_hot_3 = block_pattern_to_one_hot(block_pattern[2], 2)
        one_hot_3 = one_hot_3.flatten()
        
        x_i[0:one_hot_1.shape[0]] = one_hot_1
        x_i[200:200 + one_hot_2.shape[0]] = one_hot_2
        x_i[300:300 + one_hot_3.shape[0]] = one_hot_3
        x_i[350:352] = [atrial_cycle_length, conduction_constant]
        
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]

        x.append(x_i)
        y.append(y_i)
        

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32) 
    
    x[:, 0:-2] += 0.1 * torch.randn(x.shape[0], x.shape[1]-2)
    
    aa_mean = np.loadtxt("aa_mean_splitter3_est.csv")
    aa_std = np.loadtxt("aa_std_splitter3_est.csv")
    cc_mean = np.loadtxt("cc_mean_splitter3_est.csv")
    cc_std = np.loadtxt("cc_std_splitter3_est.csv")
    y_mean = np.loadtxt("y_mean_splitter3_est.csv")
    y_std = np.loadtxt("y_std_splitter3_est.csv")
    
    y = (y - y_mean) / y_std
    x[:, -2] = (x[:, -2] - aa_mean) / aa_std
    x[:, -1] = (x[:, -1] - cc_mean) / cc_std
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    
    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))

    return x, y


def generate_cINN_splitter2c(batch_size):
    x = []
    y = []
    for i in range(batch_size):
        x_i = np.zeros(152)
        y_i = np.zeros(24)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = "2c"
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)

        intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                        conduction_constant)      
        
        block_pattern[0] = np.array(block_pattern[0])
        block_pattern[0] -= 1
        one_hot_1 = block_pattern_to_one_hot(block_pattern[0], 2)
        one_hot_1 = one_hot_1.flatten()
        one_hot_2 = block_pattern_to_one_hot(block_pattern[1], 2)
        one_hot_2 = one_hot_2.flatten()
        
        x_i[0:one_hot_1.shape[0]] = one_hot_1
        x_i[100:100 + one_hot_2.shape[0]] = one_hot_2
        x_i[150:152] = [atrial_cycle_length, conduction_constant]
        
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]

        x.append(x_i)
        y.append(y_i)
        

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32) 
    
    x[:, 0:-2] += 0.1 * torch.randn(x.shape[0], x.shape[1]-2)
    
    aa_mean = np.loadtxt("aa_mean_splitter2c_est.csv")
    aa_std = np.loadtxt("aa_std_splitter2c_est.csv")
    cc_mean = np.loadtxt("cc_mean_splitter2c_est.csv")
    cc_std = np.loadtxt("cc_std_splitter2c_est.csv")
    y_mean = np.loadtxt("y_mean_splitter2c_est.csv")
    y_std = np.loadtxt("y_std_splitter2c_est.csv")
    
    y = (y - y_mean) / y_std
    x[:, -2] = (x[:, -2] - aa_mean) / aa_std
    x[:, -1] = (x[:, -1] - cc_mean) / cc_std
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    
    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))

    return x, y


def generate_cINN_splitter2b(batch_size):
    x = []
    y = []
    for i in range(batch_size):
        x_i = np.zeros(352)
        y_i = np.zeros(24)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = "2b"
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)

        intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)      
        
        block_pattern[1] = np.array(block_pattern[1])
        block_pattern[1] -= 1
        one_hot_2 = block_pattern_to_one_hot(block_pattern[1], 7)
        one_hot_2 = one_hot_2.flatten()
        
        x_i[0:one_hot_2.shape[0]] = one_hot_2
        x_i[350:352] = [atrial_cycle_length, conduction_constant]

        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]

        x.append(x_i)
        y.append(y_i)
        

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32) 
    
    x[:, 0:-2] += 0.1 * torch.randn(x.shape[0], x.shape[1]-2)
    
    aa_mean = np.loadtxt("aa_mean_splitter2b_est.csv")
    aa_std = np.loadtxt("aa_std_splitter2b_est.csv")
    cc_mean = np.loadtxt("cc_mean_splitter2b_est.csv")
    cc_std = np.loadtxt("cc_std_splitter2b_est.csv")
    y_mean = np.loadtxt("y_mean_splitter2b_est.csv")
    y_std = np.loadtxt("y_std_splitter2b_est.csv")
    
    y = (y - y_mean) / y_std
    x[:, -2] = (x[:, -2] - aa_mean) / aa_std
    x[:, -1] = (x[:, -1] - cc_mean) / cc_std
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    
    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))

    return x, y



def generate_cINN_splitter2a(batch_size):
    x = []
    y = []
    for i in range(batch_size):
        x_i = np.zeros(177)
        y_i = np.zeros(24)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = "2a"
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)

        intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)      
        
        block_pattern[1] = np.array(block_pattern[1])
        block_pattern[1] -= 1
        one_hot_2 = block_pattern_to_one_hot(block_pattern[1], 7)
        one_hot_2 = one_hot_2.flatten()
        
        x_i[0:one_hot_2.shape[0]] = one_hot_2
        x_i[175:177] = [atrial_cycle_length, conduction_constant]

        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]

        x.append(x_i)
        y.append(y_i)
        

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32) 
    
    x[:, 0:-2] += 0.1 * torch.randn(x.shape[0], x.shape[1]-2)
    
    aa_mean = np.loadtxt("aa_mean_splitter2a_est.csv")
    aa_std = np.loadtxt("aa_std_splitter2a_est.csv")
    cc_mean = np.loadtxt("cc_mean_splitter2a_est.csv")
    cc_std = np.loadtxt("cc_std_splitter2a_est.csv")
    y_mean = np.loadtxt("y_mean_splitter2a_est.csv")
    y_std = np.loadtxt("y_std_splitter2a_est.csv")
    
    y = (y - y_mean) / y_std
    x[:, -2] = (x[:, -2] - aa_mean) / aa_std
    x[:, -1] = (x[:, -1] - cc_mean) / cc_std
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    
    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))

    return x, y



def generate_cINN_splitter1(batch_size):
    x = []
    y = []
    for i in range(batch_size):
        x_i = np.zeros(177)
        y_i = np.zeros(24)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = "1"
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)

        intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)      
        
        block_pattern[1] = np.array(block_pattern[1])
        block_pattern[1] -= 1
        one_hot_2 = block_pattern_to_one_hot(block_pattern[1], 7)
        one_hot_2 = one_hot_2.flatten()
        
        x_i[0:one_hot_2.shape[0]] = one_hot_2
        x_i[175:177] = [atrial_cycle_length, conduction_constant]

        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]

        x.append(x_i)
        y.append(y_i)
        

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32) 
    
    x[:, 0:-2] += 0.1 * torch.randn(x.shape[0], x.shape[1]-2)
    
    aa_mean = np.loadtxt("aa_mean_splitter1_est.csv")
    aa_std = np.loadtxt("aa_std_splitter1_est.csv")
    cc_mean = np.loadtxt("cc_mean_splitter1_est.csv")
    cc_std = np.loadtxt("cc_std_splitter1_est.csv")
    y_mean = np.loadtxt("y_mean_splitter1_est.csv")
    y_std = np.loadtxt("y_std_splitter1_est.csv")
    
    y = (y - y_mean) / y_std
    x[:, -2] = (x[:, -2] - aa_mean) / aa_std
    x[:, -1] = (x[:, -1] - cc_mean) / cc_std
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    
    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))

    return x, y


def generate_cINN_batch_old(batch_size, test=False, btype=0):
    x = []
    y = []
    for i in range(batch_size):
        x_i = np.zeros(1402)
        y_i = np.zeros(24)
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        #block_type = btype
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)
        block_pattern_extra = copy.deepcopy(block_pattern)

        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)        

        one_hot_1 = block_pattern_to_one_hot(block_pattern[0])
        one_hot_2 = block_pattern_to_one_hot(block_pattern[1])
        one_hot_3 = block_pattern_to_one_hot(block_pattern[2])
        one_hot_1 = one_hot_1.flatten()
        one_hot_2 = one_hot_2.flatten()
        one_hot_3 = one_hot_3.flatten()
        
        x_i[0:one_hot_1.shape[0]] = one_hot_1
        x_i[800:800+one_hot_2.shape[0]] = one_hot_2
        x_i[1200:1200+one_hot_3.shape[0]] = one_hot_3
        x_i[1400:1402] = [atrial_cycle_length, conduction_constant]

        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]

        x.append(x_i)
        y.append(y_i)
        

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32) 
    
    x[:, 0:-2] += 0.1 * torch.randn(x.shape[0], x.shape[1]-2)
    
    aa_mean = np.loadtxt("aa_mean_est.csv")
    aa_std = np.loadtxt("aa_std_est.csv")
    cc_mean = np.loadtxt("cc_mean_est.csv")
    cc_std = np.loadtxt("cc_std_est.csv")
    y_mean = np.loadtxt("y_mean_est.csv")
    y_std = np.loadtxt("y_std_est.csv")
    
    if not test:
        y = (y - y_mean) / y_std
    x[:, -2] = (x[:, -2] - aa_mean) / aa_std
    x[:, -1] = (x[:, -1] - cc_mean) / cc_std
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    
    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))

    
    if test:
        return y, intervals, n_Rwaves, atrial_cycle_length, conduction_constant, block_pattern_extra, block_type
    else:
        return x, y
    



def generate_seq_batch_old(batch_size, test=False, btype=0):
    x = []
    y = []
    
    for i in range(batch_size):
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        #block_type = btype
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)
        block_pattern_extra = copy.deepcopy(block_pattern)
        
        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)
        
        seq, matching = block_pattern_to_seq(block_pattern)
        x_i = np.zeros(188)
        x_i[len(seq)-6] = 1
        x.append(x_i)
        
        y_i = np.zeros(24)
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        y.append(y_i)
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    x += 0.1 * torch.randn(x.shape[0], x.shape[1])
    
    y_mean = np.loadtxt('y_mean_est.csv')
    y_std = np.loadtxt('y_std_est.csv')
    
    if not test:
        y = (y - y_mean) / y_std

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))

    if test:
        return y, intervals, n_Rwaves, atrial_cycle_length, conduction_constant, block_pattern_extra, block_type
    else:
        return x, y
    


def generate_matching_batch_old(batch_size):
    x = []
    y = []

    for i in range(batch_size):
        n_Rwaves = np.random.randint(6,26)
        atrial_cycle_length = np.random.randint(188,401)
        conduction_constant = np.random.randint(1,atrial_cycle_length + 1)
        block_type = random.choice(["1", "2a", "2b", "2c", "3"])
        block_pattern = generate_block_pattern_alt(block_type, n_Rwaves)
        
        if block_type == "1":
            intervals = simulate_type_1(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2a":
            intervals = simulate_type_2a(block_pattern[1], atrial_cycle_length,
                                        conduction_constant)
        if block_type == "2b":
            intervals = simulate_type_2b(block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "2c":
            intervals = simulate_type_2c(block_pattern[0], block_pattern[1], atrial_cycle_length,
                                    conduction_constant)
        if block_type == "3":
            intervals = simulate_type_3(block_pattern[0], block_pattern[1], block_pattern[2],
                                    atrial_cycle_length, conduction_constant)
        
        seq, matching = block_pattern_to_seq(block_pattern)
        x_i = np.zeros(192)
        matching_amount = []
        for i in range(len(matching)-1):
            matching_amount.append(matching[i+1] - matching[i] + 1)
        matching_amount = np.array(matching_amount)
        matching_amount -= 2
        one_hot = block_pattern_to_one_hot(matching_amount)
        one_hot = one_hot.flatten()
        x_i[:one_hot.shape[0]] = one_hot
        x.append(x_i)
        
        y_i = np.zeros(24)
        y_i[:(n_Rwaves-1)] = intervals[:(n_Rwaves-1)]
        y.append(y_i)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    x += 0.1 * torch.randn(x.shape[0], x.shape[1])
    
    y_mean = np.loadtxt('y_mean_est.csv')
    y_std = np.loadtxt('y_std_est.csv')

    y = (y - y_mean) / y_std

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    assert(not np.any(np.isnan(np.array(x))))
    assert(not np.any(np.isnan(np.array(y))))
    
    return x, y