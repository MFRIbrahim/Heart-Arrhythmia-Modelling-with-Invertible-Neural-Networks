import torch
import numpy as np

import datagen
import config

import model as Model
import random
import os
import json

from multiprocessing import Pool


from mavb.forward import simulate_type_1
from mavb.forward import simulate_type_2a
from mavb.forward import simulate_type_2b
from mavb.forward import simulate_type_2c
from mavb.forward import simulate_type_3

import warnings
warnings.filterwarnings('ignore')


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def find_nearest(array, value):
    array = np.array(array)
    idx = np.argmin((np.abs(array - value)), axis=0)
    return idx


def train_splitter(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.generate_cINN_splitter3(
        config.batch_size * config.n_iterations)
    for i in range(0, len(x), config.batch_size):
        x_i = x[i:i+config.batch_size].cuda()
        y_i = y[i:i+config.batch_size].cuda()
        optim.zero_grad()
        z_i = model(x_i, c=y_i)
        log_jac = model.log_jacobian(run_forward=False)
        loss = torch.mean(0.5 * torch.sum(z_i**2, dim=1) - log_jac)
        l_tot += loss.data.item()
        loss.backward()
        optim.step()

    return l_tot/config.n_iterations


def train_signals_recurrent_matching(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.get_signals_recurrent_matching_batch(
        config.batch_size * config.n_iterations)

    for i in range(len(x)):
        if x[i].shape[0] >= config.batch_size:
            beg = 0
            end = config.batch_size
            for j in range(len(x[i])//config.batch_size):
                optim.zero_grad()
                x_i = x[i][beg:end].cuda()
                y_i = y[i][beg:end].cuda()
                z_i = model(x_i, c=y_i, recurrent=True)
                log_jac = model.log_jacobian(run_forward=False)
                loss = torch.mean(
                    torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
                l_tot += (config.batch_size / (config.batch_size *
                                               config.n_iterations)) * loss.data.item()
                loss.backward()
                optim.step()
                beg += config.batch_size
                end += config.batch_size
            if (len(x[i])//config.batch_size) * config.batch_size != len(x[i]):
                optim.zero_grad()
                x_i = x[i][beg:].cuda()
                y_i = y[i][beg:].cuda()
                z_i = model(x_i, c=y_i, recurrent=True)
                log_jac = model.log_jacobian(run_forward=False)
                loss = torch.mean(
                    torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
                l_tot += ((len(x[i]) - beg) / (config.batch_size *
                                               config.n_iterations)) * loss.data.item()
                loss.backward()
                optim.step()

        else:
            optim.zero_grad()
            x_i = x[i].cuda()
            y_i = y[i].cuda()
            z_i = model(x_i, c=y_i, recurrent=True)
            log_jac = model.log_jacobian(run_forward=False)
            loss = torch.mean(
                torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
            l_tot += (len(x[i]) / (config.batch_size *
                                   config.n_iterations)) * loss.data.item()
            loss.backward()
            optim.step()

    return l_tot


def train_signals_recurrent(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.get_signals_recurrent_batch(
        config.batch_size * config.n_iterations)

    for i in range(len(x)):
        if x[i].shape[0] >= config.batch_size:
            beg = 0
            end = config.batch_size
            for j in range(len(x[i])//config.batch_size):
                optim.zero_grad()
                x_i = x[i][beg:end].cuda()
                y_i = y[i][beg:end].cuda()
                z_i = model(x_i, c=y_i, recurrent=True)
                log_jac = model.log_jacobian(run_forward=False)
                loss = torch.mean(
                    torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
                l_tot += (config.batch_size / (config.batch_size *
                                               config.n_iterations)) * loss.data.item()
                loss.backward()
                optim.step()
                beg += config.batch_size
                end += config.batch_size
            if (len(x[i])//config.batch_size) * config.batch_size != len(x[i]):
                optim.zero_grad()
                x_i = x[i][beg:].cuda()
                y_i = y[i][beg:].cuda()
                z_i = model(x_i, c=y_i, recurrent=True)
                log_jac = model.log_jacobian(run_forward=False)
                loss = torch.mean(
                    torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
                l_tot += ((len(x[i]) - beg) / (config.batch_size *
                                               config.n_iterations)) * loss.data.item()
                loss.backward()
                optim.step()

        else:
            optim.zero_grad()
            x_i = x[i].cuda()
            y_i = y[i].cuda()
            z_i = model(x_i, c=y_i, recurrent=True)
            log_jac = model.log_jacobian(run_forward=False)
            loss = torch.mean(
                torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
            l_tot += (len(x[i]) / (config.batch_size *
                                   config.n_iterations)) * loss.data.item()
            loss.backward()
            optim.step()

    return l_tot


def train_signals(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.get_signals_batch(config.batch_size * config.n_iterations)
    for i in range(0, len(x), config.batch_size):
        x_i = x[i:i+config.batch_size].cuda()
        y_i = y[i:i+config.batch_size].cuda()
        optim.zero_grad()
        z_i = model(x_i, c=y_i)
        log_jac = model.log_jacobian(run_forward=False)
        loss = torch.mean(0.5 * torch.sum(z_i**2, dim=1) - log_jac)
        l_tot += loss.data.item()
        loss.backward()
        optim.step()

    return l_tot/config.n_iterations


def train_signals_matching(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.get_signals_matching_batch(
        config.batch_size * config.n_iterations)
    for i in range(0, len(x), config.batch_size):
        x_i = x[i:i+config.batch_size].cuda()
        y_i = y[i:i+config.batch_size].cuda()
        optim.zero_grad()
        z_i = model(x_i, c=y_i)
        log_jac = model.log_jacobian(run_forward=False)
        loss = torch.mean(0.5 * torch.sum(z_i**2, dim=1) - log_jac)
        l_tot += loss.data.item()
        loss.backward()
        optim.step()

    return l_tot/config.n_iterations


def train_signals_sequence(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.get_signals_sequence_batch(
        config.batch_size * config.n_iterations)
    for i in range(0, len(x), config.batch_size):
        x_i = x[i:i+config.batch_size].cuda()
        y_i = y[i:i+config.batch_size].cuda()
        optim.zero_grad()
        z_i = model(x_i, c=y_i)
        log_jac = model.log_jacobian(run_forward=False)
        loss = torch.mean(0.5 * torch.sum(z_i**2, dim=1) - log_jac)
        l_tot += loss.data.item()
        loss.backward()
        optim.step()

    return l_tot/config.n_iterations


def train_matching_old(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.generate_matching_batch_old(
        config.batch_size * config.n_iterations)
    for i in range(0, len(x), config.batch_size):
        x_i = x[i:i+config.batch_size].cuda()
        y_i = y[i:i+config.batch_size].cuda()
        optim.zero_grad()
        z_i = model(x_i, c=y_i)
        log_jac = model.log_jacobian(run_forward=False)
        loss = torch.mean(0.5 * torch.sum(z_i**2, dim=1) - log_jac)
        l_tot += loss.data.item()
        loss.backward()
        optim.step()

    return l_tot/config.n_iterations


def train_seq_old(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.generate_seq_batch_old(
        config.batch_size * config.n_iterations)
    for i in range(0, len(x), config.batch_size):
        x_i = x[i:i+config.batch_size].cuda()
        y_i = y[i:i+config.batch_size].cuda()
        optim.zero_grad()
        z_i = model(x_i, c=y_i)
        log_jac = model.log_jacobian(run_forward=False)
        loss = torch.mean(0.5 * torch.sum(z_i**2, dim=1) - log_jac)
        l_tot += loss.data.item()
        loss.backward()
        optim.step()

    return l_tot/config.n_iterations


def train_cINN_old(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.generate_cINN_batch_old(
        config.batch_size * config.n_iterations)
    for i in range(0, len(x), config.batch_size):
        x_i = x[i:i+config.batch_size].cuda()
        y_i = y[i:i+config.batch_size].cuda()
        optim.zero_grad()
        z_i = model(x_i, c=y_i)
        log_jac = model.log_jacobian(run_forward=False)
        loss = torch.mean(0.5 * torch.sum(z_i**2, dim=1) - log_jac)
        l_tot += loss.data.item()
        loss.backward()
        optim.step()

    return l_tot/config.n_iterations


def train_rcINN_matching_old(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.generate_rcINN_matching_batch_old(
        config.batch_size * config.n_iterations)

    for i in range(len(x)):
        if x[i].shape[0] >= config.batch_size:
            beg = 0
            end = config.batch_size
            for j in range(len(x[i])//config.batch_size):
                optim.zero_grad()
                x_i = x[i][beg:end].cuda()
                y_i = y[i][beg:end].cuda()
                z_i = model(x_i, c=y_i, recurrent=True)
                log_jac = model.log_jacobian(run_forward=False)
                loss = torch.mean(
                    torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
                l_tot += (config.batch_size / (config.batch_size *
                                               config.n_iterations)) * loss.data.item()
                loss.backward()
                optim.step()
                beg += config.batch_size
                end += config.batch_size
            if (len(x[i])//config.batch_size) * config.batch_size != len(x[i]):
                optim.zero_grad()
                x_i = x[i][beg:].cuda()
                y_i = y[i][beg:].cuda()
                z_i = model(x_i, c=y_i, recurrent=True)
                log_jac = model.log_jacobian(run_forward=False)
                loss = torch.mean(
                    torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
                l_tot += ((len(x[i]) - beg) / (config.batch_size *
                                               config.n_iterations)) * loss.data.item()
                loss.backward()
                optim.step()

        else:
            optim.zero_grad()
            x_i = x[i].cuda()
            y_i = y[i].cuda()
            z_i = model(x_i, c=y_i, recurrent=True)
            log_jac = model.log_jacobian(run_forward=False)
            loss = torch.mean(
                torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
            l_tot += (len(x[i]) / (config.batch_size *
                                   config.n_iterations)) * loss.data.item()
            loss.backward()
            optim.step()

    return l_tot


def train_rcINN_old(model, optim):
    model.train()
    l_tot = 0
    x, y = datagen.generate_rcINN_batch_old(
        config.batch_size * config.n_iterations)

    for i in range(len(x)):
        if x[i].shape[0] >= config.batch_size:
            beg = 0
            end = config.batch_size
            for j in range(len(x[i])//config.batch_size):
                optim.zero_grad()
                x_i = x[i][beg:end].cuda()
                y_i = y[i][beg:end].cuda()
                z_i = model(x_i, c=y_i, recurrent=True)
                log_jac = model.log_jacobian(run_forward=False)
                loss = torch.mean(
                    torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
                l_tot += (config.batch_size / (config.batch_size *
                                               config.n_iterations)) * loss.data.item()
                loss.backward()
                optim.step()
                beg += config.batch_size
                end += config.batch_size
            if (len(x[i])//config.batch_size) * config.batch_size != len(x[i]):
                optim.zero_grad()
                x_i = x[i][beg:].cuda()
                y_i = y[i][beg:].cuda()
                z_i = model(x_i, c=y_i, recurrent=True)
                log_jac = model.log_jacobian(run_forward=False)
                loss = torch.mean(
                    torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
                l_tot += ((len(x[i]) - beg) / (config.batch_size *
                                               config.n_iterations)) * loss.data.item()
                loss.backward()
                optim.step()

        else:
            optim.zero_grad()
            x_i = x[i].cuda()
            y_i = y[i].cuda()
            z_i = model(x_i, c=y_i, recurrent=True)
            log_jac = model.log_jacobian(run_forward=False)
            loss = torch.mean(
                torch.sum(0.5 * torch.sum(z_i**2, dim=1) - log_jac, dim=1))
            l_tot += (len(x[i]) / (config.batch_size *
                                   config.n_iterations)) * loss.data.item()
            loss.backward()
            optim.step()

    return l_tot


def print_stats(filtered_bp, true_stats, name, signals=False, splitter=False):
    y_true = true_stats[1]
    n_Rwaves = true_stats[2]

    bp = []

    for i in range(len(filtered_bp)):
        for j in range(len(filtered_bp[i][0])):
            if signals:
                bp.append([[filtered_bp[i][0][j]], [
                    filtered_bp[i][1][j]], filtered_bp[i][2], filtered_bp[i][3]])
            else:
                bp.append([[filtered_bp[i][0][j]], [
                    filtered_bp[i][1][j]], filtered_bp[i][2]])

    intervals = []

    if not splitter:
        for i in range(len(bp)):
            itv_sub = []
            for j in range(len(bp[i][0])):
                if bp[i][1][j] == "1":
                    itv = simulate_type_1(
                        bp[i][0][j][1], bp[i][2][0], bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if bp[i][1][j] == "2a":
                    itv = simulate_type_2a(
                        bp[i][0][j][1], bp[i][2][0], bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if bp[i][1][j] == "2b":
                    itv = simulate_type_2b(
                        bp[i][0][j][1], bp[i][2][0], bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if bp[i][1][j] == "2c":
                    itv = simulate_type_2c(
                        bp[i][0][j][0], bp[i][0][j][1], bp[i][2][0], bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if bp[i][1][j] == "3":
                    itv = simulate_type_3(
                        bp[i][0][j][0], bp[i][0][j][1], bp[i][0][j][2], bp[i][2][0], bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
            intervals.append(itv_sub)

    if splitter:
        for i in range(len(bp)):
            itv_sub = []
            for j in range(len(bp[i][0])):
                if bp[i][1][j] == "1":
                    itv = simulate_type_1(
                        bp[i][0][j][0], bp[i][2][0], bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if bp[i][1][j] == "2a":
                    itv = simulate_type_2a(
                        bp[i][0][j][0], bp[i][2][0], bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if bp[i][1][j] == "2b":
                    itv = simulate_type_2b(
                        bp[i][0][j][0], bp[i][2][0], bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if bp[i][1][j] == "2c":
                    itv = simulate_type_2c(
                        bp[i][0][j][0], bp[i][0][j][1], bp[i][2][0], bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if bp[i][1][j] == "3":
                    itv = simulate_type_3(
                        bp[i][0][j][0], bp[i][0][j][1], bp[i][0][j][2], bp[i][2][0], bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
            intervals.append(itv_sub)

    differences = []
    indeces = []

    for i in range(len(intervals)):
        for j in range(len(intervals[i])):
            differences.append(
                np.mean(np.abs(intervals[i][j]-y_true[:(n_Rwaves-1)])))
            indeces.append([i, j])

    differences = np.array(differences)
    indeces = np.array(indeces)

    idx = np.argsort(differences)
    differences = differences[idx]
    indeces = indeces[idx]

    for i in range(len(differences[:10])):
        append_new_line(
            f'top10_sol_{name}.txt', "--------------------------")
        append_new_line(f'top10_sol_{name}.txt', str(
            intervals[indeces[i][0]][indeces[i][1]]))
        append_new_line(
            f'top10_sol_{name}.txt', str(differences[i]))
        append_new_line(f'top10_sol_{name}.txt', str(
            bp[indeces[i][0]][0][indeces[i][1]]))
        append_new_line(f'top10_sol_{name}.txt', str(
            bp[indeces[i][0]][1][indeces[i][1]]))
        append_new_line(f'top10_sol_{name}.txt', str(
            bp[indeces[i][0]][2]))
        if signals:
            append_new_line(f'top10_sol_{name}.txt', str(
                bp[indeces[i][0]][3]))
        append_new_line(
            f'top10_sol_{name}.txt', "--------------------------")

    append_new_line(
        f'top10_sol_{name}.txt', "================================================")
    append_new_line(
        f'top10_sol_{name}.txt', str(y_true))
    append_new_line(
        f'top10_sol_{name}.txt', "================================================")


def make_stats(filtered_bp, true_stats, name, signals=False, splitter=False):
    y_true = true_stats[1]
    n_Rwaves = true_stats[2]
    atrial_cycle_length = true_stats[3]
    conduction_constant = true_stats[4]
    block_pattern_true = true_stats[5]
    block_type = true_stats[6]

    counter1 = 0
    counter2a = 0
    counter2b = 0
    counter2c = 0
    counter3 = 0
    same_bp = []
    alt_bp = []

    for i in range(len(filtered_bp)):
        for j in range(len(filtered_bp[i][0])):
            if filtered_bp[i][1][j] == block_type:
                if signals:
                    same_bp.append([[filtered_bp[i][0][j]], [
                                   filtered_bp[i][1][j]], filtered_bp[i][2], filtered_bp[i][3]])
                else:
                    same_bp.append([[filtered_bp[i][0][j]], [
                                   filtered_bp[i][1][j]], filtered_bp[i][2]])
            else:
                if signals:
                    alt_bp.append([[filtered_bp[i][0][j]], [
                                  filtered_bp[i][1][j]], filtered_bp[i][2], filtered_bp[i][3]])
                else:
                    alt_bp.append([[filtered_bp[i][0][j]], [
                                  filtered_bp[i][1][j]], filtered_bp[i][2]])
                if filtered_bp[i][1][j] == "1":
                    counter1 += 1
                if filtered_bp[i][1][j] == "2a":
                    counter2a += 1
                if filtered_bp[i][1][j] == "2b":
                    counter2b += 1
                if filtered_bp[i][1][j] == "2c":
                    counter2c += 1
                if filtered_bp[i][1][j] == "3":
                    counter3 += 1

    ratio1 = 0
    ratio2a = 0
    ratio2b = 0
    ratio2c = 0
    ratio3 = 0
    if len(alt_bp) > 0:
        ratio1 = counter1 / len(alt_bp)
        ratio2a = counter2a / len(alt_bp)
        ratio2b = counter2b / len(alt_bp)
        ratio2c = counter2c / len(alt_bp)
        ratio3 = counter3 / len(alt_bp)

    intervals_same = []
    intervals_alt = []

    if not splitter:
        for i in range(len(same_bp)):
            itv_sub = []
            for j in range(len(same_bp[i][0])):
                if same_bp[i][1][j] == "1":
                    itv = simulate_type_1(
                        same_bp[i][0][j][1], same_bp[i][2][0], same_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if same_bp[i][1][j] == "2a":
                    itv = simulate_type_2a(
                        same_bp[i][0][j][1], same_bp[i][2][0], same_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if same_bp[i][1][j] == "2b":
                    itv = simulate_type_2b(
                        same_bp[i][0][j][1], same_bp[i][2][0], same_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if same_bp[i][1][j] == "2c":
                    itv = simulate_type_2c(
                        same_bp[i][0][j][0], same_bp[i][0][j][1], same_bp[i][2][0], same_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if same_bp[i][1][j] == "3":
                    itv = simulate_type_3(
                        same_bp[i][0][j][0], same_bp[i][0][j][1], same_bp[i][0][j][2], same_bp[i][2][0], same_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
            intervals_same.append(itv_sub)

        for i in range(len(alt_bp)):
            itv_sub = []
            for j in range(len(alt_bp[i][0])):
                if alt_bp[i][1][j] == "1":
                    itv = simulate_type_1(
                        alt_bp[i][0][j][1], alt_bp[i][2][0], alt_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if alt_bp[i][1][j] == "2a":
                    itv = simulate_type_2a(
                        alt_bp[i][0][j][1], alt_bp[i][2][0], alt_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if alt_bp[i][1][j] == "2b":
                    itv = simulate_type_2b(
                        alt_bp[i][0][j][1], alt_bp[i][2][0], alt_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if alt_bp[i][1][j] == "2c":
                    itv = simulate_type_2c(
                        alt_bp[i][0][j][0], alt_bp[i][0][j][1], alt_bp[i][2][0], alt_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if alt_bp[i][1][j] == "3":
                    itv = simulate_type_3(
                        alt_bp[i][0][j][0], alt_bp[i][0][j][1], alt_bp[i][0][j][2], alt_bp[i][2][0], alt_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
            intervals_alt.append(itv_sub)

    if splitter:
        for i in range(len(same_bp)):
            itv_sub = []
            for j in range(len(same_bp[i][0])):
                if same_bp[i][1][j] == "1":
                    itv = simulate_type_1(
                        same_bp[i][0][j][0], same_bp[i][2][0], same_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if same_bp[i][1][j] == "2a":
                    itv = simulate_type_2a(
                        same_bp[i][0][j][0], same_bp[i][2][0], same_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if same_bp[i][1][j] == "2b":
                    itv = simulate_type_2b(
                        same_bp[i][0][j][0], same_bp[i][2][0], same_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if same_bp[i][1][j] == "2c":
                    itv = simulate_type_2c(
                        same_bp[i][0][j][0], same_bp[i][0][j][1], same_bp[i][2][0], same_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if same_bp[i][1][j] == "3":
                    itv = simulate_type_3(
                        same_bp[i][0][j][0], same_bp[i][0][j][1], same_bp[i][0][j][2], same_bp[i][2][0], same_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
            intervals_same.append(itv_sub)

        for i in range(len(alt_bp)):
            itv_sub = []
            for j in range(len(alt_bp[i][0])):
                if alt_bp[i][1][j] == "1":
                    itv = simulate_type_1(
                        alt_bp[i][0][j][0], alt_bp[i][2][0], alt_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if alt_bp[i][1][j] == "2a":
                    itv = simulate_type_2a(
                        alt_bp[i][0][j][0], alt_bp[i][2][0], alt_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if alt_bp[i][1][j] == "2b":
                    itv = simulate_type_2b(
                        alt_bp[i][0][j][0], alt_bp[i][2][0], alt_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if alt_bp[i][1][j] == "2c":
                    itv = simulate_type_2c(
                        alt_bp[i][0][j][0], alt_bp[i][0][j][1], alt_bp[i][2][0], alt_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
                if alt_bp[i][1][j] == "3":
                    itv = simulate_type_3(
                        alt_bp[i][0][j][0], alt_bp[i][0][j][1], alt_bp[i][0][j][2], alt_bp[i][2][0], alt_bp[i][2][1])
                    itv_sub.append(itv[:(n_Rwaves-1)])
            intervals_alt.append(itv_sub)

    differences_same = []
    indeces_same = []

    differences_alt = []
    indeces_alt = []

    for i in range(len(intervals_same)):
        for j in range(len(intervals_same[i])):
            differences_same.append(
                np.mean(np.abs(intervals_same[i][j]-y_true[:(n_Rwaves-1)])))
            indeces_same.append([i, j])

    for i in range(len(intervals_alt)):
        for j in range(len(intervals_alt[i])):
            differences_alt.append(
                np.mean(np.abs(intervals_alt[i][j]-y_true[:(n_Rwaves-1)])))
            indeces_alt.append([i, j])

    differences_same = np.array(differences_same)
    indeces_same = np.array(indeces_same)

    differences_alt = np.array(differences_alt)
    indeces_alt = np.array(indeces_alt)

    idx_same = np.argsort(differences_same)
    differences_same = differences_same[idx_same]
    indeces_same = indeces_same[idx_same]

    idx_alt = np.argsort(differences_alt)
    differences_alt = differences_alt[idx_alt]
    indeces_alt = indeces_alt[idx_alt]

    if len(differences_alt) + len(differences_same) > 0:
        alt_same_ratio = len(differences_alt) / \
            (len(differences_alt) + len(differences_same))
        append_new_line(f'TEST_alt_same_ratio_{name}.txt', str(alt_same_ratio))
        append_new_line(f'TEST_block_type_ratios_{name}.txt', str(ratio1))
        append_new_line(f'TEST_block_type_ratios_{name}.txt', str(ratio2a))
        append_new_line(f'TEST_block_type_ratios_{name}.txt', str(ratio2b))
        append_new_line(f'TEST_block_type_ratios_{name}.txt', str(ratio2c))
        append_new_line(f'TEST_block_type_ratios_{name}.txt', str(ratio3))

    if len(differences_same) != 0:
        append_new_line(
            f'TEST_top1_diff_same_{name}.txt', str(differences_same[0]))

    if len(differences_alt) != 0:
        append_new_line(
            f'TEST_top1_diff_alt_{name}.txt', str(differences_alt[0]))
        if alt_bp[indeces_alt[0][0]][1][indeces_alt[0][1]] == "1":
            append_new_line(f'TEST_top_alt_block_type_{name}.txt', str(0))
        if alt_bp[indeces_alt[0][0]][1][indeces_alt[0][1]] == "2a":
            append_new_line(f'TEST_top_alt_block_type_{name}.txt', str(1))
        if alt_bp[indeces_alt[0][0]][1][indeces_alt[0][1]] == "2b":
            append_new_line(f'TEST_top_alt_block_type_{name}.txt', str(2))
        if alt_bp[indeces_alt[0][0]][1][indeces_alt[0][1]] == "2c":
            append_new_line(f'TEST_top_alt_block_type_{name}.txt', str(3))
        if alt_bp[indeces_alt[0][0]][1][indeces_alt[0][1]] == "3":
            append_new_line(f'TEST_top_alt_block_type_{name}.txt', str(4))

    for i in range(len(differences_same[:5])):
        append_new_line(
            f'TEST_top10_sol_same_{name}.txt', "--------------------------")
        append_new_line(f'TEST_top10_sol_same_{name}.txt', str(
            intervals_same[indeces_same[i][0]][indeces_same[i][1]]))
        append_new_line(
            f'TEST_top10_sol_same_{name}.txt', str(differences_same[i]))
        append_new_line(f'TEST_top10_sol_same_{name}.txt', str(
            same_bp[indeces_same[i][0]][0][indeces_same[i][1]]))
        append_new_line(f'TEST_top10_sol_same_{name}.txt', str(
            same_bp[indeces_same[i][0]][1][indeces_same[i][1]]))
        append_new_line(f'TEST_top10_sol_same_{name}.txt', str(
            same_bp[indeces_same[i][0]][2]))
        if signals:
            append_new_line(f'TEST_top10_sol_same_{name}.txt', str(
                same_bp[indeces_same[i][0]][3]))
        append_new_line(
            f'TEST_top10_sol_same_{name}.txt', "--------------------------")

    for i in range(len(differences_alt[:5])):
        append_new_line(
            f'TEST_top10_sol_alt_{name}.txt', "--------------------------")
        append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(
            intervals_alt[indeces_alt[i][0]][indeces_alt[i][1]]))
        append_new_line(
            f'TEST_top10_sol_alt_{name}.txt', str(differences_alt[i]))
        append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(
            alt_bp[indeces_alt[i][0]][0][indeces_alt[i][1]]))
        append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(
            alt_bp[indeces_alt[i][0]][1][indeces_alt[i][1]]))
        append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(
            alt_bp[indeces_alt[i][0]][2]))
        if signals:
            append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(
                alt_bp[indeces_alt[i][0]][3]))
        append_new_line(
            f'TEST_top10_sol_alt_{name}.txt', "--------------------------")

    if signals:
        block_pattern_s = datagen.correct_bp(
            block_pattern_true, block_type, n_Rwaves)
        signals_true = np.array(datagen.bp_to_signals(
            block_pattern_s, block_type, n_Rwaves, fill=False), dtype='int16')
    append_new_line(f'TEST_top10_sol_same_{name}.txt',
                    "=====================================================")
    append_new_line(f'TEST_top10_sol_same_{name}.txt', str(
        y_true[:(n_Rwaves-1)]))
    append_new_line(f'TEST_top10_sol_same_{name}.txt', str(n_Rwaves))
    append_new_line(f'TEST_top10_sol_same_{name}.txt', str(block_type))
    append_new_line(f'TEST_top10_sol_same_{name}.txt', str(block_pattern_true))
    append_new_line(
        f'TEST_top10_sol_same_{name}.txt', str(atrial_cycle_length))
    append_new_line(
        f'TEST_top10_sol_same_{name}.txt', str(conduction_constant))
    if signals:
        append_new_line(f'TEST_top10_sol_same_{name}.txt', str(signals_true))
    append_new_line(f'TEST_top10_sol_same_{name}.txt',
                    "=====================================================")
    append_new_line(f'TEST_top10_sol_alt_{name}.txt',
                    "=====================================================")
    append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(
        y_true[:(n_Rwaves-1)]))
    append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(n_Rwaves))
    append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(block_type))
    append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(block_pattern_true))
    append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(atrial_cycle_length))
    append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(conduction_constant))
    if signals:
        append_new_line(f'TEST_top10_sol_alt_{name}.txt', str(signals_true))
    append_new_line(f'TEST_top10_sol_alt_{name}.txt',
                    "=====================================================")


def process_matching(data):
    output_matching = data[0]
    n_Rwaves = data[1]
    sequence_number = data[2]

    matchings = []

    for i in range(len(output_matching)):
        matching_re = output_matching[i].reshape((24, 8))
        sub = []
        for j in range(len(matching_re)):
            nearest = find_nearest(matching_re[j], 1)
            if np.abs(matching_re[j][nearest] - 1) < 0.5:
                sub.append(nearest + 2)
            else:
                sub.append(0)
        if (len(np.where(np.array(sub[n_Rwaves - 1:]) == 0)[0]) != len(sub[n_Rwaves - 1:]) or
                len(np.where(np.array(sub[:n_Rwaves - 1]) == 0)[0]) != 0):
            continue
        matchings.append(sub[:n_Rwaves - 1])

    matchings_converted = []
    for i in range(len(matchings)):
        counter = 0
        sub_array = [0]
        for j in range(len(matchings[i])):
            counter += matchings[i][j] - 1
            sub_array.append(counter)
        matchings_converted.append(sub_array)

    return (matchings_converted, sequence_number)


def get_outputs_signals_recurrent_matching(true_stats, mp=True):
    with torch.no_grad():
        aa_mean = np.loadtxt("aa_mean_est.csv")
        aa_std = np.loadtxt("aa_std_est.csv")
        cc_mean = np.loadtxt("cc_mean_est.csv")
        cc_std = np.loadtxt("cc_std_est.csv")
        y_mean = np.loadtxt("y_mean_est.csv")
        y_std = np.loadtxt("y_std_est.csv")
        cond_mean = np.loadtxt("cond_signals_mean_est.csv")
        cond_std = np.loadtxt("cond_signals_std_est.csv")
        config.n_x_features = 5
        config.n_cond_features = 2
        config.rnn_layers = 2
        config.hidden_size = 32
        model_rcINN, optim_rcINN, weight_scheduler_rcINN = Model.generate_rcINN_old()
        config.n_x_features = 194
        config.n_cond_features = 24
        config.n_hidden_layer_size = 512
        model_seq, optim_seq, weight_scheduler_seq = Model.generate_cINN_old()
        config.n_x_features = 192
        model_matching, optim_matching, weight_scheduler_matching = Model.generate_cINN_old()
        Model.load("model_signals_rcINN_matching.pth",
                   optim_rcINN, model_rcINN)
        Model.load("model_signals_sequence.pth", optim_seq, model_seq)
        Model.load("model_signals_matching.pth",
                   optim_matching, model_matching)
        model_rcINN.eval()
        model_seq.eval()
        model_matching.eval()

        seq_len_total = []
        for stat in true_stats:
            y_seq = np.stack([(stat[0] - y_mean) / y_std]*1000, axis=0)
            z_seq = np.random.randn(1000, 194)

            y_seq = torch.tensor(y_seq, dtype=torch.float32)
            z_seq = torch.tensor(z_seq, dtype=torch.float32)

            y_seq = y_seq.cuda()
            z_seq = z_seq.cuda()

            output_seq = model_seq(z_seq, c=y_seq, rev=True)
            output_seq = output_seq.cpu().detach()

            seq_lengths = []
            for i in range(len(output_seq)):
                seq_lengths.append(find_nearest(output_seq[i], 1) + 6)

            seq_len_total.append(seq_lengths)

        final_lengths = []
        for lens in seq_len_total:
            occurence = []
            len_track = []
            for len_i in lens:
                if len_i not in len_track:
                    len_track.append(len_i)
                    counter = 0
                    for len_j in lens:
                        if len_j == len_i:
                            counter += 1
                    occurence.append(counter)
            occurence = np.array(occurence)
            len_track = np.array(len_track)
            idx_occ = np.argsort(occurence)[::-1]
            len_track = len_track[idx_occ]
            final_lengths.append(len_track[:10])

        match_totals = []
        for k in range(len(true_stats)):
            y_matching = np.stack(
                [(true_stats[k][0] - y_mean) / y_std]*5000, axis=0)
            z_matching = np.random.randn(5000, 192)

            y_matching = torch.tensor(y_matching, dtype=torch.float32)
            z_matching = torch.tensor(z_matching, dtype=torch.float32)

            y_matching = y_matching.cuda()
            z_matching = z_matching.cuda()

            output_matching = model_matching(
                z_matching, c=y_matching, rev=True)
            output_matching = output_matching.cpu().detach()

            match_totals.append((output_matching, true_stats[k][2], k))

        if mp:
            with Pool(os.cpu_count() - 1) as pool:
                processed_matches = pool.map(process_matching, match_totals)
            processed_matches = sorted(processed_matches, key=lambda x: x[1])
        else:
            processed_matches = []
            processed_matches.append(process_matching(match_totals[0]))

        final_matches = []
        for matches in processed_matches:
            occurence = []
            match_track = []
            for match_i in matches[0]:
                if match_i not in match_track:
                    match_track.append(match_i)
                    counter = 0
                    for match_j in matches[0]:
                        if match_j == match_i:
                            counter += 1
                    occurence.append(counter)
            occurence = np.array(occurence)
            match_track = np.array(match_track)
            idx_occ = np.argsort(occurence)[::-1]
            match_track = match_track[idx_occ]
            final_matches.append(match_track)

        paired_totals = []
        for k in range(len(final_lengths)):
            paired_matchings = []
            for i in range(len(final_lengths[k])):
                sub_match = []
                for j in range(len(final_matches[k])):
                    if (final_matches[k][j][-1] == final_lengths[k][i] - 1 or
                        final_matches[k][j][-1] + 1 == final_lengths[k][i] - 1 or
                        final_matches[k][j][-1] + 2 == final_lengths[k][i] - 1 or
                        final_matches[k][j][-1] + 3 == final_lengths[k][i] - 1 or
                        final_matches[k][j][-1] + 4 == final_lengths[k][i] - 1 or
                        final_matches[k][j][-1] + 5 == final_lengths[k][i] - 1 or
                            final_matches[k][j][-1] + 6 == final_lengths[k][i] - 1):
                        sub_match.append(final_matches[k][j])
                paired_matchings.append((final_lengths[k][i], sub_match))
            paired_totals.append(paired_matchings)

        outputs = []
        for k in range(len(paired_totals)):
            y_rcINN = []
            z_rcINN = []

            for i in range(len(paired_totals[k])):
                if len(paired_totals[k][i][1]) == 0:
                    continue
                z_stack = []
                cond_stack = []
                counter = 0
                for j in range(len(paired_totals[k][i][1])):
                    z_stack.append(np.random.randn(
                        50, 5, paired_totals[k][i][0]))
                    cond_stack.append(np.stack([datagen.y_to_cond(
                        paired_totals[k][i][1][j], paired_totals[k][i][0], true_stats[k][0])]*50, axis=0))
                    counter += 1
                    if counter == 10:
                        break
                z_stack = np.concatenate(z_stack, axis=0)
                cond_stack = np.concatenate(cond_stack, axis=0)
                y_rcINN.append(cond_stack)
                z_rcINN.append(z_stack)

            for i in range(len(y_rcINN)):
                y_rcINN[i] = torch.tensor(y_rcINN[i], dtype=torch.float32)
                y_rcINN[i][:, 0, :] = (
                    y_rcINN[i][:, 0, :] - cond_mean[0]) / cond_std[0]
                y_rcINN[i][:, 1, :] = (
                    y_rcINN[i][:, 1, :] - cond_mean[1]) / cond_std[1]
                y_rcINN[i] = torch.tensor(y_rcINN[i], dtype=torch.float32)
                z_rcINN[i] = torch.tensor(z_rcINN[i], dtype=torch.float32)

            big_output = []

            for i in range(len(y_rcINN)):
                y_rcINN[i] = y_rcINN[i].cuda()
                z_rcINN[i] = z_rcINN[i].cuda()

                output_rcINN = model_rcINN(
                    z_rcINN[i], c=y_rcINN[i], rev=True, recurrent=True)
                output_rcINN = output_rcINN.cpu().detach()

                output_rcINN[:, -2, :] = output_rcINN[:, -2, :] * \
                    aa_std + aa_mean
                output_rcINN[:, -1, :] = output_rcINN[:, -1, :] * \
                    cc_std + cc_mean

                big_output.append(output_rcINN)

            outputs.append(big_output)

    return outputs


def process_outputs_signals_recurrent_matching(big_output, true_stats):
    n_Rwaves = true_stats[2]

    filtered_bp = []
    for i in range(len(big_output)):
        for j in range(len(big_output[i])):
            output_ij = np.array(big_output[i][j])
            aa_ij = float(np.mean(output_ij[-2]))
            cc_ij = float(np.mean(output_ij[-1]))
            if aa_ij < 188 or aa_ij > 400 or cc_ij < 1 or cc_ij > aa_ij:
                continue
            signals = output_ij[:3, :]
            for k in range(signals.shape[0]):
                for l in range(signals.shape[1]):
                    dist0 = abs(signals[k][l])
                    dist1 = abs(signals[k][l] - 1)
                    if dist0 < dist1:
                        signals[k][l] = 0
                    if dist0 > dist1:
                        signals[k][l] = 1
                    if dist0 == dist1:
                        signals[k][l] = random.choice([0, 1])

            lvl1 = signals[0]
            lvl2 = signals[1]
            lvl3 = signals[2]

            idx_1 = np.where(np.array(lvl1) == 1)[0]
            idx_2 = np.where(np.array(lvl2) == 1)[0]
            idx_3 = np.where(np.array(lvl3) == 1)[0]
            relevant_1 = np.array(lvl1)[:idx_1[-1]+1]
            relevant_2 = np.array(lvl2)[idx_1]
            relevant_3 = np.array(lvl3)[idx_2]
            id0_1 = np.where(np.array(relevant_1) == 0)[0]
            id0_2 = np.where(np.array(relevant_2) == 0)[0]
            id0_3 = np.where(np.array(relevant_3) == 0)[0]
            diff1 = abs(id0_1[:-1] - id0_1[1:])
            diff2 = abs(id0_2[:-1] - id0_2[1:])
            diff3 = abs(id0_3[:-1] - id0_3[1:])
            if len(id0_1) > 1:
                if min(diff1) <= 1:
                    continue
            if len(id0_2) > 1:
                if min(diff2) <= 1:
                    continue
            if len(id0_3) > 1:
                if min(diff3) <= 1:
                    continue
            if idx_2[-1] > idx_1[-1]:
                continue
            if idx_3[-1] > idx_2[-1]:
                continue

            res = datagen.signals_to_bp([lvl1, lvl2, lvl3], n_Rwaves)
            for res_i in res:
                if len(res_i) != 0:
                    filtered_bp.append([res_i[0], res_i[1], [float(aa_ij), float(
                        cc_ij)], np.array([lvl1, lvl2, lvl3], dtype='int16')])

    return (filtered_bp, true_stats)


def get_outputs_signals_recurrent(true_stats):
    with torch.no_grad():
        aa_mean = np.loadtxt("aa_mean_est.csv")
        aa_std = np.loadtxt("aa_std_est.csv")
        cc_mean = np.loadtxt("cc_mean_est.csv")
        cc_std = np.loadtxt("cc_std_est.csv")
        y_mean = np.loadtxt("y_mean_est.csv")
        y_std = np.loadtxt("y_std_est.csv")
        config.n_x_features = 5
        config.n_cond_features = 24
        config.rnn_layers = 2
        config.hidden_size = 64
        model_rcINN, optim_rcINN, weight_scheduler_rcINN = Model.generate_rcINN_old()
        config.n_x_features = 194
        config.n_hidden_layer_size = 512
        model_seq, optim_seq, weight_scheduler_seq = Model.generate_cINN_old()
        Model.load("model_signals_rcINN.pth", optim_rcINN, model_rcINN)
        Model.load("model_signals_sequence.pth", optim_seq, model_seq)
        model_rcINN.eval()
        model_seq.eval()

        seq_len_total = []
        for stat in true_stats:
            y_seq = np.stack([(stat[0] - y_mean) / y_std]*1000, axis=0)
            z_seq = np.random.randn(1000, 194)

            y_seq = torch.tensor(y_seq, dtype=torch.float32)
            z_seq = torch.tensor(z_seq, dtype=torch.float32)

            y_seq = y_seq.cuda()
            z_seq = z_seq.cuda()

            output_seq = model_seq(z_seq, c=y_seq, rev=True)
            output_seq = output_seq.cpu().detach()

            seq_lengths = []
            for i in range(len(output_seq)):
                seq_lengths.append(find_nearest(output_seq[i], 1) + 6)

            seq_len_total.append(seq_lengths)

        final_lengths = []
        for lens in seq_len_total:
            occurence = []
            len_track = []
            for len_i in lens:
                if len_i not in len_track:
                    len_track.append(len_i)
                    counter = 0
                    for len_j in lens:
                        if len_j == len_i:
                            counter += 1
                    occurence.append(counter)
            occurence = np.array(occurence)
            len_track = np.array(len_track)
            idx_occ = np.argsort(occurence)[::-1]
            len_track = len_track[idx_occ]
            final_lengths.append(len_track[:10])

        outputs = []
        for j in range(len(final_lengths)):
            big_output = []
            for seq_len in final_lengths[j]:
                y_array = np.stack(
                    [(true_stats[j][0] - y_mean) / y_std]*seq_len, axis=1)
                y_array = np.stack([y_array]*500, axis=0)
                z_array = np.random.randn(500, 5, seq_len)

                y_array = torch.tensor(y_array, dtype=torch.float32)
                z_array = torch.tensor(z_array, dtype=torch.float32)

                y_array = y_array.cuda()
                z_array = z_array.cuda()

                output = model_rcINN(z_array, c=y_array,
                                     rev=True, recurrent=True)
                output = output.cpu().detach()

                output[:, -2, :] = output[:, -2, :] * aa_std + aa_mean
                output[:, -1, :] = output[:, -1, :] * cc_std + cc_mean

                big_output.append(output)

            outputs.append(big_output)

    return outputs


def process_outputs_signals_recurrent(big_output, true_stats):
    n_Rwaves = true_stats[2]

    filtered_bp = []
    for i in range(len(big_output)):
        for j in range(len(big_output[i])):
            output_ij = np.array(big_output[i][j])
            aa_ij = float(np.mean(output_ij[-2]))
            cc_ij = float(np.mean(output_ij[-1]))
            if aa_ij < 188 or aa_ij > 400 or cc_ij < 1 or cc_ij > aa_ij:
                continue
            signals = output_ij[:3, :]
            for k in range(signals.shape[0]):
                for l in range(signals.shape[1]):
                    dist0 = abs(signals[k][l])
                    dist1 = abs(signals[k][l] - 1)
                    if dist0 < dist1:
                        signals[k][l] = 0
                    if dist0 > dist1:
                        signals[k][l] = 1
                    if dist0 == dist1:
                        signals[k][l] = random.choice([0, 1])

            lvl1 = signals[0]
            lvl2 = signals[1]
            lvl3 = signals[2]

            idx_1 = np.where(np.array(lvl1) == 1)[0]
            idx_2 = np.where(np.array(lvl2) == 1)[0]
            idx_3 = np.where(np.array(lvl3) == 1)[0]
            relevant_1 = np.array(lvl1)[:idx_1[-1]+1]
            relevant_2 = np.array(lvl2)[idx_1]
            relevant_3 = np.array(lvl3)[idx_2]
            id0_1 = np.where(np.array(relevant_1) == 0)[0]
            id0_2 = np.where(np.array(relevant_2) == 0)[0]
            id0_3 = np.where(np.array(relevant_3) == 0)[0]
            diff1 = abs(id0_1[:-1] - id0_1[1:])
            diff2 = abs(id0_2[:-1] - id0_2[1:])
            diff3 = abs(id0_3[:-1] - id0_3[1:])
            if len(id0_1) > 1:
                if min(diff1) <= 1:
                    continue
            if len(id0_2) > 1:
                if min(diff2) <= 1:
                    continue
            if len(id0_3) > 1:
                if min(diff3) <= 1:
                    continue
            if idx_2[-1] > idx_1[-1]:
                continue
            if idx_3[-1] > idx_2[-1]:
                continue

            res = datagen.signals_to_bp([lvl1, lvl2, lvl3], n_Rwaves)
            for res_i in res:
                if len(res_i) != 0:
                    filtered_bp.append([res_i[0], res_i[1], [float(aa_ij), float(
                        cc_ij)], np.array([lvl1, lvl2, lvl3], dtype='int16')])

    return (filtered_bp, true_stats)


def get_outputs_signals(true_stats):
    with torch.no_grad():
        aa_mean = np.loadtxt("aa_mean_est.csv")
        aa_std = np.loadtxt("aa_std_est.csv")
        cc_mean = np.loadtxt("cc_mean_est.csv")
        cc_std = np.loadtxt("cc_std_est.csv")
        y_mean = np.loadtxt("y_mean_est.csv")
        y_std = np.loadtxt("y_std_est.csv")
        config.n_x_features = 602
        config.n_cond_features = 24
        config.n_hidden_layer_size = 1024
        model, optim, weight_scheduler = Model.generate_cINN_old()
        Model.load("model_signals_cINN.pth", optim, model)
        model.eval()

        outputs = []

        for stat in true_stats:
            y_array = np.stack([(stat[0] - y_mean) / y_std]*5000, axis=0)
            z_array = np.random.randn(5000, 602)

            y_array = torch.tensor(y_array, dtype=torch.float32)
            z_array = torch.tensor(z_array, dtype=torch.float32)

            y_array = y_array.cuda()
            z_array = z_array.cuda()

            big_output = model(z_array, c=y_array, rev=True)
            big_output = big_output.cpu().detach()

            big_output[:, -2] = big_output[:, -2] * aa_std + aa_mean
            big_output[:, -1] = big_output[:, -1] * cc_std + cc_mean
            outputs.append(big_output)

    return outputs


def process_outputs_signals(big_output, true_stats):
    n_Rwaves = true_stats[2]

    aa = np.array(big_output[:, -2])
    cc = np.array(big_output[:, -1])

    filtered_bp = []
    for i in range(len(big_output)):
        if aa[i] < 188 or aa[i] > 400 or cc[i] < 1 or cc[i] > aa[i]:
            continue
        signals = np.array(big_output[i][0:600])
        for k in range(len(signals)):
            dist0 = abs(signals[k])
            dist1 = abs(signals[k] - 1)
            if dist0 < dist1:
                signals[k] = 0
            if dist0 > dist1:
                signals[k] = 1
            if dist0 == dist1:
                signals[k] = random.choice([0, 1])

        lvl1 = signals[0:200]
        lvl2 = signals[200:400]
        lvl3 = signals[400:600]

        idx_1 = np.where(np.array(lvl1) == 1)[0]
        idx_2 = np.where(np.array(lvl2) == 1)[0]
        idx_3 = np.where(np.array(lvl3) == 1)[0]
        relevant_1 = np.array(lvl1)[:idx_1[-1]+1]
        relevant_2 = np.array(lvl2)[idx_1]
        relevant_3 = np.array(lvl3)[idx_2]
        id0_1 = np.where(np.array(relevant_1) == 0)[0]
        id0_2 = np.where(np.array(relevant_2) == 0)[0]
        id0_3 = np.where(np.array(relevant_3) == 0)[0]
        diff1 = abs(id0_1[:-1] - id0_1[1:])
        diff2 = abs(id0_2[:-1] - id0_2[1:])
        diff3 = abs(id0_3[:-1] - id0_3[1:])
        if len(id0_1) > 1:
            if min(diff1) <= 1:
                continue
        if len(id0_2) > 1:
            if min(diff2) <= 1:
                continue
        if len(id0_3) > 1:
            if min(diff3) <= 1:
                continue
        if idx_2[-1] > idx_1[-1]:
            continue
        if idx_3[-1] > idx_2[-1]:
            continue

        lvl1 = lvl1[:idx_1[-1]+1]
        lvl2 = lvl2[:idx_1[-1]+1]
        lvl3 = lvl3[:idx_1[-1]+1]

        res = datagen.signals_to_bp([lvl1, lvl2, lvl3], n_Rwaves)

        for res_i in res:
            if len(res_i) != 0:
                filtered_bp.append([res_i[0], res_i[1], [float(aa[i]), float(
                    cc[i])], np.array([lvl1, lvl2, lvl3], dtype='int16')])

    return (filtered_bp, true_stats)


def process_outputs_splitter(data):
    big_output = data[0]
    n_Rwaves = data[1][2]
    splitter_type = data[2]
    sequence_number = data[3]

    aa = np.array(big_output[:, -2])
    cc = np.array(big_output[:, -1])

    output_bp = []
    output_constants = []
    for i in range(len(big_output)):
        if aa[i] < 188 or aa[i] > 400 or cc[i] < 1 or cc[i] > aa[i]:
            continue
        if splitter_type == "1" or splitter_type == "2a":
            x_re = big_output[:, :-2][i].reshape((25, 7))
        if splitter_type == "2b":
            x_re = big_output[:, :-2][i].reshape((50, 7))
        if splitter_type == "2c":
            x_re = big_output[:, :-2][i].reshape((75, 2))
        if splitter_type == "3":
            x_re = big_output[:, :-2][i].reshape((175, 2))

        output_bp_sub = []
        lvl1 = []
        lvl2 = []
        lvl3 = []

        if splitter_type == "1" or splitter_type == "2a":
            for j in range(25):
                nearest = find_nearest(x_re[j], 1)
                if np.abs(x_re[j][nearest] - 1) < 0.5:
                    lvl2.append(nearest + 1)
                else:
                    lvl2.append(-1)
        if splitter_type == "2b":
            for j in range(50):
                nearest = find_nearest(x_re[j], 1)
                if np.abs(x_re[j][nearest] - 1) < 0.5:
                    lvl2.append(nearest + 1)
                else:
                    lvl2.append(-1)

        if splitter_type == "2c":
            for j in range(50):
                nearest = find_nearest(x_re[j], 1)
                if np.abs(x_re[j][nearest] - 1) < 0.5:
                    lvl1.append(nearest + 1)
                else:
                    lvl1.append(-1)
            for j in range(50, 75):
                nearest = find_nearest(x_re[j], 1)
                if np.abs(x_re[j][nearest] - 1) < 0.5:
                    lvl2.append(nearest)
                else:
                    lvl2.append(-1)

        if splitter_type == "3":
            for j in range(100):
                nearest = find_nearest(x_re[j], 1)
                if np.abs(x_re[j][nearest] - 1) < 0.5:
                    lvl1.append(nearest + 1)
                else:
                    lvl1.append(-1)
            for j in range(100, 150):
                nearest = find_nearest(x_re[j], 1)
                if np.abs(x_re[j][nearest] - 1) < 0.5:
                    lvl2.append(nearest)
                else:
                    lvl2.append(-1)

            for j in range(150, 175):
                nearest = find_nearest(x_re[j], 1)
                if np.abs(x_re[j][nearest] - 1) < 0.5:
                    lvl3.append(nearest)
                else:
                    lvl3.append(-1)

        lvl1_elist = np.where(np.array(lvl1) == -1)[0]
        lvl2_elist = np.where(np.array(lvl2) == -1)[0]
        lvl3_elist = np.where(np.array(lvl3) == -1)[0]
        lvl1_end = len(lvl1)
        lvl2_end = len(lvl2)
        lvl3_end = len(lvl3)
        if len(lvl1_elist) != 0:
            lvl1_end = lvl1_elist[0]
            if len(np.where(np.array(lvl1[lvl1_end:]) == -1)[0]) != len(lvl1[lvl1_end:]):
                continue
        if len(lvl2_elist) != 0:
            lvl2_end = lvl2_elist[0]
            if len(np.where(np.array(lvl2[lvl2_end:]) == -1)[0]) != len(lvl2[lvl2_end:]):
                continue
        if len(lvl3_elist) != 0:
            lvl3_end = lvl3_elist[0]
            if len(np.where(np.array(lvl3[lvl3_end:]) == -1)[0]) != len(lvl3[lvl3_end:]):
                continue
        output_bp_sub.append(lvl1[:lvl1_end])
        output_bp_sub.append(lvl2[:lvl2_end])
        output_bp_sub.append(lvl3[:lvl3_end])
        output_bp.append(output_bp_sub)
        output_constants.append([float(aa[i]), float(cc[i])])

    filtered_bp = []
    for i in range(len(output_bp)):
        bp, bp_type = datagen.check_block_pattern_splitter(
            output_bp[i], n_Rwaves, splitter_type)
        if len(bp) != 0:
            filtered_bp.append([bp, bp_type, output_constants[i]])

    return (filtered_bp, data[1], sequence_number)


def get_outputs_splitter(true_stats, splitter_type):
    with torch.no_grad():
        aa_mean = np.loadtxt(f"aa_mean_splitter{splitter_type}_est.csv")
        aa_std = np.loadtxt(f"aa_std_splitter{splitter_type}_est.csv")
        cc_mean = np.loadtxt(f"cc_mean_splitter{splitter_type}_est.csv")
        cc_std = np.loadtxt(f"cc_std_splitter{splitter_type}_est.csv")
        y_mean = np.loadtxt(f"y_mean_splitter{splitter_type}_est.csv")
        y_std = np.loadtxt(f"y_std_splitter{splitter_type}_est.csv")
        if splitter_type == "1" or splitter_type == "2a":
            config.n_x_features = 177
            config.n_hidden_layer_size = 512
        if splitter_type == "2b" or splitter_type == "3":
            config.n_x_features = 352
            config.n_hidden_layer_size = 1024
        if splitter_type == "2c":
            config.n_x_features = 152
            config.n_hidden_layer_size = 512
        model, optim, weight_scheduler = Model.generate_cINN_old()
        Model.load(f"model_splitter{splitter_type}.pth", optim, model)
        model.eval()

        if splitter_type == "1" or splitter_type == "2a":
            z_size = 177
        if splitter_type == "2b" or splitter_type == "3":
            z_size = 352
        if splitter_type == "2c":
            z_size = 152

        outputs = []
        for stat in true_stats:
            y_array = np.stack([(stat[0] - y_mean) / y_std]*1000, axis=0)
            z_array = np.random.randn(1000, z_size)

            y_array = torch.tensor(y_array, dtype=torch.float32)
            z_array = torch.tensor(z_array, dtype=torch.float32)

            y_array = y_array.cuda()
            z_array = z_array.cuda()

            big_output = model(z_array, c=y_array, rev=True)
            big_output = big_output.cpu().detach()

            big_output[:, -2] = big_output[:, -2] * aa_std + aa_mean
            big_output[:, -1] = big_output[:, -1] * cc_std + cc_mean

            outputs.append(big_output)

    return outputs


def get_outputs_bp(true_stats):
    with torch.no_grad():
        aa_mean = np.loadtxt("aa_mean_est.csv")
        aa_std = np.loadtxt("aa_std_est.csv")
        cc_mean = np.loadtxt("cc_mean_est.csv")
        cc_std = np.loadtxt("cc_std_est.csv")
        y_mean = np.loadtxt("y_mean_est.csv")
        y_std = np.loadtxt("y_std_est.csv")
        config.n_x_features = 1402
        config.n_cond_features = 24
        config.n_hidden_layer_size = 2048
        model, optim, weight_scheduler = Model.generate_cINN_old()
        Model.load("model_bp_cINN.pth", optim, model)
        model.eval()

        outputs = []

        for stat in true_stats:
            y_array = np.stack([(stat[0] - y_mean) / y_std]*5000, axis=0)
            z_array = np.random.randn(5000, 1402)

            y_array = torch.tensor(y_array, dtype=torch.float32)
            z_array = torch.tensor(z_array, dtype=torch.float32)

            y_array = y_array.cuda()
            z_array = z_array.cuda()

            big_output = model(z_array, c=y_array, rev=True)
            big_output = big_output.cpu().detach()

            big_output[:, -2] = big_output[:, -2] * aa_std + aa_mean
            big_output[:, -1] = big_output[:, -1] * cc_std + cc_mean
            outputs.append(big_output)

    return outputs


def process_outputs_bp(big_output, true_stats):
    n_Rwaves = true_stats[2]

    aa = np.array(big_output[:, -2])
    cc = np.array(big_output[:, -1])

    output_bp = []
    output_constants = []
    for i in range(len(big_output)):
        if aa[i] < 188 or aa[i] > 400 or cc[i] < 1 or cc[i] > aa[i]:
            continue
        x_re = big_output[:, :-2][i].reshape((175, 8))
        output_bp_sub = []
        lvl1 = []
        lvl2 = []
        lvl3 = []
        for j in range(100):
            nearest = find_nearest(x_re[j], 1)
            if np.abs(x_re[j][nearest] - 1) < 0.5:
                lvl1.append(nearest)
            else:
                lvl1.append(-1)
        for j in range(100, 150):
            nearest = find_nearest(x_re[j], 1)
            if np.abs(x_re[j][nearest] - 1) < 0.5:
                lvl2.append(nearest)
            else:
                lvl2.append(-1)
        for j in range(150, 175):
            nearest = find_nearest(x_re[j], 1)
            if np.abs(x_re[j][nearest] - 1) < 0.5:
                lvl3.append(nearest)
            else:
                lvl3.append(-1)

        lvl1_elist = np.where(np.array(lvl1) == -1)[0]
        lvl2_elist = np.where(np.array(lvl2) == -1)[0]
        lvl3_elist = np.where(np.array(lvl3) == -1)[0]
        lvl1_end = len(lvl1)
        lvl2_end = len(lvl2)
        lvl3_end = len(lvl3)
        if len(lvl1_elist) != 0:
            lvl1_end = lvl1_elist[0]
            if len(np.where(np.array(lvl1[lvl1_end:]) == -1)[0]) != len(lvl1[lvl1_end:]):
                continue
        if len(lvl2_elist) != 0:
            lvl2_end = lvl2_elist[0]
            if len(np.where(np.array(lvl2[lvl2_end:]) == -1)[0]) != len(lvl2[lvl2_end:]):
                continue
        if len(lvl3_elist) != 0:
            lvl3_end = lvl3_elist[0]
            if len(np.where(np.array(lvl3[lvl3_end:]) == -1)[0]) != len(lvl3[lvl3_end:]):
                continue
        output_bp_sub.append(lvl1[:lvl1_end])
        output_bp_sub.append(lvl2[:lvl2_end])
        output_bp_sub.append(lvl3[:lvl3_end])
        output_bp.append(output_bp_sub)
        output_constants.append([float(aa[i]), float(cc[i])])

    filtered_bp = []
    for i in range(len(output_bp)):
        bp, bp_type = datagen.check_block_pattern_alt(output_bp[i], n_Rwaves)
        if len(bp) != 0:
            filtered_bp.append([bp, bp_type, output_constants[i]])

    return (filtered_bp, true_stats)


def get_outputs_bp_recurrent(true_stats):
    with torch.no_grad():
        aa_mean = np.loadtxt("aa_mean_est.csv")
        aa_std = np.loadtxt("aa_std_est.csv")
        cc_mean = np.loadtxt("cc_mean_est.csv")
        cc_std = np.loadtxt("cc_std_est.csv")
        y_mean = np.loadtxt("y_mean_est.csv")
        y_std = np.loadtxt("y_std_est.csv")
        config.n_x_features = 26
        config.n_cond_features = 24
        config.rnn_layers = 2
        config.hidden_size = 64
        model_rcINN, optim_rcINN, weight_scheduler_rcINN = Model.generate_rcINN_old()
        config.n_x_features = 188
        config.n_hidden_layer_size = 512
        model_seq, optim_seq, weight_scheduler_seq = Model.generate_cINN_old()
        Model.load("model_bp_rcINN.pth", optim_rcINN, model_rcINN)
        Model.load("model_bp_sequence.pth", optim_seq, model_seq)
        model_rcINN.eval()
        model_seq.eval()

        seq_len_total = []
        for stat in true_stats:
            y_seq = np.stack([(stat[0] - y_mean) / y_std]*1000, axis=0)
            z_seq = np.random.randn(1000, 188)

            y_seq = torch.tensor(y_seq, dtype=torch.float32)
            z_seq = torch.tensor(z_seq, dtype=torch.float32)

            y_seq = y_seq.cuda()
            z_seq = z_seq.cuda()

            output_seq = model_seq(z_seq, c=y_seq, rev=True)
            output_seq = output_seq.cpu().detach()

            seq_lengths = []
            for i in range(len(output_seq)):
                seq_lengths.append(find_nearest(output_seq[i], 1) + 6)

            seq_len_total.append(seq_lengths)

        final_lengths = []
        for lens in seq_len_total:
            occurence = []
            len_track = []
            for len_i in lens:
                if len_i not in len_track:
                    len_track.append(len_i)
                    counter = 0
                    for len_j in lens:
                        if len_j == len_i:
                            counter += 1
                    occurence.append(counter)
            occurence = np.array(occurence)
            len_track = np.array(len_track)
            idx_occ = np.argsort(occurence)[::-1]
            len_track = len_track[idx_occ]
            final_lengths.append(len_track[:10])

        outputs = []
        for j in range(len(final_lengths)):
            big_output = []
            for seq_len in final_lengths[j]:
                y_array = np.stack(
                    [(true_stats[j][0] - y_mean) / y_std]*seq_len, axis=1)
                y_array = np.stack([y_array]*500, axis=0)
                z_array = np.random.randn(500, 26, seq_len)

                y_array = torch.tensor(y_array, dtype=torch.float32)
                z_array = torch.tensor(z_array, dtype=torch.float32)

                y_array = y_array.cuda()
                z_array = z_array.cuda()

                output = model_rcINN(z_array, c=y_array,
                                     rev=True, recurrent=True)
                output = output.cpu().detach()

                output[:, -2, :] = output[:, -2, :] * aa_std + aa_mean
                output[:, -1, :] = output[:, -1, :] * cc_std + cc_mean

                big_output.append(output)

            outputs.append(big_output)

    return outputs


def process_outputs_bp_recurrent(outputs, true_stats):
    n_Rwaves = true_stats[2]

    filtered_bp = []

    for i in range(len(outputs)):
        for j in range(len(outputs[i])):
            outputs_ij = np.array(outputs[i][j])
            aa_i = float(np.mean(outputs_ij[-2]))
            cc_i = float(np.mean(outputs_ij[-1]))
            if aa_i < 188 or aa_i > 400 or cc_i < 1 or cc_i > aa_i:
                continue
            outputs_ij = outputs_ij.T
            seq_i = []
            for time_step in outputs_ij:
                lvl1 = find_nearest(time_step[0:8], 1)
                lvl2 = find_nearest(time_step[8:16], 1)
                lvl3 = find_nearest(time_step[16:24], 1)
                seq_i.append([lvl1, lvl2, lvl3])
            bp_i = datagen.seq_to_block_pattern(seq_i)
            bp_i_checked, bp_type = datagen.check_block_pattern_alt(
                bp_i, n_Rwaves)
            if len(bp_i_checked) != 0:
                filtered_bp.append([bp_i_checked, bp_type, [aa_i, cc_i]])

    return (filtered_bp, true_stats)


def get_outputs_bp_recurrent_matching(true_stats, mp=True):
    with torch.no_grad():
        aa_mean = np.loadtxt("aa_mean_est.csv")
        aa_std = np.loadtxt("aa_std_est.csv")
        cc_mean = np.loadtxt("cc_mean_est.csv")
        cc_std = np.loadtxt("cc_std_est.csv")
        y_mean = np.loadtxt("y_mean_est.csv")
        y_std = np.loadtxt("y_std_est.csv")
        cond_mean = np.loadtxt("cond_mean_est.csv")
        cond_std = np.loadtxt("cond_std_est.csv")
        config.n_x_features = 26
        config.n_cond_features = 2
        config.rnn_layers = 2
        config.hidden_size = 64
        model_rcINN, optim_rcINN, weight_scheduler_rcINN = Model.generate_rcINN_old()
        config.n_x_features = 188
        config.n_cond_features = 24
        config.n_hidden_layer_size = 512
        model_seq, optim_seq, weight_scheduler_seq = Model.generate_cINN_old()
        config.n_x_features = 192
        model_matching, optim_matching, weight_scheduler_matching = Model.generate_cINN_old()
        Model.load("model_bp_rcINN_matching.pth", optim_rcINN, model_rcINN)
        Model.load("model_bp_sequence.pth", optim_seq, model_seq)
        Model.load("model_bp_matching.pth", optim_matching, model_matching)
        model_rcINN.eval()
        model_seq.eval()
        model_matching.eval()

        seq_len_total = []
        for stat in true_stats:
            y_seq = np.stack([(stat[0] - y_mean) / y_std]*1000, axis=0)
            z_seq = np.random.randn(1000, 188)

            y_seq = torch.tensor(y_seq, dtype=torch.float32)
            z_seq = torch.tensor(z_seq, dtype=torch.float32)

            y_seq = y_seq.cuda()
            z_seq = z_seq.cuda()

            output_seq = model_seq(z_seq, c=y_seq, rev=True)
            output_seq = output_seq.cpu().detach()

            seq_lengths = []
            for i in range(len(output_seq)):
                seq_lengths.append(find_nearest(output_seq[i], 1) + 6)

            seq_len_total.append(seq_lengths)

        final_lengths = []
        for lens in seq_len_total:
            occurence = []
            len_track = []
            for len_i in lens:
                if len_i not in len_track:
                    len_track.append(len_i)
                    counter = 0
                    for len_j in lens:
                        if len_j == len_i:
                            counter += 1
                    occurence.append(counter)
            occurence = np.array(occurence)
            len_track = np.array(len_track)
            idx_occ = np.argsort(occurence)[::-1]
            len_track = len_track[idx_occ]
            final_lengths.append(len_track[:10])

        match_totals = []
        for k in range(len(true_stats)):
            y_matching = np.stack(
                [(true_stats[k][0] - y_mean) / y_std]*5000, axis=0)
            z_matching = np.random.randn(5000, 192)

            y_matching = torch.tensor(y_matching, dtype=torch.float32)
            z_matching = torch.tensor(z_matching, dtype=torch.float32)

            y_matching = y_matching.cuda()
            z_matching = z_matching.cuda()

            output_matching = model_matching(
                z_matching, c=y_matching, rev=True)
            output_matching = output_matching.cpu().detach()

            match_totals.append((output_matching, true_stats[k][2], k))

        if mp:
            with Pool(os.cpu_count() - 1) as pool:
                processed_matches = pool.map(process_matching, match_totals)
            processed_matches = sorted(processed_matches, key=lambda x: x[1])
        else:
            processed_matches = []
            processed_matches.append(process_matching(match_totals[0]))

        final_matches = []
        for matches in processed_matches:
            occurence = []
            match_track = []
            for match_i in matches[0]:
                if match_i not in match_track:
                    match_track.append(match_i)
                    counter = 0
                    for match_j in matches[0]:
                        if match_j == match_i:
                            counter += 1
                    occurence.append(counter)
            occurence = np.array(occurence)
            match_track = np.array(match_track)
            idx_occ = np.argsort(occurence)[::-1]
            match_track = match_track[idx_occ]
            final_matches.append(match_track)

        paired_totals = []
        for k in range(len(final_lengths)):
            paired_matchings = []
            for i in range(len(final_lengths[k])):
                sub_match = []
                for j in range(len(final_matches[k])):
                    if final_matches[k][j][-1] + 1 == final_lengths[k][i]:
                        sub_match.append(final_matches[k][j])
                paired_matchings.append((final_lengths[k][i], sub_match))
            paired_totals.append(paired_matchings)

        outputs = []
        for k in range(len(paired_totals)):
            y_rcINN = []
            z_rcINN = []

            for i in range(len(paired_totals[k])):
                if len(paired_totals[k][i][1]) == 0:
                    continue
                z_stack = []
                cond_stack = []
                counter = 0
                for j in range(len(paired_totals[k][i][1])):
                    z_stack.append(np.random.randn(
                        50, 26, paired_totals[k][i][0]))
                    cond_stack.append(np.stack([datagen.y_to_cond(
                        paired_totals[k][i][1][j], paired_totals[k][i][0], true_stats[k][0])]*50, axis=0))
                    counter += 1
                    if counter == 10:
                        break
                z_stack = np.concatenate(z_stack, axis=0)
                cond_stack = np.concatenate(cond_stack, axis=0)
                y_rcINN.append(cond_stack)
                z_rcINN.append(z_stack)

            for i in range(len(y_rcINN)):
                y_rcINN[i] = torch.tensor(y_rcINN[i], dtype=torch.float32)
                y_rcINN[i][:, 0, :] = (
                    y_rcINN[i][:, 0, :] - cond_mean[0]) / cond_std[0]
                y_rcINN[i][:, 1, :] = (
                    y_rcINN[i][:, 1, :] - cond_mean[1]) / cond_std[1]
                y_rcINN[i] = torch.tensor(y_rcINN[i], dtype=torch.float32)
                z_rcINN[i] = torch.tensor(z_rcINN[i], dtype=torch.float32)

            big_output = []

            for i in range(len(y_rcINN)):
                y_rcINN[i] = y_rcINN[i].cuda()
                z_rcINN[i] = z_rcINN[i].cuda()

                output_rcINN = model_rcINN(
                    z_rcINN[i], c=y_rcINN[i], rev=True, recurrent=True)
                output_rcINN = output_rcINN.cpu().detach()

                output_rcINN[:, -2, :] = output_rcINN[:, -2, :] * \
                    aa_std + aa_mean
                output_rcINN[:, -1, :] = output_rcINN[:, -1, :] * \
                    cc_std + cc_mean

                big_output.append(output_rcINN)

            outputs.append(big_output)

    return outputs


def process_outputs_bp_recurrent_matching(outputs, true_stats):
    n_Rwaves = true_stats[2]

    filtered_bp = []

    for i in range(len(outputs)):
        for j in range(len(outputs[i])):
            outputs_ij = np.array(outputs[i][j])
            aa_i = float(np.mean(outputs_ij[-2]))
            cc_i = float(np.mean(outputs_ij[-1]))
            if aa_i < 188 or aa_i > 400 or cc_i < 1 or cc_i > aa_i:
                continue
            outputs_ij = outputs_ij.T
            seq_i = []
            for time_step in outputs_ij:
                lvl1 = find_nearest(time_step[0:8], 1)
                lvl2 = find_nearest(time_step[8:16], 1)
                lvl3 = find_nearest(time_step[16:24], 1)
                seq_i.append([lvl1, lvl2, lvl3])
            bp_i = datagen.seq_to_block_pattern(seq_i)
            bp_i_checked, bp_type = datagen.check_block_pattern_alt(
                bp_i, n_Rwaves)
            if len(bp_i_checked) != 0:
                filtered_bp.append([bp_i_checked, bp_type, [aa_i, cc_i]])

    return (filtered_bp, true_stats)


def get_solution_mp(intervals, network_name):
    intervals = np.array(intervals)
    stats = []
    for interval in intervals:
        y_i = np.zeros(24)
        n_Rwaves = len(interval) + 1
        y_i[:(n_Rwaves-1)] = interval
        stats_i = [y_i, interval, n_Rwaves]
        stats.append(stats_i)

    if network_name == "bp_cINN":
        outputs = get_outputs_bp(stats)
        pairs = [(x, y) for x, y in zip(outputs, stats)]
        with Pool(os.cpu_count() - 1) as pool:
            filtered_outputs = pool.starmap(process_outputs_bp, pairs)
        for filter_pair in filtered_outputs:
            print_stats(filter_pair[0], filter_pair[1],
                        network_name, signals=False, splitter=False)

    if network_name == "bp_cINN_multi":
        outputs_1 = get_outputs_splitter(stats, "1")
        outputs_2a = get_outputs_splitter(stats, "2a")
        outputs_2b = get_outputs_splitter(stats, "2b")
        outputs_2c = get_outputs_splitter(stats, "2c")
        outputs_3 = get_outputs_splitter(stats, "3")

        inputs1 = [(x, y, "1", i) for x, y, i in zip(
            outputs_1, stats, [k for k in range(len(stats))])]
        inputs2a = [(x, y, "2a", i) for x, y, i in zip(
            outputs_2a, stats, [k for k in range(len(stats))])]
        inputs2b = [(x, y, "2b", i) for x, y, i in zip(
            outputs_2b, stats, [k for k in range(len(stats))])]
        inputs2c = [(x, y, "2c", i) for x, y, i in zip(
            outputs_2c, stats, [k for k in range(len(stats))])]
        inputs3 = [(x, y, "3", i) for x, y, i in zip(
            outputs_3, stats, [k for k in range(len(stats))])]

        filtered_bp_total = []
        with Pool(os.cpu_count() - 1) as pool:
            filtered_bp1 = pool.map(process_outputs_splitter, inputs1)
            filtered_bp2a = pool.map(process_outputs_splitter, inputs2a)
            filtered_bp2b = pool.map(process_outputs_splitter, inputs2b)
            filtered_bp2c = pool.map(process_outputs_splitter, inputs2c)
            filtered_bp3 = pool.map(process_outputs_splitter, inputs3)

        filtered_bp1 = sorted(filtered_bp1, key=lambda x: x[-1])
        filtered_bp2a = sorted(filtered_bp2a, key=lambda x: x[-1])
        filtered_bp2b = sorted(filtered_bp2b, key=lambda x: x[-1])
        filtered_bp2c = sorted(filtered_bp2c, key=lambda x: x[-1])
        filtered_bp3 = sorted(filtered_bp3, key=lambda x: x[-1])

        for i in range(len(filtered_bp1)):
            sub = []
            sub.extend(filtered_bp1[i][0])
            sub.extend(filtered_bp2a[i][0])
            sub.extend(filtered_bp2b[i][0])
            sub.extend(filtered_bp2c[i][0])
            sub.extend(filtered_bp3[i][0])

            filtered_bp_total.append((sub, filtered_bp1[i][1]))

        for filter_pair in filtered_bp_total:
            print_stats(filter_pair[0], filter_pair[1],
                        network_name, signals=False, splitter=True)

    if network_name == "bp_rcINN":
        outputs = get_outputs_bp_recurrent(stats)
        pairs = [(x, y) for x, y in zip(outputs, stats)]
        with Pool(os.cpu_count() - 1) as pool:
            filtered_outputs = pool.starmap(
                process_outputs_bp_recurrent, pairs)
        for filter_pair in filtered_outputs:
            print_stats(filter_pair[0], filter_pair[1],
                        network_name, signals=False, splitter=False)

    if network_name == "bp_rcINN_matching":
        outputs = get_outputs_bp_recurrent_matching(stats, mp=True)
        pairs = [(x, y) for x, y in zip(outputs, stats)]
        with Pool(os.cpu_count() - 1) as pool:
            filtered_outputs = pool.starmap(
                process_outputs_bp_recurrent_matching, pairs)
        for filter_pair in filtered_outputs:
            print_stats(filter_pair[0], filter_pair[1],
                        network_name, signals=False, splitter=False)

    if network_name == "signal_cINN":
        outputs = get_outputs_signals(stats)
        pairs = [(x, y) for x, y in zip(outputs, stats)]
        with Pool(os.cpu_count() - 1) as pool:
            filtered_outputs = pool.starmap(process_outputs_signals, pairs)
        for filter_pair in filtered_outputs:
            print_stats(filter_pair[0], filter_pair[1],
                        network_name, signals=True, splitter=False)

    if network_name == "signal_rcINN":
        outputs = get_outputs_signals_recurrent(stats)
        pairs = [(x, y) for x, y in zip(outputs, stats)]
        with Pool(os.cpu_count() - 1) as pool:
            filtered_outputs = pool.starmap(
                process_outputs_signals_recurrent, pairs)
        for filter_pair in filtered_outputs:
            print_stats(filter_pair[0], filter_pair[1],
                        network_name, signals=True, splitter=False)

    if network_name == "signal_rcINN_matching":
        outputs = get_outputs_signals_recurrent_matching(stats, mp=True)
        pairs = [(x, y) for x, y in zip(outputs, stats)]
        with Pool(os.cpu_count() - 1) as pool:
            filtered_outputs = pool.starmap(
                process_outputs_signals_recurrent_matching, pairs)
        for filter_pair in filtered_outputs:
            print_stats(filter_pair[0], filter_pair[1],
                        network_name, signals=True, splitter=False)


def get_solution(intervals, network_name):
    intervals = np.array(intervals)
    y_i = np.zeros(24)
    n_Rwaves = len(intervals) + 1
    y_i[:(n_Rwaves-1)] = intervals
    stats = [y_i, intervals, n_Rwaves]

    if network_name == "bp_cINN":
        outputs = get_outputs_bp([stats])
        filter_output = process_outputs_bp(outputs[0], stats)
        print_stats(filter_output[0], stats,
                    network_name, signals=False, splitter=False)

    if network_name == "bp_cINN_multi":
        outputs_1 = get_outputs_splitter([stats], "1")
        outputs_2a = get_outputs_splitter([stats], "2a")
        outputs_2b = get_outputs_splitter([stats], "2b")
        outputs_2c = get_outputs_splitter([stats], "2c")
        outputs_3 = get_outputs_splitter([stats], "3")

        filter_output_1 = process_outputs_splitter(
            (outputs_1[0], stats, "1", 0))
        filter_output_2a = process_outputs_splitter(
            (outputs_2a[0], stats, "2a", 0))
        filter_output_2b = process_outputs_splitter(
            (outputs_2b[0], stats, "2b", 0))
        filter_output_2c = process_outputs_splitter(
            (outputs_2c[0], stats, "2c", 0))
        filter_output_3 = process_outputs_splitter(
            (outputs_3[0], stats, "3", 0))

        filter_total = []
        filter_total.extend(filter_output_1[0])
        filter_total.extend(filter_output_2a[0])
        filter_total.extend(filter_output_2b[0])
        filter_total.extend(filter_output_2c[0])
        filter_total.extend(filter_output_3[0])

        print_stats(filter_total, stats, network_name,
                    signals=False, splitter=True)

    if network_name == "bp_rcINN":
        outputs = get_outputs_bp_recurrent([stats])
        filter_output = process_outputs_bp_recurrent(outputs[0], stats)
        print_stats(filter_output[0], stats,
                    network_name, signals=False, splitter=False)

    if network_name == "bp_rcINN_matching":
        outputs = get_outputs_bp_recurrent_matching([stats], mp=False)
        filter_output = process_outputs_bp_recurrent_matching(
            outputs[0], stats)
        print_stats(filter_output[0], stats,
                    network_name, signals=False, splitter=False)

    if network_name == "signal_cINN":
        outputs = get_outputs_signals([stats])
        filter_output = process_outputs_signals(outputs[0], stats)
        print_stats(filter_output[0], stats,
                    network_name, signals=True, splitter=False)

    if network_name == "signal_rcINN":
        outputs = get_outputs_signals_recurrent([stats])
        filter_output = process_outputs_signals_recurrent(outputs[0], stats)
        print_stats(filter_output[0], stats,
                    network_name, signals=True, splitter=False)

    if network_name == "signal_rcINN_matching":
        outputs = get_outputs_signals_recurrent_matching([stats], mp=False)
        filter_output = process_outputs_signals_recurrent_matching(
            outputs[0], stats)
        print_stats(filter_output[0], stats,
                    network_name, signals=True, splitter=False)


def main():
    total_intervals = []
    for i in range(100):
        y = datagen.get_random_y()
        intervals = y[1][0:24]
        total_intervals.append(intervals)
    network_name = "bp_cINN_multi"
    get_solution_mp(total_intervals, network_name)


"""
def main():
    model, optim, weight_scheduler = Model.generate_cINN_old()
    old_avg_loss = float("inf")
    for i in range(config.n_epochs):
        print("epoch: ", i)
        avg_loss = train_splitter(model, optim)
        print(avg_loss)
        if avg_loss < old_avg_loss:
            Model.save("model_splitter3.pth", optim, model)
            old_avg_loss = avg_loss
        weight_scheduler.step()
"""


"""
def main():
    with open("test_y3.txt", "r") as fp:
        true_stats = json.load(fp)
    
    for i in range(len(true_stats)):
        true_stats[i][0] = np.array(true_stats[i][0])
        true_stats[i][1] = np.array(true_stats[i][1]) 
        
    outputs = get_outputs_bp(true_stats[750:1000])
    pairs = [(x,y) for x,y in zip(outputs, true_stats[750:1000])]
    print("done with GPU")
    with Pool(os.cpu_count() - 1) as pool:
        filtered_bp = pool.starmap(process_outputs_bp, pairs)
    print("done with multiprocessing")
    for filter_pair in filtered_bp:
        make_stats(filter_pair[0], filter_pair[1], "cINN_bp" + "3", signals=False, splitter=False)
"""

"""
def main():
    with open("test_y3.txt", "r") as fp:
        true_stats3 = json.load(fp)
    
    for i in range(len(true_stats3)):
        true_stats3[i][0] = np.array(true_stats3[i][0])
        true_stats3[i][1] = np.array(true_stats3[i][1])
    
    outputs1 = get_outputs_splitter(true_stats3, "1")
    outputs2a = get_outputs_splitter(true_stats3, "2a")
    outputs2b = get_outputs_splitter(true_stats3, "2b")
    outputs2c = get_outputs_splitter(true_stats3, "2c")
    outputs3 = get_outputs_splitter(true_stats3, "3")
    
    inputs1 = [(x, y, "1", i) for x, y, i in zip(outputs1, true_stats3, [k for k in range(1000)])]
    inputs2a = [(x, y, "2a", i) for x, y, i in zip(outputs2a, true_stats3, [k for k in range(1000)])]
    inputs2b = [(x, y, "2b", i) for x, y, i in zip(outputs2b, true_stats3, [k for k in range(1000)])]
    inputs2c = [(x, y, "2c", i) for x, y, i in zip(outputs2c, true_stats3, [k for k in range(1000)])]
    inputs3 = [(x, y, "3", i) for x, y, i in zip(outputs3, true_stats3, [k for k in range(1000)])]
    
    filtered_bp_total = []
    with Pool(os.cpu_count() - 1) as pool:
        filtered_bp1 = pool.map(process_outputs_splitter, inputs1)
        filtered_bp2a = pool.map(process_outputs_splitter, inputs2a)
        filtered_bp2b = pool.map(process_outputs_splitter, inputs2b)
        filtered_bp2c = pool.map(process_outputs_splitter, inputs2c)
        filtered_bp3 = pool.map(process_outputs_splitter, inputs3)
    
    filtered_bp1 = sorted(filtered_bp1, key=lambda x: x[-1])
    filtered_bp2a = sorted(filtered_bp2a, key=lambda x: x[-1])
    filtered_bp2b = sorted(filtered_bp2b, key=lambda x: x[-1])
    filtered_bp2c = sorted(filtered_bp2c, key=lambda x: x[-1])
    filtered_bp3 = sorted(filtered_bp3, key=lambda x: x[-1])
    
    for i in range(len(filtered_bp1)):
        sub = []
        sub.extend(filtered_bp1[i][0])
        sub.extend(filtered_bp2a[i][0])
        sub.extend(filtered_bp2b[i][0])
        sub.extend(filtered_bp2c[i][0])
        sub.extend(filtered_bp3[i][0])
        
        filtered_bp_total.append((sub, filtered_bp1[i][1]))

    for filter_pair in filtered_bp_total:
        make_stats(filter_pair[0], filter_pair[1], "splitter" + "3", signals=False, splitter=True)

"""

"""
def main():
    with open("test_y1.txt", "r") as fp:
        true_stats1 = json.load(fp)
    with open("test_y2a.txt", "r") as fp:
        true_stats2a = json.load(fp)
    with open("test_y2b.txt", "r") as fp:
        true_stats2b = json.load(fp)
    with open("test_y2c.txt", "r") as fp:
        true_stats2c = json.load(fp)
    with open("test_y3.txt", "r") as fp:
        true_stats3 = json.load(fp)
    
    for i in range(len(true_stats1)):
        true_stats1[i][0] = np.array(true_stats1[i][0])
        true_stats1[i][1] = np.array(true_stats1[i][1])
    for i in range(len(true_stats2a)):
        true_stats2a[i][0] = np.array(true_stats2a[i][0])
        true_stats2a[i][1] = np.array(true_stats2a[i][1])
    for i in range(len(true_stats2b)):
        true_stats2b[i][0] = np.array(true_stats2b[i][0])
        true_stats2b[i][1] = np.array(true_stats2b[i][1])
    for i in range(len(true_stats2c)):
        true_stats2c[i][0] = np.array(true_stats2c[i][0])
        true_stats2c[i][1] = np.array(true_stats2c[i][1])
    for i in range(len(true_stats3)):
        true_stats3[i][0] = np.array(true_stats3[i][0])
        true_stats3[i][1] = np.array(true_stats3[i][1])
    print("stats ready")
    
    for stat in true_stats1:
        splitter_type = ["1", "2a", "2b", "2c", "3"]
        filtered_bp_total = []
        for sp_type in splitter_type:
            filtered_bp = test_splitter(sp_type, stat)
            filtered_bp_total.extend(filtered_bp)
        make_stats(filtered_bp_total, stat, "splitter" + "1", signals=False, splitter=True)
        torch.cuda.empty_cache()
    print("1 done")
    for stat in true_stats2a:
        splitter_type = ["1", "2a", "2b", "2c", "3"]
        filtered_bp_total = []
        for sp_type in splitter_type:
            filtered_bp = test_splitter(sp_type, stat)
            filtered_bp_total.extend(filtered_bp)
        make_stats(filtered_bp_total, stat, "splitter" + "2a", signals=False, splitter=True)
        torch.cuda.empty_cache()
    print("2a done")
    for stat in true_stats2b:
        splitter_type = ["1", "2a", "2b", "2c", "3"]
        filtered_bp_total = []
        for sp_type in splitter_type:
            filtered_bp = test_splitter(sp_type, stat)
            filtered_bp_total.extend(filtered_bp)
        make_stats(filtered_bp_total, stat, "splitter" + "2b", signals=False, splitter=True)
        torch.cuda.empty_cache()
    print("2b done")
    for stat in true_stats2c:
        splitter_type = ["1", "2a", "2b", "2c", "3"]
        filtered_bp_total = []
        for sp_type in splitter_type:
            filtered_bp = test_splitter(sp_type, stat)
            filtered_bp_total.extend(filtered_bp)
        make_stats(filtered_bp_total, stat, "splitter" + "2c", signals=False, splitter=True)
        torch.cuda.empty_cache()
    print("2c done")
    for stat in true_stats3:
        splitter_type = ["1", "2a", "2b", "2c", "3"]
        filtered_bp_total = []
        for sp_type in splitter_type:
            filtered_bp = test_splitter(sp_type, stat)
            filtered_bp_total.extend(filtered_bp)
        make_stats(filtered_bp_total, stat, "splitter" + "3", signals=False, splitter=True)
        torch.cuda.empty_cache()
    print("3 done")      
"""


if __name__ == "__main__":
    main()
