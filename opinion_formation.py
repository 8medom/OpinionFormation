#!/usr/bin/env python


import sys
import numpy as np
from numpy.random import rand, choice, seed


num_realizations = 100                  # how many realizations to do for a given parameter setting
num_realizations_for_stability = 100    # how many realizations of each (R, opinion_seeds) setting to do to estimate opinion stability
rule_type = 'random1'                   # options: random1 (form opinion using signal from a randomly chosen neighbor), majority (use the majority signal)


# get the sign of the connection between entities i and j
def get_sign(i, j, GT, beta):
    if GT[i] == GT[j]:                  # if i and j are from the same ground-truth camp
        if rand() < beta: return -1
        else: return 1
    else:                               #  if i and j are from the different ground-truth camps
        if rand() < beta: return 1
        else: return -1


# check if R corresponds to a connected network
def is_connected(R):
    N = R.shape[0]
    labels = np.arange(N)               # assign a distinct label to each node
    for i in range(N):
        neighbors = np.where(R[i, :] <> 0)[0]
        for j in neighbors:
            label_to_remove = labels[j] # re-label all nodes with the same label as j to the label of node i
            labels[labels == label_to_remove] = labels[i]
    if len(set(labels)) > 1:            # if more than one label remains, the network is not connected
        return False
    else: return True


# initalize a random network where the probability of two nodes connected is z / (N - 1) to achieve a desired mean degree z
def initialize_R(N, z, beta, GT):
    p_connected = float(z) / (N - 1)
    R = np.zeros((N, N), dtype = np.int8)
    while True:
        R[:, :] = 0
        for i in range(N):
            for j in range(i + 1, N):
                if rand() < p_connected:
                    R[i][j] = get_sign(i, j, GT, beta)
                    R[j][i] = R[i][j]
        if is_connected(R): break       # check if the network is connected (otherwise some opinions cannot be formed); if not, generate a new network
    return R


# measure the consistency of a given opinion vector, op, and a given ground truth vector, GT; opinions_seeds do not contribute to opinion consistency
def compute_consistency(op, GT, opinion_seeds):
    N = op.size
    C_sum = 0
    for i in range(N):
        if i not in opinion_seeds:
            C_sum += op[i] * GT[i]
    return float(C_sum) / (N - opinion_seeds.size)


# compute the stability value corresponding to a given average opinion vector op_avg
def evaluate_stability(op_avg, opinion_seeds):
    N = op_avg.size
    S_sum = 0
    for i in range(N):
        if i not in opinion_seeds:
            S_sum += abs(op_avg[i])
    S_expected_for_random = np.sqrt(2. / (num_realizations_for_stability * np.pi))
    return (float(S_sum) / (N - opinion_seeds.size) - S_expected_for_random) / (1. - S_expected_for_random)


# form opinions on all subjects
def form_opinions(R, opinion_seeds):
    N = R.shape[0]
    op = np.zeros(N, dtype = np.int8)                     # empty initial opinion vector
    for i in opinion_seeds: op[i] = 1                     # set seed opinions
    while np.where(op == 0)[0].size > 0:                  # continue until all opinions are formed
        target = np.random.choice(np.where(op == 0)[0])   # choose a target node for which opinion has not been formed yet
        signal = R[target, :] * op                        # compute the signals on the target node from all other nodes
        num_pos = np.where(signal > 0)[0].size            # count the number of positive signals
        num_neg = np.where(signal < 0)[0].size            # count the number of negative signals
        if num_pos + num_neg > 0:
            if rule_type == 'random1':                    # form opinion using one random neighbor
                random_neighbor = np.random.choice(np.where(signal <> 0)[0])
                op[target] = op[random_neighbor] * R[target, random_neighbor]
            elif rule_type == 'majority':                 # form opinion using the majority signal
                if num_pos > num_neg: op[target] = 1
                elif num_neg > num_pos: op[target] = -1
                else:
                    if np.random.rand() < 0.5: op[target] = 1
                    else: op[target] = -1
            else:
                print 'rule type {} is unknown'.format(rule_type)
                sys.exit(1)
    return op


# run simulations on random signed networks
def run_on_synthetic_networks(N = 100, z = 10, beta = 0.05, num_seed_opinions = 1):
    print 'starting simulations for N = {}, z = {}, beta = {}, rule type = {}\n'.format(N, z, beta, rule_type)
    reporting_step = num_realizations / 100
    GT = np.concatenate([np.ones(N / 2, dtype = int), -np.ones(N / 2, dtype = int)])  # the ground truth
    op_all = np.zeros((num_realizations_for_stability, N))                            # all formed opinions
    consistency_values, stability_values = [], []                                     # resulting opinion consistency and opinion stability values
    print 'completed:',
    for n in range(num_realizations):                                                 # each realization represents the opinion-making of one agent
        if (n + 1) % reporting_step == 0:                                             # progress reporting
            print '{}%'.format(1 + n / reporting_step),
            sys.stdout.flush()
        R = initialize_R(N, z, beta, GT)                                              # initialize the relation network
        seeds = np.random.choice(N / 2, size = num_seed_opinions, replace = False)    # choose num_seed_opinions from camp 1  
        for rep in range(num_realizations_for_stability):                             # num_realizations_for_stability independent model realizations for given R & seeds
            op = form_opinions(R, seeds)                                              # form all opinions
            op_all[rep, :] = op
            consistency = compute_consistency(op, GT, seeds)                          # compute the corresponding opinion consistency
            consistency_values.append(consistency)
        op_avg = np.mean(op_all, axis = 0)                                            # use num_realizations_for_stability to compute one opinion stability
        stability_values.append(evaluate_stability(op_avg, seeds))
    print '\n\nopinion consistency: mean {:.4f}, std {:.4f}, 10th percentile {:.4f}, 90th percentile {:.4f}'.format(np.mean(consistency_values), np.std(consistency_values), np.percentile(consistency_values, 10), np.percentile(consistency_values, 90))
    print '  opinion stability: mean {:.4f}, std {:.4f}, 10th percentile {:.4f}, 90th percentile {:.4f}'.format(np.mean(stability_values), np.std(stability_values), np.percentile(stability_values, 10), np.percentile(stability_values, 90))


np.random.seed(0)             # initialize random numbers generator
run_on_synthetic_networks()   # run simulations
