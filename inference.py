import numpy as np
import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import numpy as np

import sys
import os
import math
import time
import argparse

import functools

from dnc import DNC


def generate_data(batch_size, length, size, device="cpu"):
    input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    target_output = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)

    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1))

    input_data[:, :length, :size - 1] = sequence
    input_data[:, length, -1] = 1  # the end symbol
    target_output[:, length + 1:, :size - 1] = sequence

    input_data = T.from_numpy(input_data)
    target_output = T.from_numpy(target_output)
    input_data = input_data.to(device)
    target_output = target_output.to(device)

    return var(input_data), var(target_output)

def inference(args):
    T.manual_seed(1111)

    input_size = 100
    hidden_size = 100
    rnn_type = 'rnn'
    num_layers = 1
    num_hidden_layers = 1
    dropout = 0
    nr_cells = 1
    cell_size = 1
    read_heads = 1
    gpu_id = 0 if args.device == "cuda" else -1
    debug = False
    lr = 0.001
    sequence_max_length = 10
    batch_size = args.batch_size
    clip = 10
    length = 10

    rnn = DNC(
        input_size=input_size,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        num_layers=num_layers,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        nr_cells=nr_cells,
        cell_size=cell_size,
        read_heads=read_heads,
        gpu_id=gpu_id,
        debug=debug
    ).to(args.device)

    input_data, target_output = generate_data(batch_size, length, input_size, args.device)
    
    total_time = 0.0
    total_sample = 0
    for i in range(args.num_iter + args.num_warmup):
        elapsed = time.time()
        output, (chx, mhx, rv) = rnn(input_data, None)
        elapsed = time.time() - elapsed
        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
        if i >= args.num_warmup:
            total_sample += args.batch_size
            total_time += elapsed

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parse_args()
    inference(args)

if __name__ == "__main__":
    main()
