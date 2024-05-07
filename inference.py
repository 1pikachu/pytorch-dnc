import numpy as np
import torch
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

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

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
        debug=debug,
        device=args.device
    ).to(args.device)
    rnn.eval()
    if args.device == "xpu":
        datatype = torch.float16 if args.precision == "float16" else torch.bfloat16 if args.precision == "bfloat16" else torch.float
        rnn = torch.xpu.optimize(model=rnn, dtype=datatype)

    input_data, target_output = generate_data(batch_size, length, input_size, args.device)

    if args.channels_last:
        rnn = rnn.to(memory_format=torch.channels_last)
        input_data = input_data.to(memory_format=torch.channels_last) if len(input_data.size()) == 4 else input_data
    # disable jit, jit make slow
    #if args.jit:
    #    try:
    #        rnn = torch.jit.trace(rnn, input_data, check_trace=False, strict=False)
    #        print("---- JIT trace enable.")
    #    except (RuntimeError, TypeError) as e:
    #        print("---- JIT trace disable.")
    #        print("failed to use PyTorch jit mode due to: ", e)
    if args.nv_fuser:
       fuser_mode = "fuser2"
    else:
       fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    if args.compile:
        print("----enable compiler")
        rnn = torch.compile(rnn, backend=args.backend, options={"freezing": True})
    
    total_time = 0.0
    total_sample = 0

    if args.profile and args.device == "xpu":
        for i in range(args.num_iter + args.num_warmup):
            input_data, target_output = generate_data(batch_size, length, input_size, "cpu")
            with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                elapsed = time.time()
                input_data = input_data.to(args.device)
                output, (chx, mhx, rv) = rnn(input_data)
                torch.xpu.synchronize()
                elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
            if args.profile and i == int((args.num_iter + args.num_warmup)/2):
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
                torch.save(prof.table(sort_by="id", row_limit=100000),
                    timeline_dir+'profile_detail_withId.pt')
                prof.export_chrome_trace(timeline_dir+"trace.json")    
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter + args.num_warmup)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter + args.num_warmup):
                input_data, target_output = generate_data(batch_size, length, input_size, "cpu")
                elapsed = time.time()
                input_data = input_data.to(args.device)
                with torch.jit.fuser(fuser_mode):
                    output, (chx, mhx, rv) = rnn(input_data)
                torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif args.profile and args.device == "cpu":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int((args.num_iter + args.num_warmup)/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter + args.num_warmup):
                input_data, target_output = generate_data(batch_size, length, input_size, "cpu")
                elapsed = time.time()
                input_data = input_data.to(args.device)
                output, (chx, mhx, rv) = rnn(input_data)
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif not args.profile and args.device == "cuda":
        for i in range(args.num_iter + args.num_warmup):
            input_data, target_output = generate_data(batch_size, length, input_size, "cpu")
            elapsed = time.time()
            input_data = input_data.to(args.device)
            with torch.jit.fuser(fuser_mode):
                output, (chx, mhx, rv) = rnn(input_data)
            torch.cuda.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
    else:
        for i in range(args.num_iter + args.num_warmup):
            input_data, target_output = generate_data(batch_size, length, input_size, "cpu")
            elapsed = time.time()
            input_data = input_data.to(args.device)
            output, (chx, mhx, rv) = rnn(input_data)
            if args.device == "xpu":
                torch.xpu.synchronize()
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
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')
    parser.add_argument('--compile', action='store_true', default=False, help='compile model')
    parser.add_argument('--backend', default="inductor", type=str, help='backend')
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parse_args()

    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    with torch.no_grad():
        if args.precision == "float16" and args.device == "cuda":
            print("---- Use autocast fp16 cuda")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                inference(args)
        elif args.precision == "float16" and args.device == "xpu":
            print("---- Use autocast fp16 xpu")
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                inference(args)
        elif args.precision == "bfloat16" and args.device == "cpu":
            print("---- Use autocast bf16 cpu")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                inference(args)
        elif args.precision == "bfloat16" and args.device == "xpu":
            print("---- Use autocast bf16 xpu")
            with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                inference(args)
        else:
            print("---- no autocast")
            inference(args)

if __name__ == "__main__":
    main()
