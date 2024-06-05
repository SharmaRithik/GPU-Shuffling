import random
import time
import math
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

@cuda.jit(cache=True, opt=True)
def fisher_yates_gpu(arr, chk_size, rng_states):
    """
    GPU kernel function for shuffling an array using random numbers generated from
    xoroshiro128+ RNG states.
    
    Parameters:
    arr (cuda device array): Array to be shuffled.
    rng_states (cuda device array): RNG states for generating random numbers.
    """
    idx = cuda.grid(1)
    chk_start = chk_size * idx
    chk_end = chk_size * (idx + 1) - 1
    for i in range(chk_end, chk_start, -1):
        j = int(xoroshiro128p_uniform_float32(rng_states, i) * (i - chk_start) + chk_start)
        arr[i], arr[j] = arr[j], arr[i]

def in_place_shuffled_merge_cpu(arr, start, mid, end):
    i = start
    j = mid
    while True:
        if random.randint(0, 1):
            if i == j:
                break
        else:
            if j == end:
                break
            arr[i], arr[j] = arr[j], arr[i]
            j += 1
        i += 1
    while i < end:
        m = random.randint(start, i)
        arr[i], arr[m] = arr[m], arr[i]
        i += 1

@cuda.jit(cache=True, opt=True)
def in_place_shuffled_merge_gpu(arr, n, chk_size, rng_states):
    idx = cuda.grid(1)
    start = idx * chk_size * 2
    mid = min(start + chk_size, n)
    end = min(start + (chk_size * 2), n)
    i = start
    j = mid
    while True:
        if round(xoroshiro128p_uniform_float32(rng_states, i)):
            if i == j:
                break
        else:
            if j == end:
                break
            arr[i], arr[j] = arr[j], arr[i]
            j += 1
        i += 1
    while i < end:
        m = int(xoroshiro128p_uniform_float32(rng_states, i) * (i - start) + start)
        arr[i], arr[m] = arr[m], arr[i]
        i += 1

def round_to_n(x, n):
    return round(x, -int(math.floor(math.log10(x))) + (n - 1))

def gpu_shuffle(arr):
    """
    Shuffle an array using merge shuffle on GPU using Numba CUDA compilation.
    
    Parameters:
    arr (numpy array): Array to be shuffled.
    
    Returns:
    tuple: Shuffled array
    """
    threads_per_block = 64  # Set the number of threads per block
    blocks_per_grid = 128
    n = arr.shape[0]
    rng_states = create_xoroshiro128p_states(n, seed=np.random.randint(0, 100000))
    chk_size = n // (blocks_per_grid * threads_per_block) 

    print(f'Dispatching Fisher-Yates to GPU with {blocks_per_grid} blocks and {threads_per_block} threads per block')
    print(f'Using chunk size {chk_size} for array of size {n}')

    t0 = time.perf_counter()
    d_arr = cuda.to_device(arr)
    t1 = time.perf_counter()
    fisher_yates_gpu[blocks_per_grid, threads_per_block](d_arr, chk_size, rng_states)
    cuda.synchronize()
    t2 = time.perf_counter()
    while chk_size * 2 < n:
        threads_per_block = 1
        blocks_per_grid = n // (chk_size * 2)
        print(f'Dispatching merge to GPU with {blocks_per_grid} blocks and {threads_per_block} threads per block')
        print(f'Using chunk size {chk_size} for array of size {n}')
        in_place_shuffled_merge_gpu[blocks_per_grid, threads_per_block](d_arr, n, chk_size, rng_states)
        cuda.synchronize()
        chk_size *= 2
    t3 = time.perf_counter()
    shuffled_arr = d_arr.copy_to_host()
    t4 = time.perf_counter()

    print(f'Device mem copy time: {round_to_n((t1 - t0) + (t4 - t3), 2)}')
    print(f'Fisher-Yates time: {round_to_n(t2 - t1, 2)}')
    print(f'Merge time: {round_to_n(t3 - t2, 2)}')

    return shuffled_arr