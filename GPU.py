import numpy as np
import time
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

@cuda.jit
def gpu_shuffle_kernel(arr, rng_states):
    """
    GPU kernel function for shuffling an array using random numbers generated from
    xoroshiro128+ RNG states.
    
    Parameters:
    arr (cuda device array): Array to be shuffled.
    rng_states (cuda device array): RNG states for generating random numbers.
    """
    n = arr.shape[0]
    if cuda.threadIdx.x == 0:
        for i in range(n - 1, 0, -1):
            j = int(xoroshiro128p_uniform_float32(rng_states, cuda.threadIdx.x) * (i + 1))
            arr[i], arr[j] = arr[j], arr[i]

def gpu_shuffle(arr):
    """
    Shuffle an array using a GPU-based algorithm.
    
    Parameters:
    arr (numpy array): Array to be shuffled.
    
    Returns:
    numpy array: Shuffled array.
    """
    n = arr.shape[0]
    d_arr = cuda.to_device(arr)
    threads_per_block = 1  # Only one thread
    blocks_per_grid = 1   # Only one block

    rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=np.random.randint(0, 100000))

    start_time = time.time()
    gpu_shuffle_kernel[blocks_per_grid, threads_per_block](d_arr, rng_states)
    cuda.synchronize()
    end_time = time.time()
    
    shuffled_arr = d_arr.copy_to_host()
    print("Shuffle time: {:.3f} seconds".format(end_time - start_time))
    return shuffled_arr

def verify_shuffling(original_data, shuffled_data):
    """
    Verify that the shuffling was performed correctly by comparing sets of original
    and shuffled data.
    
    Parameters:
    original_data (numpy array): Original array before shuffling.
    shuffled_data (numpy array): Array after shuffling.
    
    Returns:
    bool: True if verification passed, False otherwise.
    """
    original_set = set(original_data)
    shuffled_set = set(shuffled_data)
    is_correct = original_set == shuffled_set
    print(f"Verification: {'Passed' if is_correct else 'Failed'}")
    return is_correct

def main():
    """
    Main function to test GPU shuffling algorithm with different data sizes.
    """
    sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]  # Different sizes to test
    for size in sizes:
        print("\nTesting size:", size)
        original_data = np.arange(size)
        shuffled_data = gpu_shuffle(original_data.copy())
        verify_shuffling(original_data, shuffled_data)

if __name__ == "__main__":
    main()

