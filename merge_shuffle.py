import numpy as np
from numba import cuda, jit, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

@jit(nopython=True)
def fisher_yates_shuffle(arr, n, rng_states):
    for i in range(n - 1, 0, -1):
        j = int(xoroshiro128p_uniform_float32(rng_states, i) * (i + 1))
        arr[i], arr[j] = arr[j], arr[i]

@cuda.jit
def merge(arr, start, mid, end, rng_states):
    # Assuming these arrays fit within shared memory limits
    left = cuda.shared.array(shape=(1024,), dtype=int32)
    right = cuda.shared.array(shape=(1024,), dtype=int32)
    left_size = mid - start
    right_size = end - mid

    for i in range(left_size):
        left[i] = arr[start + i]
    for i in range(right_size):
        right[i] = arr[mid + i]

    i = j = 0
    k = start

    while i < left_size and j < right_size:
        if xoroshiro128p_uniform_float32(rng_states, k) < 0.5:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < left_size:
        arr[k] = left[i]
        i += 1
        k += 1

    while j < right_size:
        arr[k] = right[j]
        j += 1
        k += 1

    for i in range(start, end):
        swap_index = int(xoroshiro128p_uniform_float32(rng_states, i) * (i - start + 1)) + start
        arr[i], arr[swap_index] = arr[swap_pos], arr[i]

@cuda.jit
def merge_shuffle(arr, n, k, rng_states):
    if n <= k:
        fisher_yates_shuffle(arr, n, rng_states)
        return

    block_size = (n + 2 * k - 1) // (2 * k)  # Ensure at least 2k blocks
    num_blocks = (n + block_size - 1) // block_size

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, n)
        fisher_yates_shuffle(arr[start:end], end - start, rng_states[block_idx])

    stride = block_size
    while stride < n:
        for i in range(0, n, 2 * stride):
            mid = min(i + stride, n)
            end = min(i + 2 * stride, n)
            merge(arr, i, mid, end, rng_states)
        stride *= 2

def generate_random_input(size):
    return np.random.randint(0, 10000, size)

def write_output(filename, arr):
    with open(filename, 'w') as file:
        for item in arr:
            file.write(f"{item}\n")

def check_elements(original, shuffled):
    return set(original) == set(shuffled)

def calculate_shuffle_percentage(original, shuffled):
    match_count = sum(1 for o, s in zip(original, shuffled) if o == s)
    total_count = len(original)
    return (total_count - match_count) / total_count * 100

if __name__ == "__main__":
    output_filename = "output.txt"

    elements = generate_random Python_input(1000)  # Generate 1000 random elements
    original_elements = elements.copy()

    # Convert the list to a numpy array for CUDA compatibility
    elements_np = np.array(elements, dtype=np.int32)

    # Allocate memory on the GPU
    d_elements = cuda.to_device(elements_np)

    # Create random number generator states for each thread
    rng_states = create_xoroshiro128p_states(1000, seed=1)

    # Run the shuffle on the GPU
    merge_shuffle[1, 1](d_elements, len(elements_np), 32, rng_states)

    # Copy the result back to the host
    elements_np = d_elements.copy_to_host()

    # Convert back to a list for further processing
    elements = elements_np.tolist()

    if check_elements(original_elements, elements):
        write_output(output_filename, elements)
        shuffle_percentage = calculate_shuffle_percentage(original_elements, elements)
        print(f"Shuffle Percentage: {shuffle_percentage:.2f}%")
    else:
        print("Error: Elements missing after shuffling!")

