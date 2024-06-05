import numpy as np
from numba import cuda

@cuda.jit
def gpu_merge_shuffle(elements, k):
    idx = cuda.grid(1)
    if idx < elements.size:
        # Example operation: just an identity operation for now
        elements[idx] = elements[idx]

def gpu_shuffle_task(elements, k):
    try:
        print("Allocating memory on GPU")
        d_elements = cuda.to_device(elements)
        print("Memory allocation successful")
        
        threads_per_block = 128
        blocks_per_grid = (elements.size + (threads_per_block - 1)) // threads_per_block
        
        print("Launching kernel")
        gpu_merge_shuffle[blocks_per_grid, threads_per_block](d_elements, k)
        
        print("Synchronizing GPU operations")
        cuda.synchronize()
        print("Synchronization successful")
        
        print("Copying data back to host")
        result = d_elements.copy_to_host()
        print("Data copied back to host")
        
        # Free GPU memory
        d_elements = None
        cuda.current_context().deallocations.clear()
        print("GPU memory deallocated")
        
        return result
    except cuda.CudaAPIError as e:
        print(f"CUDA API Error: {e}")
    except Exception as e:
        print(f"General Error: {e}")

def main():
    # Initialize data
    n = 1000000
    elements = np.random.rand(n).astype(np.float32)
    k = 5  # Example value for k

    # Run the GPU shuffle task
    print("Running GPU shuffle task")
    result = gpu_shuffle_task(elements, k)
    
    if result is not None:
        print(f"First 5 results: {result[:5]}")

if __name__ == "__main__":
    main()

