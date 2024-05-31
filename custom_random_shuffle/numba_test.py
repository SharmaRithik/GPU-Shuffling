import numpy as np
from numba import jit
import time

# Standard Python function to compute the sum of squares
def sum_of_squares(arr):
    total = 0.0
    for i in range(len(arr)):
        total += arr[i] ** 2
    return total

# Numba-accelerated function to compute the sum of squares
@jit(nopython=True)
def sum_of_squares_numba(arr):
    total = 0.0
    for i in range(len(arr)):
        total += arr[i] ** 2
    return total

def main():
    # Create a large array of random numbers
    arr = np.random.rand(10000000)

    # Compute the sum of squares using the standard Python function
    start_time = time.time()
    result_python = sum_of_squares(arr)
    end_time = time.time()
    python_time = end_time - start_time
    print(f"Standard Python function result: {result_python:.6f}, Time taken: {python_time:.6f} seconds")

    # Compute the sum of squares using the Numba-accelerated function
    start_time = time.time()
    result_numba = sum_of_squares_numba(arr)
    end_time = time.time()
    numba_time = end_time - start_time
    print(f"Numba function result: {result_numba:.6f}, Time taken: {numba_time:.6f} seconds")

if __name__ == "__main__":
    main()

