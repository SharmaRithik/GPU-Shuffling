import os
import ray
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the correct GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_small_dataset(size):
    """
    Create a small dataset of dictionaries with consecutive integers.
    
    Args:
    size (int): Number of elements in the dataset.

    Returns:
    ray.data.Dataset: A Ray dataset containing dictionaries.
    """
    data = [{'data': i} for i in range(size)]
    return ray.data.from_items(data)

@ray.remote(num_gpus=0)
def shuffle_batch_parallel(data_batch):
    """
    Shuffle each batch of data in parallel.

    Args:
    data_batch (list of dict): Batch of data to shuffle.

    Returns:
    list of dict: Shuffled batch of data.
    """
    batch_array = np.array([item['data'] for item in data_batch])
    np.random.shuffle(batch_array)
    shuffled_batch = [{'data': item} for item in batch_array]
    return shuffled_batch

def verify_shuffling(original_data, shuffled_data, description):
    """
    Verify that all elements are correctly shuffled by comparing sets.

    Args:
    original_data (list): The original data list.
    shuffled_data (list): The shuffled data list.
    description (str): Description of the shuffle method being verified.

    Returns:
    bool: True if verification passed, False otherwise.
    """
    original_set = set(item['data'] for item in original_data)
    shuffled_set = set(item['data'] for item in shuffled_data)
    is_correct = original_set == shuffled_set
    print(f"Verification of {description}: {'Passed' if is_correct else 'Failed'}")
    return is_correct

def ray_random_shuffle(dataset, original_data):
    """
    Shuffle the dataset using Ray's built-in random_shuffle and verify.

    Args:
    dataset (ray.data.Dataset): The Ray dataset to shuffle.
    original_data (list): The original data list for verification.

    Returns:
    float: Time taken to shuffle in seconds.
    """
    start_time = time.time()
    shuffled_dataset = dataset.random_shuffle().materialize()
    elapsed_time = time.time() - start_time
    shuffled_data = list(shuffled_dataset.take_all())
    verify_shuffling(original_data, shuffled_data, "Ray random_shuffle")
    return elapsed_time

def custom_batch_shuffle_parallel(dataset, original_data, batch_size):
    """
    Shuffle the dataset using custom batch-wise shuffling in parallel and verify.

    Args:
    dataset (ray.data.Dataset): The Ray dataset to shuffle.
    original_data (list): The original data list for verification.
    batch_size (int): The size of each batch for shuffling.

    Returns:
    float: Time taken to shuffle in seconds.
    """
    shuffled_data = []
    start_time = time.time()
    shuffle_tasks = []

    for batch in dataset.iter_batches(batch_size=batch_size, batch_format='pandas'):
        batch_list = batch.to_dict(orient='records')
        shuffle_tasks.append(shuffle_batch_parallel.remote(batch_list))

    shuffled_batches = ray.get(shuffle_tasks)

    for shuffled_batch in shuffled_batches:
        shuffled_data.extend(shuffled_batch)
    
    elapsed_time = time.time() - start_time
    verify_shuffling(original_data, shuffled_data, "custom batch-wise (parallel)")
    return elapsed_time

def debug(*args):
    """
    Print debug information.

    Args:
    *args: Variable length argument list to print.
    """
    print(*args)

def plot_results(results_df):
    """
    Plot the benchmark results.

    Args:
    results_df (pd.DataFrame): DataFrame containing the benchmark results.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Dataset Size'], results_df['Custom Batch-wise Shuffle Time (s)'], label='Custom Batch-wise Shuffle', marker='o')
    plt.plot(results_df['Dataset Size'], results_df['Ray random_shuffle Time (s)'], label='Ray random_shuffle', marker='o')
    plt.xlabel('Dataset Size')
    plt.ylabel('Shuffle Time (seconds)')
    plt.title('Benchmarking Shuffle Algorithms')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def benchmark():
    """
    Benchmark the shuffling algorithms for varying dataset sizes.
    """
    ray.init(ignore_reinit_error=True)

    sizes = [100, 1000, 10000, 100000, 1000000, 10000000]  # Different dataset sizes for benchmarking
    batch_size = 1000  # Fixed batch size

    results = []

    for size in sizes:
        print(f"Benchmarking for dataset size: {size}")

        # Create dataset
        dataset = create_small_dataset(size)
        original_data = list(dataset.take_all())

        # Custom batch-wise shuffling (parallel)
        custom_time = custom_batch_shuffle_parallel(dataset, original_data, batch_size)

        # Recreate dataset to ensure same data
        dataset = create_small_dataset(size)

        # Ray's random_shuffle
        ray_time = ray_random_shuffle(dataset, original_data)

        results.append({
            'Dataset Size': size,
            'Custom Batch-wise Shuffle Time (s)': custom_time,
            'Ray random_shuffle Time (s)': ray_time
        })

    # Create a DataFrame to display results
    results_df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(results_df)

    # Plot the results
    plot_results(results_df)

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    benchmark()

