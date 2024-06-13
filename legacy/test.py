import ray
import time
import numpy as np

def verify_shuffling(original_data, shuffled_data, method="Shuffle"):
    """
    Verify that the shuffling was performed correctly by comparing sets of original
    and shuffled data.
    
    Parameters:
    original_data (numpy array): Original array before shuffling.
    shuffled_data (numpy array): Array after shuffling.
    method (str): Method used for shuffling, for display purposes.
    
    Returns:
    bool: True if verification passed, False otherwise.
    """
    if isinstance(shuffled_data[0], dict):
        shuffled_data = [v for d in shuffled_data for v in d.values()]

    original_set = set(original_data)
    shuffled_set = set(shuffled_data)
    is_correct = original_set == shuffled_set and len(original_data) == len(shuffled_data)

    print(f"Verification ({method}): {'Passed' if is_correct else 'Failed'}")

    if len(original_data) != len(shuffled_data):
        print(f"Length mismatch: original_data length = {len(original_data)}, shuffled_data length = {len(shuffled_data)}")
        return False

    # Only bother checking shuffle percentage if under certain length
    if len(original_data) < 1000000:
        num_shuffled = 0
        for i in range(len(original_data)):
            if original_data[i] != shuffled_data[i]:
                num_shuffled += 1
        print(f"% of items shuffled: {num_shuffled / len(original_data) * 100}%")

    return is_correct

@ray.remote
def shuffle_partition(data):
    """
    Perform shuffle on a portion of the data.

    Parameters:
    data (list): Portion of the data to be shuffled.

    Returns:
    list: Shuffled portion of the data.
    """
    data_list = list(data)  # Convert to list if needed
    np.random.shuffle(data_list)
    return data_list

def topological_shuffle(dataset, original_data, num_partitions):
    """
    Shuffle an array using topological shuffle with Ray.
    
    Parameters:
    dataset (ray.data.Dataset): Dataset to be shuffled.
    original_data (numpy array): Original array before shuffling.
    num_partitions (int): Number of partitions to split the data into.
    
    Returns:
    float: Time taken to shuffle in seconds.
    numpy array: Shuffled data array.
    """
    start_time = time.perf_counter()
    partitioned_dataset = dataset.repartition(num_partitions)
    partitions = list(partitioned_dataset.iter_batches())
    
    shuffled_partitions = ray.get([shuffle_partition.remote(batch["data"]) for batch in partitions])
    
    # Ensure the shuffled partitions are concatenated correctly and match the original length
    shuffled_data = np.concatenate(shuffled_partitions)
    shuffled_data = shuffled_data[:len(original_data)]  # Adjust the length to match the original data
    
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, shuffled_data

def round_robin_shuffle(dataset, original_data, num_partitions):
    """
    Shuffle an array using round-robin shuffle with Ray.
    
    Parameters:
    dataset (ray.data.Dataset): Dataset to be shuffled.
    original_data (numpy array): Original array before shuffling.
    num_partitions (int): Number of partitions to split the data into.
    
    Returns:
    float: Time taken to shuffle in seconds.
    numpy array: Shuffled data array.
    """
    start_time = time.perf_counter()
    # Create partitions
    partitions = [[] for _ in range(num_partitions)]
    for i, item in enumerate(original_data):
        partitions[i % num_partitions].append(item)
    
    # Shuffle each partition
    shuffled_partitions = ray.get([shuffle_partition.remote(partition) for partition in partitions])
    
    # Concatenate the shuffled partitions
    shuffled_data = np.concatenate(shuffled_partitions)
    
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, shuffled_data

def ray_random_shuffle(dataset, original_data):
    """
    Shuffle an array using Ray's random_shuffle() method.
    
    Parameters:
    dataset (ray.data.Dataset): Dataset to be shuffled.
    original_data (numpy array): Original array before shuffling.
    
    Returns:
    float: Time taken to shuffle in seconds.
    numpy array: Shuffled data array.
    """
    start_time = time.perf_counter()
    shuffled_dataset = dataset.random_shuffle().materialize()
    shuffled_data = list(shuffled_dataset.take_all())
    shuffled_data = [v for d in shuffled_data for v in d.values()]
    shuffled_data = np.array(shuffled_data)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, shuffled_data

def human_readable_size(num_elements, element_size_bytes):
    """
    Convert a size in bytes to a human-readable string in MB or GB.
    
    Parameters:
    num_elements (int): Number of elements in the array.
    element_size_bytes (int): Size of each element in bytes.
    
    Returns:
    str: Human-readable size string.
    """
    size_bytes = num_elements * element_size_bytes
    size_mb = size_bytes / (1024 ** 2)
    size_gb = size_bytes / (1024 ** 3)
    if size_gb >= 1:
        return f"{size_gb:.2f} GB"
    else:
        return f"{size_mb:.2f} MB"

def main():
    """
    Main function to test topological, round-robin, and random shuffling with different data sizes.
    """
    sizes = [100000, 500000, 1000000, 10000000]
    element_size_bytes = np.dtype(np.int64).itemsize  # Size of each element in bytes (assuming int64)
    results = []

    for size in sizes:
        print("\nTesting size:", size)
        original_data = np.arange(size)

        # Topological Shuffle
        ray.init(ignore_reinit_error=True)
        dataset = ray.data.from_numpy(original_data.copy())
        num_partitions = int(np.sqrt(size))  # Number of partitions based on data size
        topo_time, topo_shuffled_data = topological_shuffle(dataset, original_data, num_partitions)
        verify_shuffling(original_data, topo_shuffled_data, "Topological Shuffle")

        # Round-robin Shuffle
        rr_time, rr_shuffled_data = round_robin_shuffle(dataset, original_data, num_partitions)
        verify_shuffling(original_data, rr_shuffled_data, "Round-robin Shuffle")

        # Random Shuffle
        rand_time, rand_shuffled_data = ray_random_shuffle(dataset, original_data)
        verify_shuffling(original_data, rand_shuffled_data, "Random Shuffle")
        ray.shutdown()

        data_size_human_readable = human_readable_size(size, element_size_bytes)
        results.append((size, data_size_human_readable, topo_time, rr_time, rand_time))

    print("\nResults:")
    print("{:<10} {:<15} {:<20} {:<20} {:<20}".format("Size", "Data Size", "Topological Time (s)", "Round-robin Time (s)", "Random Time (s)"))
    for size, data_size_human_readable, topo_time, rr_time, rand_time in results:
        print("{:<10} {:<15} {:<20.6f} {:<20.6f} {:<20.6f}".format(size, data_size_human_readable, topo_time, rr_time, rand_time))

if __name__ == "__main__":
    main()

