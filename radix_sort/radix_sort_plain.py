import os
import ray
import time

@ray.remote
def radix_sort_partition(data, digit, max_digits):
    """
    Partitions data into buckets based on the specified digit position.

    Args:
        data: List of elements to be sorted.
        digit: The digit position to partition on (0-indexed, least significant digit).
        max_digits: The maximum number of digits in the data.

    Returns:
        A list of lists, where each sub-list represents elements belonging to the same bucket value.
    """
    buckets = [[] for _ in range(10)]
    for element in data:
        bucket_index = (element // (10**digit)) % 10
        buckets[bucket_index].append(element)
    return buckets

@ray.remote
def radix_sort_serial(data, max_digits):
    """
    Sorts a list of data using radix sort recursively.

    Args:
        data: List of elements to be sorted.
        max_digits: The maximum number of digits in the data.

    Returns:
        A sorted list of elements.
    """
    if not data or max_digits == 0:
        return data
    buckets = ray.get(radix_sort_partition.remote(data, 0, max_digits))
    sorted_buckets = [ray.get(radix_sort_serial.remote(bucket, max_digits - 1)) for bucket in buckets]
    return [element for bucket in sorted_buckets for element in bucket]

def radix_sort_plain(filename):
    """
    Sorts data from a file using radix sort with Ray, measuring and printing execution time.

    Args:
        filename: The path to the file containing the data.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    abs_filename = os.path.join(script_dir, "data", filename)
    start_time = time.time()
    try:
        with open(abs_filename, "r") as file:
            data = [int(line.strip()) for line in file]
        if not data:
            raise ValueError("No data found in the file.")
        max_digits = len(str(max(data)))
        sorted_data = ray.get(radix_sort_serial.remote(data, max_digits))
        end_time = time.time()
        print(f"Total Execution Time for Radix Sort: {end_time - start_time:.4f} seconds")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except ValueError as ve:
        print(f"Error: {ve}")

if __name__ == "__main__":
    ray.init()
    radix_sort_plain("element_100000.txt")
    ray.shutdown()
