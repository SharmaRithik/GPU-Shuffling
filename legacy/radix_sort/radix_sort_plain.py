import os
import ray
import time

@ray.remote
def radix_sort_partition(data, digit):
    """
    Partitions data into buckets based on the specified digit position.
    
    Args:
        data: List of elements to be sorted.
        digit: The digit position to partition on (0-indexed, least significant digit).
    
    Returns:
        A list of lists, where each sub-list represents elements belonging to the same bucket value.
    """
    buckets = [[] for _ in range(10)]
    for element in data:
        bucket_index = (element // (10**digit)) % 10
        buckets[bucket_index].append(element)
    return buckets

@ray.remote
def radix_sort_recursive(data, digit, max_digits):
    """
    Sorts a list of data using radix sort recursively.
    
    Args:
        data: List of elements to be sorted.
        digit: The current digit position.
        max_digits: The maximum number of digits in the data.
    
    Returns:
        A sorted list of elements.
    """
    if digit >= max_digits or len(data) <= 1:
        return data

    buckets = ray.get(radix_sort_partition.remote(data, digit))
    sorted_buckets = [ray.get(radix_sort_recursive.remote(bucket, digit + 1, max_digits)) for bucket in buckets]
    return [element for bucket in sorted_buckets for element in bucket]

def read_elements_from_file(filename):
    """
    Read elements from a file and convert them into a list.
    
    Args:
        filename (str): The path to the file containing the elements.
    
    Returns:
        list: The list containing the elements read from the file.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    abs_filename = os.path.join(script_dir, filename)
    
    with open(abs_filename, 'r') as file:
        data = [int(line.strip()) for line in file]
    return data

def write_elements_to_file(filename, data):
    """
    Write elements to a file.
    
    Args:
        filename (str): The path to the file where elements will be written.
        data (list): The list containing the elements to be written.
    
    Returns:
        None
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    abs_filename = os.path.join(script_dir, filename)
    with open(abs_filename, 'w') as file:
        for element in data:
            file.write(f"{int(element)}\n")

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
        data = read_elements_from_file(abs_filename)
        if not data:
            raise ValueError("No data found in the file.")

        max_digits = len(str(max(data)))
        sorted_data = ray.get(radix_sort_recursive.remote(data, 0, max_digits))

        end_time = time.time()
        print(f"Total Execution Time for Radix Sort: {end_time - start_time:.4f} seconds")

        # Print the sorted data
        print("Sorted Data:")
        print(sorted_data)

        # Optionally, write the sorted data to a file
        write_elements_to_file("result/sorted_elements.txt", sorted_data)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except ValueError as ve:
        print(f"Error: {ve}")

if __name__ == "__main__":
    ray.init()
    radix_sort_plain("element_100.txt")
    ray.shutdown()

