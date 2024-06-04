import numpy as np
import os

def read_elements_from_file(filename):
    """
    Read elements from a file and convert them into a numpy array.

    Args:
        filename (str): The path to the file containing the elements.

    Returns:
        numpy.ndarray: The array containing the elements read from the file.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    abs_filename = os.path.join(script_dir, filename)
    
    with open(abs_filename, 'r') as file:
        data = [int(line.strip()) for line in file]
    return np.array(data, dtype=np.int32)

def write_elements_to_file(filename, data):
    """
    Write elements to a file.

    Args:
        filename (str): The path to the file where elements will be written.
        data (numpy.ndarray): The array containing the elements to be written.

    Returns:
        None
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    abs_filename = os.path.join(script_dir, filename)
    
    with open(abs_filename, 'w') as file:
        for element in data:
            file.write(f"{int(element)}\n")

def radix_sort_numpy(data):
    """
    Sort an array using NumPy's stable sort.

    Args:
        data (numpy.ndarray): The array to be sorted.

    Returns:
        numpy.ndarray: The sorted array.
    """
    return np.sort(data, kind='stable')

if __name__ == "__main__":
    input_file = 'data/element_100.txt'
    output_file = 'result/sorted_elements.txt'

    # Read the input data
    data = read_elements_from_file(input_file)

    # Sort the data using NumPy's stable sort
    sorted_data = radix_sort_numpy(data)

    # Print the sorted data
    print("Sorted Data:")
    print(sorted_data)

    # Optionally, write the sorted data to a file
    write_elements_to_file(output_file, sorted_data)

