import random

def fisher_yates_shuffle(arr):
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]

def merge(arr, start, mid, end):
    left = arr[start:mid]
    right = arr[mid:end]
    i = j = 0
    k = start

    while i < len(left) and j < len(right):
        if random.randint(0, 1) == 0:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1

    for i in range(start, end):
        swap_index = random.randint(start, i)
        arr[i], arr[swap_index] = arr[swap_index], arr[i]

def merge_shuffle(arr, k):
    n = len(arr)
    if n <= k:
        fisher_yates_shuffle(arr)
        return

    block_size = (n + 2 * k - 1) // (2 * k)  # Ensure at least 2k blocks
    blocks = [arr[i:i + block_size] for i in range(0, n, block_size)]

    for block in blocks:
        fisher_yates_shuffle(block)

    while len(blocks) > 1:
        new_blocks = []
        for i in range(0, len(blocks) - 1, 2):
            new_block = blocks[i] + blocks[i + 1]
            merge(new_block, 0, len(blocks[i]), len(new_block))
            new_blocks.append(new_block)
        if len(blocks) % 2 == 1:
            new_blocks.append(blocks[-1])
        blocks = new_blocks

    for i in range(len(arr)):
        arr[i] = blocks[0][i]

def read_input(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

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
    input_filename = "input.txt"
    output_filename = "output.txt"
    
    elements = read_input(input_filename)
    original_elements = elements.copy()
    
    merge_shuffle(elements, k=32)
    
    if check_elements(original_elements, elements):
        write_output(output_filename, elements)
        shuffle_percentage = calculate_shuffle_percentage(original_elements, elements)
        print(f"Shuffle Percentage: {shuffle_percentage:.2f}%")
    else:
        print("Error: Elements missing after shuffling!")

