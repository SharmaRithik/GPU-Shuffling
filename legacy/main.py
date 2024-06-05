import sys
import subprocess

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <algorithm>")
        return

    algorithm = sys.argv[1]

    if algorithm.startswith("radix"):
        run_radix_commands()
    else:
        print(f"Algorithm '{algorithm}' is not supported.")

def run_radix_commands():
    try:
        subprocess.run(["python3", "radix_sort/radix_sort_shuffle.py"])
        subprocess.run(["python3", "radix_sort/radix_sort_plain.py"])
    except FileNotFoundError:
        print("Error: One or both of the radix sort scripts not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

