import random

def generate_random_int():
  return random.randint(1, 1000)
data = [generate_random_int() for _ in range(100000)]

with open("element_100000.txt", "w") as file:
  for element in data:
    file.write(str(element) + "\n")

print("Data generated and saved to element_100000.txt")
