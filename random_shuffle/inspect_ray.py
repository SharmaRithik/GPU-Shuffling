import subprocess

# Step 1: Run ray_shuffle.py to generate output.txt
subprocess.run(["python3", "ray_shuffle.py"])
print("Step 1: ray_shuffle.py executed successfully.")

# Step 2: Run clean_output.py to generate cleaned_output.txt
subprocess.run(["python3", "clear_output.py"])
print("Step 2: clear_output.py executed successfully.")

# Step 3: Run shuffle_percentage.py to calculate the shuffle percentage
subprocess.run(["python3", "shuffle_percentage.py"])
print("Step 3: shuffle_percentage.py executed successfully.")

# Step 4: Run verify_elements.py to verify elements
subprocess.run(["python3", "verify_elements.py"])
print("Step 4: verify_elements.py executed successfully.")

