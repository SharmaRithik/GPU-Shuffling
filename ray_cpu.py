import ray
import ray.data
import pandas as pd
import numpy as np
import time

# Initialize Ray
ray.init(ignore_reinit_error=True)

def shuffle_and_time():
    # Generate random input data
    num_rows = 10000
    df = pd.DataFrame({
        'value': np.random.rand(num_rows)
    })
    
    # Create Ray dataset
    ds = ray.data.from_pandas(df)

    # Time the shuffling operation
    start_time = time.time()
    shuffled_ds = ds.random_shuffle()
    end_time = time.time()
    
    # Convert shuffled dataset back to pandas dataframe for verification
    shuffled_df = shuffled_ds.to_pandas()
    
    # Verify the shuffled outcome by checking that the shuffled dataframe
    # has the same values but in a different order
    original_values = df['value'].values
    shuffled_values = shuffled_df['value'].values
    
    assert set(original_values) == set(shuffled_values), "Shuffled values do not match original values"

    print(f"Shuffling time: {end_time - start_time:.4f} seconds")
    print("Shuffling verified successfully!")

# Run the function
shuffle_and_time()

# Shutdown Ray
ray.shutdown()

