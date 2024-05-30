# GPU-Shuffling

(ray_env) user@machine:~/random_shuffle$ python3 inspect_ray.py 
2024-05-30 16:31:59,841 INFO worker.py:1564 -- Connecting to existing Ray cluster at address: [IP address]:[Port]
2024-05-30 16:31:59,845 INFO worker.py:1749 -- Connected to Ray cluster.
2024-05-30 16:32:01,580 INFO streaming_executor.py:108 -- Starting execution of Dataset.
2024-05-30 16:32:01,580 INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> AllToAllOperator[RandomShuffle]

(autoscaler +2s) Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.                                                   
(autoscaler +2s) Warning: The following resource request cannot be scheduled right now: {'CPU': 1.0}.                                                                                         
                                                                                                                                                                                 Step 1: ray_shuffle.py executed successfully.                                                                                                                                    Cleaned data written to cleaned_output.txt                                                                                                                                       
Step 2: clear_output.py executed successfully.                                                                                                                                   
Shuffle Percentage: 100.00%
Step 3: shuffle_percentage.py executed successfully.
All elements from input.txt are present in cleaned_output.txt.
Step 4: verify_elements.py executed successfully.
