from generate_demo import calculate_optimal_demo
from optimised_demo import OptimisedDemo
import json
import numpy as np

starting_conditions = { "x": 0.0, "y": 1.0 }
ending_conditions = { "x": 0.0, "y": 3.0 }
duration = 2.5

trajectoy = calculate_optimal_demo(starting_conditions, ending_conditions, duration)
metadata = {"starting_position": starting_conditions, "duration": duration}

optimised_demo = OptimisedDemo(metadata=metadata, trajectory=trajectoy)
optimised_demo.save_to_file("data/demonstration_1.json")