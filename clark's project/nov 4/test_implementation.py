#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the Greedy vs CP comparison implementation.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_robots import load_apo_grid
from targets import sample_ppp
from greedy import GreedyStrategy
from cp import ORToolsStrategy
from target_utils import add_targets_to_robotgrid

def test_single_experiment():
    """Test a single experiment with both algorithms."""
    print("=== Testing Single Experiment ===")
    
    # Set up parameters
    lambda_val = 0.001
    xlim = (-300, 300)
    ylim = (-300, 300)
    seed = 42
    
    # Generate targets
    rng = np.random.default_rng(seed)
    x, y = sample_ppp(lambda_val, xlim, ylim, rng)
    print(f"Generated {len(x)} targets")
    
    # Load robot grid
    robotgrid = load_apo_grid(seed=seed)
    print(f"Loaded {robotgrid.nRobots} robots")
    
    # Add targets to robotgrid
    add_targets_to_robotgrid(robotgrid, x, y)
    print(f"Added {len(robotgrid.targetDict)} targets to robotgrid")
    
    # Test Greedy Algorithm
    print("\n--- Testing Greedy Algorithm ---")
    greedy_strategy = GreedyStrategy(robotgrid=robotgrid, targets=(x, y), seed=seed)
    greedy_results = greedy_strategy.run_optimizer()
    greedy_summary = greedy_strategy.get_assignment_summary()
    print(f"Greedy - Assigned: {greedy_summary['num_assigned']}, Time: {greedy_summary['elapsed_time']:.3f}s")
    
    # Clear assignments for CP test
    for robot_id in robotgrid.robotDict:
        robotgrid.homeRobot(robot_id)
    
    # Test CP Algorithm
    print("\n--- Testing CP Algorithm ---")
    cp_strategy = ORToolsStrategy(robotgrid=robotgrid, targets=(x, y), seed=seed, time_limit=10.0)
    cp_results = cp_strategy.run_optimizer()
    cp_summary = cp_strategy.get_assignment_summary()
    print(f"CP - Assigned: {cp_summary['num_assigned']}, Time: {cp_summary['elapsed_time']:.3f}s")
    
    print("\n=== Test Completed Successfully ===")
    return True

if __name__ == "__main__":
    try:
        test_single_experiment()
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
