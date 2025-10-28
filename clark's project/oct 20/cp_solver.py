#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constraint Programming Solver

This module implements optimal target assignment using Google OR-Tools CP-SAT solver.
It ensures collision avoidance and maximizes total priority.
"""

from ortools.sat.python import cp_model
import numpy as np
from typing import List, Tuple, Optional
from kaiju import RobotGridNominal
from kaiju.cKaiju import BossFiber, ApogeeFiber


def cp_assignment(grid: RobotGridNominal,
                 x: np.ndarray,
                 y: np.ndarray,
                 priorities: np.ndarray,
                 fiber_types: np.ndarray,
                 collision_buffer: float = 50.0,
                 time_limit: float = 300.0,
                 num_workers: int = 0,
                 verbose: bool = True) -> Tuple[int, List[Tuple[int, int]], float]:
    """
    Constraint Programming assignment algorithm using OR-Tools CP-SAT solver.
    
    Args:
        grid: RobotGridNominal instance with loaded robots
        x, y: Target coordinates
        priorities: Target priority levels (1-5)
        fiber_types: Target fiber types ('BOSS' or 'APOGEE')
        collision_buffer: Minimum distance between assigned robots (mm)
        time_limit: Maximum solver time in seconds
        num_workers: Number of worker threads for CP solver (0 = use all cores)
        verbose: Whether to print solver progress
        
    Returns:
        total_priority: Total priority of assigned targets
        assignments: List of (robot_id, target_id) tuples
        solver_time: Time taken by the solver
    """
    if verbose:
        print(f"Setting up CP assignment problem with {len(x)} targets...")
    
    # Create CP model
    model = cp_model.CpModel()
    
    # Get robot lists
    boss_robots = [robot for robot in grid.robotDict.values() if robot.hasBoss and not robot.isOffline]
    apogee_robots = [robot for robot in grid.robotDict.values() if robot.hasApogee and not robot.isOffline]
    all_robots = [robot for robot in grid.robotDict.values() if not robot.isOffline]
    
    # Create robot ID to index mapping
    robot_id_to_idx = {robot.id: i for i, robot in enumerate(all_robots)}
    robot_idx_to_id = {i: robot.id for i, robot in enumerate(all_robots)}
    
    # Separate targets by fiber type
    boss_targets = []
    apogee_targets = []
    
    for i in range(len(x)):
        target_id = i
        priority = priorities[i]
        fiber_type = fiber_types[i]
        
        if fiber_type == 'BOSS':
            boss_targets.append((target_id, x[i], y[i], priority))
        elif fiber_type == 'APOGEE':
            apogee_targets.append((target_id, x[i], y[i], priority))
    
    if verbose:
        print(f"  BOSS targets: {len(boss_targets)}")
        print(f"  APOGEE targets: {len(apogee_targets)}")
        print(f"  Available robots: {len(all_robots)}")
    
    # Create assignment variables
    # assignment[r][t] = 1 if robot r is assigned to target t, 0 otherwise
    assignments = {}
    target_priorities = {}
    
    # Add BOSS assignments
    for target_id, target_x, target_y, priority in boss_targets:
        target_priorities[target_id] = priority
        assignments[target_id] = {}
        
        for robot in boss_robots:
            robot_idx = robot_id_to_idx[robot.id]
            assignments[target_id][robot_idx] = model.NewBoolVar(f'assign_boss_r{robot.id}_t{target_id}')
    
    # Add APOGEE assignments
    for target_id, target_x, target_y, priority in apogee_targets:
        target_priorities[target_id] = priority
        assignments[target_id] = {}
        
        for robot in apogee_robots:
            robot_idx = robot_id_to_idx[robot.id]
            assignments[target_id][robot_idx] = model.NewBoolVar(f'assign_apogee_r{robot.id}_t{target_id}')
    
    if verbose:
        total_vars = sum(len(assignments[t]) for t in assignments)
        print(f"  Created {total_vars} assignment variables")
    
    # Constraint 1: Each robot can be assigned to at most one target
    if verbose:
        print("  Adding robot assignment constraints...")
    
    for robot_idx in range(len(all_robots)):
        robot_assignments = []
        for target_id in assignments:
            if robot_idx in assignments[target_id]:
                robot_assignments.append(assignments[target_id][robot_idx])
        
        if robot_assignments:
            model.Add(sum(robot_assignments) <= 1)
    
    # Constraint 2: Each target can be observed by at most one robot
    if verbose:
        print("  Adding target observation constraints...")
    
    for target_id in assignments:
        target_assignments = list(assignments[target_id].values())
        if target_assignments:
            model.Add(sum(target_assignments) <= 1)
    
    # Constraint 3: Collision avoidance - robots assigned to targets must be far enough apart
    if verbose:
        print("  Adding collision avoidance constraints...")
    
    collision_constraints = 0
    
    for target1_id in assignments:
        for target2_id in assignments:
            if target1_id >= target2_id:  # Avoid duplicates
                continue
                
            # Get target positions
            if target1_id < len(boss_targets):
                _, x1, y1, _ = boss_targets[target1_id]
            else:
                idx = target1_id - len(boss_targets)
                _, x1, y1, _ = apogee_targets[idx]
                
            if target2_id < len(boss_targets):
                _, x2, y2, _ = boss_targets[target2_id]
            else:
                idx = target2_id - len(boss_targets)
                _, x2, y2, _ = apogee_targets[idx]
            
            # Calculate distance between targets
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            if distance < collision_buffer:
                # If targets are too close, at most one can be assigned
                target1_vars = list(assignments[target1_id].values())
                target2_vars = list(assignments[target2_id].values())
                
                if target1_vars and target2_vars:
                    model.Add(sum(target1_vars) + sum(target2_vars) <= 1)
                    collision_constraints += 1
    
    if verbose:
        print(f"  Added {collision_constraints} collision avoidance constraints")
    
    # Objective: Maximize total priority
    if verbose:
        print("  Setting up objective function...")
    
    objective_terms = []
    
    for target_id in assignments:
        priority = target_priorities[target_id]
        for robot_idx in assignments[target_id]:
            objective_terms.append(assignments[target_id][robot_idx] * priority)
    
    model.Maximize(sum(objective_terms))
    
    # Solve the model
    if verbose:
        print("  Solving CP model...")
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    
    status = solver.Solve(model)
    
    # Process results
    total_priority = 0
    assignments_list = []
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if verbose:
            print(f"  Solution found! Status: {solver.StatusName(status)}")
        
        for target_id in assignments:
            for robot_idx in assignments[target_id]:
                if solver.Value(assignments[target_id][robot_idx]) == 1:
                    robot_id = robot_idx_to_id[robot_idx]
                    priority = target_priorities[target_id]
                    total_priority += priority
                    assignments_list.append((robot_id, target_id))
    else:
        if verbose:
            print(f"  No solution found. Status: {solver.StatusName(status)}")
    
    solver_time = solver.WallTime()
    
    if verbose:
        print(f"  Solver time: {solver_time:.2f} seconds")
        print(f"  Total priority: {total_priority}")
        print(f"  Total assignments: {len(assignments_list)}")
    
    return total_priority, assignments_list, solver_time


def cp_assignment_simple(grid: RobotGridNominal,
                        x: np.ndarray,
                        y: np.ndarray,
                        priorities: np.ndarray,
                        fiber_types: np.ndarray,
                        time_limit: float = 300.0,
                        num_workers: int = 0,
                        verbose: bool = True) -> Tuple[int, List[Tuple[int, int]], float]:
    """
    Simplified CP assignment without collision constraints (faster).
    
    Args:
        grid: RobotGridNominal instance with loaded robots
        x, y: Target coordinates
        priorities: Target priority levels (1-5)
        fiber_types: Target fiber types ('BOSS' or 'APOGEE')
        time_limit: Maximum solver time in seconds
        num_workers: Number of worker threads for CP solver (0 = use all cores)
        verbose: Whether to print solver progress
        
    Returns:
        total_priority: Total priority of assigned targets
        assignments: List of (robot_id, target_id) tuples
        solver_time: Time taken by the solver
    """
    if verbose:
        print(f"Setting up simplified CP assignment problem with {len(x)} targets...")
    
    # Create CP model
    model = cp_model.CpModel()
    
    # Get robot lists
    boss_robots = [robot for robot in grid.robotDict.values() if robot.hasBoss and not robot.isOffline]
    apogee_robots = [robot for robot in grid.robotDict.values() if robot.hasApogee and not robot.isOffline]
    all_robots = [robot for robot in grid.robotDict.values() if not robot.isOffline]
    
    # Create robot ID to index mapping
    robot_id_to_idx = {robot.id: i for i, robot in enumerate(all_robots)}
    robot_idx_to_id = {i: robot.id for i, robot in enumerate(all_robots)}
    
    # Create assignment variables for each target-robot pair
    assignment_vars = {}
    target_priorities = {}
    
    for i in range(len(x)):
        target_id = i
        priority = priorities[i]
        fiber_type = fiber_types[i]
        target_priorities[target_id] = priority
        
        assignment_vars[target_id] = {}
        
        # Find compatible robots
        if fiber_type == 'BOSS':
            compatible_robots = boss_robots
        elif fiber_type == 'APOGEE':
            compatible_robots = apogee_robots
        else:
            continue
        
        for robot in compatible_robots:
            robot_idx = robot_id_to_idx[robot.id]
            assignment_vars[target_id][robot_idx] = model.NewBoolVar(f'assign_r{robot.id}_t{target_id}')
    
    if verbose:
        total_vars = sum(len(assignment_vars[t]) for t in assignment_vars)
        print(f"  Created {total_vars} assignment variables")
    
    # Constraint 1: Each robot can be assigned to at most one target
    if verbose:
        print("  Adding robot assignment constraints...")
    
    for robot_idx in range(len(all_robots)):
        robot_assignments = []
        for target_id in assignment_vars:
            if robot_idx in assignment_vars[target_id]:
                robot_assignments.append(assignment_vars[target_id][robot_idx])
        
        if robot_assignments:
            model.Add(sum(robot_assignments) <= 1)
    
    # Constraint 2: Each target can be observed by at most one robot
    if verbose:
        print("  Adding target observation constraints...")
    
    for target_id in assignment_vars:
        target_assignments = list(assignment_vars[target_id].values())
        if target_assignments:
            model.Add(sum(target_assignments) <= 1)
    
    # Objective: Maximize total priority
    if verbose:
        print("  Setting up objective function...")
    
    objective_terms = []
    
    for target_id in assignment_vars:
        priority = target_priorities[target_id]
        for robot_idx in assignment_vars[target_id]:
            objective_terms.append(assignment_vars[target_id][robot_idx] * priority)
    
    model.Maximize(sum(objective_terms))
    
    # Solve the model
    if verbose:
        print("  Solving simplified CP model...")
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    
    status = solver.Solve(model)
    
    # Process results
    total_priority = 0
    assignments_list = []
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if verbose:
            print(f"  Solution found! Status: {solver.StatusName(status)}")
        
        for target_id in assignment_vars:
            for robot_idx in assignment_vars[target_id]:
                if solver.Value(assignment_vars[target_id][robot_idx]) == 1:
                    robot_id = robot_idx_to_id[robot_idx]
                    priority = target_priorities[target_id]
                    total_priority += priority
                    assignments_list.append((robot_id, target_id))
    else:
        if verbose:
            print(f"  No solution found. Status: {solver.StatusName(status)}")
    
    solver_time = solver.WallTime()
    
    if verbose:
        print(f"  Solver time: {solver_time:.2f} seconds")
        print(f"  Total priority: {total_priority}")
        print(f"  Total assignments: {len(assignments_list)}")
    
    return total_priority, assignments_list, solver_time


if __name__ == "__main__":
    # Example usage
    from config_robots import load_apo_grid
    from targets import generate_targets
    
    # Load grid and generate targets
    grid = load_apo_grid()
    x, y, priorities, fiber_types = generate_targets(
        lam_per_area=0.001,
        xlim=(-300, 300),
        ylim=(-300, 300),
        enable_fiber_types=True
    )
    
    # Run CP assignment
    total_priority, assignments, solver_time = cp_assignment_simple(
        grid, x, y, priorities, fiber_types, time_limit=60.0, num_workers=0
    )
    print(f"\nFinal results: {len(assignments)} assignments, total priority: {total_priority}, time: {solver_time:.2f}s")
