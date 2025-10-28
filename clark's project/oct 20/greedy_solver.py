#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Greedy Assignment Solver

This module implements a greedy algorithm for assigning robots to targets.
The algorithm assigns highest priority targets first to the nearest available robot.
"""

import numpy as np
from typing import List, Tuple, Optional
from kaiju import RobotGridNominal
from kaiju.cKaiju import BossFiber, ApogeeFiber


def greedy_assignment(grid: RobotGridNominal, 
                     x: np.ndarray, 
                     y: np.ndarray, 
                     priorities: np.ndarray, 
                     fiber_types: np.ndarray,
                     verbose: bool = True) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Greedy algorithm to assign targets to robots.
    
    Args:
        grid: RobotGridNominal instance with loaded robots
        x, y: Target coordinates
        priorities: Target priority levels (1-5)
        fiber_types: Target fiber types ('BOSS' or 'APOGEE')
        verbose: Whether to print assignment progress
        
    Returns:
        total_priority: Total priority of assigned targets
        assignments: List of (robot_id, target_id) tuples
    """
    if verbose:
        print(f"Running greedy assignment on {len(x)} targets...")
    
    # Convert fiber type strings to kaiju fiber types
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
    
    # Sort targets by priority (highest first)
    boss_targets.sort(key=lambda x: x[3], reverse=True)
    apogee_targets.sort(key=lambda x: x[3], reverse=True)
    
    if verbose:
        print(f"  BOSS targets: {len(boss_targets)}")
        print(f"  APOGEE targets: {len(apogee_targets)}")
    
    total_priority = 0
    assignments = []
    
    # Get available robots for each fiber type
    boss_robots = [robot for robot in grid.robotDict.values() if robot.hasBoss and not robot.isOffline]
    apogee_robots = [robot for robot in grid.robotDict.values() if robot.hasApogee and not robot.isOffline]
    
    # Track which robots are already assigned
    assigned_robots = set()
    
    # First, assign BOSS targets
    if verbose:
        print(f"Assigning {len(boss_targets)} BOSS targets...")
    
    for target_id, target_x, target_y, priority in boss_targets:
        best_robot = None
        best_distance = float('inf')
        
        # Find closest available BOSS robot
        for robot in boss_robots:
            if robot.id not in assigned_robots:
                distance = np.sqrt((robot.xPos - target_x)**2 + (robot.yPos - target_y)**2)
                if distance < best_distance:
                    best_distance = distance
                    best_robot = robot
        
        if best_robot is not None:
            # Add target to grid
            grid.addTarget(target_id, (target_x, target_y, 0), BossFiber, priority)
            
            # Try to assign robot to target
            try:
                grid.assignRobot2Target(best_robot.id, target_id)
                assigned_robots.add(best_robot.id)
                total_priority += priority
                assignments.append((best_robot.id, target_id))
                if verbose:
                    print(f"  Assigned BOSS target {target_id} (priority {priority}) to robot {best_robot.id}")
            except Exception as e:
                if verbose:
                    print(f"  Failed to assign BOSS target {target_id} to robot {best_robot.id}: {e}")
    
    # Then, assign APOGEE targets
    if verbose:
        print(f"Assigning {len(apogee_targets)} APOGEE targets...")
    
    for target_id, target_x, target_y, priority in apogee_targets:
        best_robot = None
        best_distance = float('inf')
        
        # Find closest available APOGEE robot
        for robot in apogee_robots:
            if robot.id not in assigned_robots:
                distance = np.sqrt((robot.xPos - target_x)**2 + (robot.yPos - target_y)**2)
                if distance < best_distance:
                    best_distance = distance
                    best_robot = robot
        
        if best_robot is not None:
            # Add target to grid
            grid.addTarget(target_id, (target_x, target_y, 0), ApogeeFiber, priority)
            
            # Try to assign robot to target
            try:
                grid.assignRobot2Target(best_robot.id, target_id)
                assigned_robots.add(best_robot.id)
                total_priority += priority
                assignments.append((best_robot.id, target_id))
                if verbose:
                    print(f"  Assigned APOGEE target {target_id} (priority {priority}) to robot {best_robot.id}")
            except Exception as e:
                if verbose:
                    print(f"  Failed to assign APOGEE target {target_id} to robot {best_robot.id}: {e}")
    
    if verbose:
        print(f"Greedy assignment complete: {len(assignments)} assignments, total priority: {total_priority}")
    
    return total_priority, assignments


def greedy_assignment_with_collision_check(grid: RobotGridNominal,
                                         x: np.ndarray,
                                         y: np.ndarray,
                                         priorities: np.ndarray,
                                         fiber_types: np.ndarray,
                                         collision_buffer: float = 50.0,
                                         verbose: bool = True) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Greedy assignment with collision checking.
    
    Args:
        grid: RobotGridNominal instance with loaded robots
        x, y: Target coordinates
        priorities: Target priority levels (1-5)
        fiber_types: Target fiber types ('BOSS' or 'APOGEE')
        collision_buffer: Minimum distance between assigned robots (mm)
        verbose: Whether to print assignment progress
        
    Returns:
        total_priority: Total priority of assigned targets
        assignments: List of (robot_id, target_id) tuples
    """
    if verbose:
        print(f"Running greedy assignment with collision checking on {len(x)} targets...")
    
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
    
    # Sort targets by priority (highest first)
    boss_targets.sort(key=lambda x: x[3], reverse=True)
    apogee_targets.sort(key=lambda x: x[3], reverse=True)
    
    total_priority = 0
    assignments = []
    assigned_robots = set()
    assigned_target_positions = []  # Track positions of assigned targets
    
    # Assign BOSS targets
    if verbose:
        print(f"Assigning {len(boss_targets)} BOSS targets...")
    
    boss_robots = [robot for robot in grid.robotDict.values() if robot.hasBoss and not robot.isOffline]
    
    for target_id, target_x, target_y, priority in boss_targets:
        best_robot = None
        best_distance = float('inf')
        
        # Find closest available BOSS robot
        for robot in boss_robots:
            if robot.id not in assigned_robots:
                distance = np.sqrt((robot.xPos - target_x)**2 + (robot.yPos - target_y)**2)
                if distance < best_distance:
                    best_distance = distance
                    best_robot = robot
        
        if best_robot is not None:
            # Check for collisions with already assigned targets
            collision = False
            for assigned_x, assigned_y in assigned_target_positions:
                distance = np.sqrt((target_x - assigned_x)**2 + (target_y - assigned_y)**2)
                if distance < collision_buffer:
                    collision = True
                    break
            
            if not collision:
                # Add target to grid
                grid.addTarget(target_id, (target_x, target_y, 0), BossFiber, priority)
                
                # Try to assign robot to target
                try:
                    grid.assignRobot2Target(best_robot.id, target_id)
                    assigned_robots.add(best_robot.id)
                    assigned_target_positions.append((target_x, target_y))
                    total_priority += priority
                    assignments.append((best_robot.id, target_id))
                    if verbose:
                        print(f"  Assigned BOSS target {target_id} (priority {priority}) to robot {best_robot.id}")
                except Exception as e:
                    if verbose:
                        print(f"  Failed to assign BOSS target {target_id} to robot {best_robot.id}: {e}")
            else:
                if verbose:
                    print(f"  Skipped BOSS target {target_id} due to collision")
    
    # Assign APOGEE targets
    if verbose:
        print(f"Assigning {len(apogee_targets)} APOGEE targets...")
    
    apogee_robots = [robot for robot in grid.robotDict.values() if robot.hasApogee and not robot.isOffline]
    
    for target_id, target_x, target_y, priority in apogee_targets:
        best_robot = None
        best_distance = float('inf')
        
        # Find closest available APOGEE robot
        for robot in apogee_robots:
            if robot.id not in assigned_robots:
                distance = np.sqrt((robot.xPos - target_x)**2 + (robot.yPos - target_y)**2)
                if distance < best_distance:
                    best_distance = distance
                    best_robot = robot
        
        if best_robot is not None:
            # Check for collisions with already assigned targets
            collision = False
            for assigned_x, assigned_y in assigned_target_positions:
                distance = np.sqrt((target_x - assigned_x)**2 + (target_y - assigned_y)**2)
                if distance < collision_buffer:
                    collision = True
                    break
            
            if not collision:
                # Add target to grid
                grid.addTarget(target_id, (target_x, target_y, 0), ApogeeFiber, priority)
                
                # Try to assign robot to target
                try:
                    grid.assignRobot2Target(best_robot.id, target_id)
                    assigned_robots.add(best_robot.id)
                    assigned_target_positions.append((target_x, target_y))
                    total_priority += priority
                    assignments.append((best_robot.id, target_id))
                    if verbose:
                        print(f"  Assigned APOGEE target {target_id} (priority {priority}) to robot {best_robot.id}")
                except Exception as e:
                    if verbose:
                        print(f"  Failed to assign APOGEE target {target_id} to robot {best_robot.id}: {e}")
            else:
                if verbose:
                    print(f"  Skipped APOGEE target {target_id} due to collision")
    
    if verbose:
        print(f"Greedy assignment with collision checking complete: {len(assignments)} assignments, total priority: {total_priority}")
    
    return total_priority, assignments


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
    
    # Run greedy assignment
    total_priority, assignments = greedy_assignment(grid, x, y, priorities, fiber_types)
    print(f"\nFinal results: {len(assignments)} assignments, total priority: {total_priority}")



