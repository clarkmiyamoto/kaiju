#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collision Matrix Module

This module builds collision matrices for constraint programming,
identifying which robots can potentially collide when assigned to targets.
"""

import numpy as np
from typing import Dict, List, Tuple
from kaiju import RobotGridNominal


def calculate_robot_reach(grid: RobotGridNominal) -> Dict[int, List[int]]:
    """
    Calculate which targets each robot can reach.
    
    Args:
        grid: RobotGridNominal instance with loaded robots
        
    Returns:
        Dictionary mapping robot_id -> list of reachable target_ids
    """
    print("Calculating robot reach capabilities...")
    
    robot_reach = {}
    for robot_id, robot in grid.robotDict.items():
        if robot.isOffline:
            robot_reach[robot_id] = []
            continue
            
        # Get valid targets for this robot
        valid_targets = robot.validTargetIDs
        robot_reach[robot_id] = list(valid_targets)
    
    return robot_reach


def build_collision_matrix(grid: RobotGridNominal, 
                          targets: Tuple[np.ndarray, np.ndarray], 
                          collision_buffer: float = 50.0) -> np.ndarray:
    """
    Build collision matrix based on robot reach capabilities and target proximity.
    
    Args:
        grid: RobotGridNominal instance with loaded robots
        targets: Tuple of (x_coords, y_coords) for targets
        collision_buffer: Minimum distance between assigned robots (mm)
        
    Returns:
        Collision matrix where entry (i,j) = 1 if robots i and j can collide
    """
    print(f"Building collision matrix with buffer={collision_buffer}mm...")
    
    x_targets, y_targets = targets
    robot_reach = calculate_robot_reach(grid)
    
    # Get active robots (not offline)
    active_robots = [robot_id for robot_id, reach in robot_reach.items() if reach]
    n_robots = len(active_robots)
    
    # Create mapping from robot_id to matrix index
    robot_id_to_index = {robot_id: i for i, robot_id in enumerate(active_robots)}
    
    # Initialize collision matrix
    collision_matrix = np.zeros((n_robots, n_robots), dtype=int)
    
    # Check for potential collisions between robot pairs
    collision_pairs = 0
    for i, robot_id_i in enumerate(active_robots):
        for j, robot_id_j in enumerate(active_robots):
            if i >= j:  # Only check upper triangle, matrix is symmetric
                continue
                
            # Check if robots can reach any common targets
            targets_i = set(robot_reach[robot_id_i])
            targets_j = set(robot_reach[robot_id_j])
            common_targets = targets_i.intersection(targets_j)
            
            if common_targets:
                # Check if any common targets are close enough to cause collision
                for target_id in common_targets:
                    if target_id < len(x_targets):
                        target_x, target_y = x_targets[target_id], y_targets[target_id]
                        
                        # Check distance to other targets that both robots can reach
                        for other_target_id in common_targets:
                            if other_target_id != target_id and other_target_id < len(x_targets):
                                other_x, other_y = x_targets[other_target_id], y_targets[other_target_id]
                                distance = np.sqrt((target_x - other_x)**2 + (target_y - other_y)**2)
                                
                                if distance < collision_buffer:
                                    collision_matrix[i, j] = 1
                                    collision_matrix[j, i] = 1  # Symmetric matrix
                                    collision_pairs += 1
                                    break
                        if collision_matrix[i, j] == 1:
                            break
    
    print(f"Found {collision_pairs} potential collision pairs")
    return collision_matrix


def build_simple_collision_matrix(grid: RobotGridNominal, 
                                 targets: Tuple[np.ndarray, np.ndarray], 
                                 collision_buffer: float = 50.0) -> np.ndarray:
    """
    Build a simpler collision matrix based only on target proximity.
    
    Args:
        grid: RobotGridNominal instance with loaded robots
        targets: Tuple of (x_coords, y_coords) for targets
        collision_buffer: Minimum distance between targets to avoid collision
        
    Returns:
        Collision matrix where entry (i,j) = 1 if targets i and j are too close
    """
    print(f"Building simple collision matrix with buffer={collision_buffer}mm...")
    
    x_targets, y_targets = targets
    n_targets = len(x_targets)
    
    # Initialize collision matrix
    collision_matrix = np.zeros((n_targets, n_targets), dtype=int)
    
    # Check distance between all target pairs
    collision_pairs = 0
    for i in range(n_targets):
        for j in range(i + 1, n_targets):
            distance = np.sqrt((x_targets[i] - x_targets[j])**2 + (y_targets[i] - y_targets[j])**2)
            if distance < collision_buffer:
                collision_matrix[i, j] = 1
                collision_matrix[j, i] = 1  # Symmetric matrix
                collision_pairs += 1
    
    print(f"Found {collision_pairs} target pairs within collision buffer")
    return collision_matrix


def analyze_collision_matrix(collision_matrix: np.ndarray, robot_ids: List[int] = None) -> Dict:
    """
    Analyze the collision matrix and return statistics.
    
    Args:
        collision_matrix: Collision matrix
        robot_ids: List of robot IDs (for labeling)
        
    Returns:
        Dictionary with collision statistics
    """
    n_robots = collision_matrix.shape[0]
    total_pairs = n_robots * (n_robots - 1) // 2
    collision_pairs = np.sum(collision_matrix) // 2  # Divide by 2 for symmetric matrix
    
    # Find robots with most collision potential
    collision_counts = np.sum(collision_matrix, axis=1)
    max_collisions = np.max(collision_counts)
    
    if robot_ids is not None:
        most_collision_robots = [robot_ids[i] for i in np.where(collision_counts == max_collisions)[0]]
    else:
        most_collision_robots = list(np.where(collision_counts == max_collisions)[0])
    
    stats = {
        'total_robots': n_robots,
        'total_pairs': total_pairs,
        'collision_pairs': collision_pairs,
        'collision_fraction': collision_pairs / total_pairs if total_pairs > 0 else 0,
        'max_collisions_per_robot': max_collisions,
        'robots_with_most_collisions': most_collision_robots,
        'collision_matrix': collision_matrix
    }
    
    return stats


def print_collision_summary(collision_matrix: np.ndarray, robot_ids: List[int] = None):
    """
    Print a summary of the collision matrix.
    
    Args:
        collision_matrix: Collision matrix
        robot_ids: List of robot IDs (for labeling)
    """
    stats = analyze_collision_matrix(collision_matrix, robot_ids)
    
    print(f"\n=== Collision Matrix Summary ===")
    print(f"Total robots: {stats['total_robots']}")
    print(f"Total robot pairs: {stats['total_pairs']}")
    print(f"Pairs with potential collisions: {stats['collision_pairs']}")
    print(f"Collision fraction: {stats['collision_fraction']:.1%}")
    print(f"Maximum collisions per robot: {stats['max_collisions_per_robot']}")
    print(f"Robots with most collision potential: {stats['robots_with_most_collisions']}")


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
    
    # Build collision matrix
    collision_matrix = build_simple_collision_matrix(grid, (x, y), collision_buffer=50.0)
    print_collision_summary(collision_matrix)



