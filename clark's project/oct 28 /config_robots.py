#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robot Configuration Module

This module handles loading and configuring telescope robot grids.
Currently supports Apache Point Observatory (APO) configuration.
"""

from kaiju import RobotGridNominal
from typing import Dict, Any


def load_apo_grid(stepSize: float = 0.1, seed: int = 0) -> RobotGridNominal:
    """
    Load the Apache Point Observatory (APO) telescope robot configuration.
    
    Args:
        stepSize: Step size for robot path planning (degrees)
        seed: Random seed for reproducibility
        
    Returns:
        RobotGridNominal instance with loaded APO robots
    """
    print(f"Loading APO telescope configuration (stepSize={stepSize}, seed={seed})...")
    grid = RobotGridNominal(stepSize=stepSize, seed=seed)
    print(f"Loaded {grid.nRobots} robots")
    return grid


def get_robot_stats(grid: RobotGridNominal) -> Dict[str, Any]:
    """
    Get statistics about the robot configuration.
    
    Args:
        grid: RobotGridNominal instance with loaded robots
        
    Returns:
        Dictionary containing robot statistics
    """
    stats = {
        'total_robots': grid.nRobots,
        'n_fiducials': len(grid.fiducialDict),
        'apogee_robots': 0,
        'boss_only_robots': 0,
        'dual_fiber_robots': 0,
        'offline_robots': 0
    }
    
    for robot in grid.robotDict.values():
        if robot.isOffline:
            stats['offline_robots'] += 1
        elif robot.hasApogee and robot.hasBoss:
            stats['dual_fiber_robots'] += 1
        elif robot.hasBoss and not robot.hasApogee:
            stats['boss_only_robots'] += 1
        elif robot.hasApogee and not robot.hasBoss:
            stats['apogee_robots'] += 1
    
    return stats


def print_robot_summary(grid: RobotGridNominal, show_positions: bool = False, max_positions: int = 5):
    """
    Print a summary of the robot configuration.
    
    Args:
        grid: RobotGridNominal instance with loaded robots
        show_positions: Whether to show sample robot positions
        max_positions: Maximum number of positions to show
    """
    stats = get_robot_stats(grid)
    
    print(f"\n=== Robot Configuration Summary ===")
    print(f"Total robots: {stats['total_robots']}")
    print(f"Fiducials: {stats['n_fiducials']}")
    print(f"Offline robots: {stats['offline_robots']}")
    print(f"Robots with both fibers: {stats['dual_fiber_robots']}")
    print(f"BOSS-only robots: {stats['boss_only_robots']}")
    print(f"APOGEE-only robots: {stats['apogee_robots']}")
    
    if show_positions:
        print(f"\n=== Sample Robot Positions ===")
        count = 0
        for robot_id, robot in grid.robotDict.items():
            if count < max_positions:
                print(f"Robot {robot_id}: position ({robot.xPos:.1f}, {robot.yPos:.1f})")
                count += 1


def get_robots_by_fiber_type(grid: RobotGridNominal) -> Dict[str, list]:
    """
    Get robots organized by their fiber capabilities.
    
    Args:
        grid: RobotGridNominal instance with loaded robots
        
    Returns:
        Dictionary with 'boss', 'apogee', 'dual' keys containing robot lists
    """
    robots = {
        'boss': [],
        'apogee': [],
        'dual': []
    }
    
    for robot in grid.robotDict.values():
        if robot.isOffline:
            continue
            
        if robot.hasBoss and robot.hasApogee:
            robots['dual'].append(robot)
        elif robot.hasBoss:
            robots['boss'].append(robot)
        elif robot.hasApogee:
            robots['apogee'].append(robot)
    
    return robots


if __name__ == "__main__":
    # Example usage
    grid = load_apo_grid()
    print_robot_summary(grid, show_positions=True)
    
    robots_by_type = get_robots_by_fiber_type(grid)
    print(f"\nRobots by fiber type:")
    for fiber_type, robot_list in robots_by_type.items():
        print(f"  {fiber_type}: {len(robot_list)} robots")



