#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Target Generation Module

This module generates astronomical targets using Poisson Point Process
for realistic target density simulation.
"""

from typing import Tuple, Optional
import numpy as np
import kaiju


def sample_ppp(lam_per_area: float, 
               xlim: Tuple[float, float], 
               ylim: Tuple[float, float], 
               rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points from a Poisson Point Process.
    
    Args:
        lam_per_area: Rate parameter λ (points per unit^2)
        xlim: X-coordinate limits (min, max)
        ylim: Y-coordinate limits (min, max)
        rng: Random number generator
        
    Returns:
        Tuple of (x_coords, y_coords) for generated points
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Calculate area
    A = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])
    
    # Sample number of points from Poisson distribution
    N = rng.poisson(lam_per_area * A)
    
    # Generate coordinates uniformly in the window
    x = rng.uniform(xlim[0], xlim[1], size=N)
    y = rng.uniform(ylim[0], ylim[1], size=N)
    
    return x, y


def filter_unreachable_targets(x: np.ndarray, 
                              y: np.ndarray, 
                              priorities: np.ndarray, 
                              fiber_types: np.ndarray = None,
                              robotgrid: Optional[kaiju.RobotGridNominal] = None,
                              is_scientific: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out targets that are not accessible by any robot.
    
    Args:
        x: Array of x coordinates
        y: Array of y coordinates
        priorities: Array of priority levels
        fiber_types: Array of fiber types ('BOSS' or 'APOGEE'), or None
        robotgrid: RobotGridNominal instance to check accessibility. If None, returns inputs unchanged.
        is_scientific: Array of boolean values indicating which targets are scientific, or None
        
    Returns:
        Filtered arrays: (x_filtered, y_filtered, priorities_filtered, fiber_types_filtered, is_scientific_filtered)
    """
    if robotgrid is None:
        # If no robotgrid provided, return inputs unchanged
        return x, y, priorities, fiber_types
    
    # Convert fiber type strings to kaiju enum
    fiber_type_map = {
        'BOSS': kaiju.cKaiju.BossFiber,
        'APOGEE': kaiju.cKaiju.ApogeeFiber
    }
    
    # Clear any existing targets from the robotgrid
    robotgrid.clearTargetDict()
    
    # Add all targets to the robotgrid
    for i in range(len(x)):
        target_id = i
        xyz_wok = [x[i], y[i], 0.0]  # z=0 for flat field
        
        if fiber_types is not None:
            fiber_type = fiber_type_map[fiber_types[i]]
        else:
            fiber_type = kaiju.cKaiju.BossFiber  # Default to BOSS
        
        priority = int(priorities[i])
        
        robotgrid.addTarget(
            targetID=target_id,
            xyzWok=xyz_wok,
            fiberType=fiber_type,
            priority=priority
        )
    
    # Get unreachable targets
    unreachable_target_ids = robotgrid.unreachableTargets()
    unreachable_set = set(unreachable_target_ids)
    
    # Create mask for reachable targets
    reachable_mask = np.array([i not in unreachable_set for i in range(len(x))])
    
    # Filter arrays
    x_filtered = x[reachable_mask]
    y_filtered = y[reachable_mask]
    priorities_filtered = priorities[reachable_mask]
    
    if fiber_types is not None:
        fiber_types_filtered = fiber_types[reachable_mask]
    else:
        fiber_types_filtered = None
    
    if is_scientific is not None:
        is_scientific_filtered = is_scientific[reachable_mask]
    else:
        is_scientific_filtered = None
    
    return x_filtered, y_filtered, priorities_filtered, fiber_types_filtered, is_scientific_filtered


def generate_targets(lam_per_area: float, 
                     xlim: Tuple[float, float], 
                     ylim: Tuple[float, float], 
                     rng: Optional[np.random.Generator] = None,
                     enable_fiber_types: bool = True,
                     robotgrid: Optional[kaiju.RobotGridNominal] = None,
                     num_priorities: int = 5,
                     scientific_percentage: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate targets using Poisson Point Process with priority levels and fiber types.
    Filters out targets that are not accessible by any robot if robotgrid is provided.

    Args:
        lam_per_area: Rate λ (points per unit^2 in the same units as x,y)
        xlim: X-coordinate limits (min, max)
        ylim: Y-coordinate limits (min, max)
        rng: Random number generator
        enable_fiber_types: If True, generates BOSS/APOGEE fiber types, otherwise returns None
        robotgrid: RobotGridNominal instance to check accessibility. If provided, filters out unreachable targets.
        num_priorities: Number of priority levels (default: 5, generates priorities 1 to num_priorities)
        scientific_percentage: Percentage of targets to mark as scientific (0.0 to 1.0, default: 0.0)
        
    Returns:
        x: Array of x coordinates
        y: Array of y coordinates  
        priorities: Array of priority levels (1 to num_priorities)
        fiber_types: Array of fiber types ('BOSS' or 'APOGEE'), or None if disabled
        is_scientific: Array of boolean values indicating which targets are scientific
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate coordinates using Poisson Point Process
    x, y = sample_ppp(lam_per_area, xlim, ylim, rng)
    
    # Generate priority levels (1 to num_priorities)
    priorities = rng.integers(1, num_priorities + 1, size=len(x))
    
    # Generate fiber types if enabled
    fiber_types = None
    if enable_fiber_types:
        fiber_types = rng.choice(['BOSS', 'APOGEE'], size=len(x))
    
    # Generate scientific targets boolean array if enabled
    is_scientific = None
    if scientific_percentage > 0.0:
        # Validate percentage range
        if scientific_percentage > 1.0:
            raise ValueError("scientific_percentage must be between 0.0 and 1.0")
        is_scientific = rng.random(size=len(x)) < scientific_percentage
    
    # Filter out unreachable targets if robotgrid is provided
    if robotgrid is not None:
        x, y, priorities, fiber_types, is_scientific = filter_unreachable_targets(
            x, y, priorities, fiber_types, robotgrid, is_scientific
        )
    
    return x, y, priorities, fiber_types, is_scientific


def generate_targets_with_custom_priorities(lam_per_area: float,
                                           xlim: Tuple[float, float],
                                           ylim: Tuple[float, float],
                                           priority_weights: Optional[np.ndarray] = None,
                                           rng: Optional[np.random.Generator] = None,
                                           enable_fiber_types: bool = True,
                                           robotgrid: Optional[kaiju.RobotGridNominal] = None,
                                           num_priorities: int = 5,
                                           scientific_percentage: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate targets with custom priority distribution.
    Filters out targets that are not accessible by any robot if robotgrid is provided.
    
    Args:
        lam_per_area: Rate λ (points per unit^2)
        xlim: X-coordinate limits (min, max)
        ylim: Y-coordinate limits (min, max)
        priority_weights: Weights for priority levels 1 to num_priorities (default: uniform)
        rng: Random number generator
        enable_fiber_types: If True, generates BOSS/APOGEE fiber types
        robotgrid: RobotGridNominal instance to check accessibility. If provided, filters out unreachable targets.
        num_priorities: Number of priority levels (default: 5, generates priorities 1 to num_priorities)
        scientific_percentage: Percentage of targets to mark as scientific (0.0 to 1.0, default: 0.0)
        
    Returns:
        x: Array of x coordinates
        y: Array of y coordinates
        priorities: Array of priority levels (1 to num_priorities)
        fiber_types: Array of fiber types ('BOSS' or 'APOGEE'), or None if disabled
        is_scientific: Array of boolean values indicating which targets are scientific
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate coordinates
    x, y = sample_ppp(lam_per_area, xlim, ylim, rng)
    
    # Generate priorities with custom weights
    if priority_weights is None:
        priority_weights = np.ones(num_priorities)  # Uniform distribution
    
    priorities = rng.choice(range(1, num_priorities + 1), size=len(x), p=priority_weights/np.sum(priority_weights))
    
    # Generate fiber types if enabled
    fiber_types = None
    if enable_fiber_types:
        fiber_types = rng.choice(['BOSS', 'APOGEE'], size=len(x))
    
    # Generate scientific targets boolean array if enabled
    is_scientific = None
    if scientific_percentage > 0.0:
        # Validate percentage range
        if scientific_percentage > 1.0:
            raise ValueError("scientific_percentage must be between 0.0 and 1.0")
        is_scientific = rng.random(size=len(x)) < scientific_percentage
    
    # Filter out unreachable targets if robotgrid is provided
    if robotgrid is not None:
        x, y, priorities, fiber_types, is_scientific = filter_unreachable_targets(
            x, y, priorities, fiber_types, robotgrid, is_scientific
        )
    
    return x, y, priorities, fiber_types, is_scientific


def print_target_summary(x: np.ndarray, y: np.ndarray, priorities: np.ndarray, fiber_types: np.ndarray = None, is_scientific: np.ndarray = None):
    """
    Print a summary of generated targets.
    
    Args:
        x: Array of x coordinates
        y: Array of y coordinates
        priorities: Array of priority levels
        fiber_types: Array of fiber types (optional)
        is_scientific: Array of boolean values indicating which targets are scientific (optional)
    """
    print(f"\n=== Target Generation Summary ===")
    print(f"Total targets: {len(x)}")
    print(f"Priority distribution:")
    
    # Get unique priority levels dynamically
    unique_priorities = np.unique(priorities)
    for p in sorted(unique_priorities):
        count = np.sum(priorities == p)
        print(f"  Priority {p}: {count} targets ({count/len(x)*100:.1f}%)")
    
    if fiber_types is not None:
        print(f"Fiber type distribution:")
        boss_count = np.sum(fiber_types == 'BOSS')
        apogee_count = np.sum(fiber_types == 'APOGEE')
        print(f"  BOSS: {boss_count} targets ({boss_count/len(x)*100:.1f}%)")
        print(f"  APOGEE: {apogee_count} targets ({apogee_count/len(x)*100:.1f}%)")
    
    if is_scientific is not None:
        print(f"Scientific targets distribution:")
        scientific_count = np.sum(is_scientific)
        non_scientific_count = len(x) - scientific_count
        print(f"  Scientific: {scientific_count} targets ({scientific_count/len(x)*100:.1f}%)")
        print(f"  Non-scientific: {non_scientific_count} targets ({non_scientific_count/len(x)*100:.1f}%)")


if __name__ == "__main__":
    # Example usage
    print("Generating targets with λ=0.001 per mm²...")
    x, y, priorities, fiber_types, is_scientific = generate_targets(
        lam_per_area=0.001,
        xlim=(-300, 300),
        ylim=(-300, 300),
        enable_fiber_types=True,
        num_priorities=5,
        scientific_percentage=0.3
    )
    
    print_target_summary(x, y, priorities, fiber_types, is_scientific)
    
    # Example with different number of priorities
    print("\nGenerating targets with 3 priority levels...")
    x2, y2, priorities2, fiber_types2, is_scientific2 = generate_targets(
        lam_per_area=0.001,
        xlim=(-300, 300),
        ylim=(-300, 300),
        enable_fiber_types=True,
        num_priorities=3,
        scientific_percentage=0.3
    )
    
    print_target_summary(x2, y2, priorities2, fiber_types2, is_scientific2)
    
    # Example with robotgrid filtering (requires kaiju environment)
    try:
        print("\nGenerating targets with robot accessibility filtering...")
        robotgrid = kaiju.RobotGridNominal()
        x_filtered, y_filtered, priorities_filtered, fiber_types_filtered, is_scientific_filtered = generate_targets(
            lam_per_area=0.001,
            xlim=(-300, 300),
            ylim=(-300, 300),
            enable_fiber_types=True,
            robotgrid=robotgrid,
            num_priorities=5,
            scientific_percentage=0.3
        )
        
        print(f"\nFiltered targets: {len(x_filtered)} (removed {len(x) - len(x_filtered)} unreachable)")
        print_target_summary(x_filtered, y_filtered, priorities_filtered, fiber_types_filtered, is_scientific_filtered)
    except Exception as e:
        print(f"RobotGrid filtering example skipped: {e}")