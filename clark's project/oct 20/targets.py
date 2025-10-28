#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Target Generation Module

This module generates astronomical targets using Poisson Point Process
for realistic target density simulation.
"""

from typing import Tuple, Optional
import numpy as np


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


def generate_targets(lam_per_area: float, 
                     xlim: Tuple[float, float], 
                     ylim: Tuple[float, float], 
                     rng: Optional[np.random.Generator] = None,
                     enable_fiber_types: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate targets using Poisson Point Process with priority levels and fiber types.

    Args:
        lam_per_area: Rate λ (points per unit^2 in the same units as x,y)
        xlim: X-coordinate limits (min, max)
        ylim: Y-coordinate limits (min, max)
        rng: Random number generator
        enable_fiber_types: If True, generates BOSS/APOGEE fiber types, otherwise returns None
        
    Returns:
        x: Array of x coordinates
        y: Array of y coordinates  
        priorities: Array of priority levels (1-5)
        fiber_types: Array of fiber types ('BOSS' or 'APOGEE'), or None if disabled
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate coordinates using Poisson Point Process
    x, y = sample_ppp(lam_per_area, xlim, ylim, rng)
    
    # Generate priority levels (1-5)
    priorities = rng.integers(1, 6, size=len(x))
    
    # Generate fiber types if enabled
    fiber_types = None
    if enable_fiber_types:
        fiber_types = rng.choice(['BOSS', 'APOGEE'], size=len(x))
    
    return x, y, priorities, fiber_types


def generate_targets_with_custom_priorities(lam_per_area: float,
                                           xlim: Tuple[float, float],
                                           ylim: Tuple[float, float],
                                           priority_weights: Optional[np.ndarray] = None,
                                           rng: Optional[np.random.Generator] = None,
                                           enable_fiber_types: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate targets with custom priority distribution.
    
    Args:
        lam_per_area: Rate λ (points per unit^2)
        xlim: X-coordinate limits (min, max)
        ylim: Y-coordinate limits (min, max)
        priority_weights: Weights for priority levels 1-5 (default: uniform)
        rng: Random number generator
        enable_fiber_types: If True, generates BOSS/APOGEE fiber types
        
    Returns:
        x: Array of x coordinates
        y: Array of y coordinates
        priorities: Array of priority levels (1-5)
        fiber_types: Array of fiber types ('BOSS' or 'APOGEE'), or None if disabled
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate coordinates
    x, y = sample_ppp(lam_per_area, xlim, ylim, rng)
    
    # Generate priorities with custom weights
    if priority_weights is None:
        priority_weights = np.ones(5)  # Uniform distribution
    
    priorities = rng.choice([1, 2, 3, 4, 5], size=len(x), p=priority_weights/np.sum(priority_weights))
    
    # Generate fiber types if enabled
    fiber_types = None
    if enable_fiber_types:
        fiber_types = rng.choice(['BOSS', 'APOGEE'], size=len(x))
    
    return x, y, priorities, fiber_types


def print_target_summary(x: np.ndarray, y: np.ndarray, priorities: np.ndarray, fiber_types: np.ndarray = None):
    """
    Print a summary of generated targets.
    
    Args:
        x: Array of x coordinates
        y: Array of y coordinates
        priorities: Array of priority levels
        fiber_types: Array of fiber types (optional)
    """
    print(f"\n=== Target Generation Summary ===")
    print(f"Total targets: {len(x)}")
    print(f"Priority distribution:")
    for p in range(1, 6):
        count = np.sum(priorities == p)
        print(f"  Priority {p}: {count} targets ({count/len(x)*100:.1f}%)")
    
    if fiber_types is not None:
        print(f"Fiber type distribution:")
        boss_count = np.sum(fiber_types == 'BOSS')
        apogee_count = np.sum(fiber_types == 'APOGEE')
        print(f"  BOSS: {boss_count} targets ({boss_count/len(x)*100:.1f}%)")
        print(f"  APOGEE: {apogee_count} targets ({apogee_count/len(x)*100:.1f}%)")


if __name__ == "__main__":
    # Example usage
    print("Generating targets with λ=0.001 per mm²...")
    x, y, priorities, fiber_types = generate_targets(
        lam_per_area=0.001,
        xlim=(-300, 300),
        ylim=(-300, 300),
        enable_fiber_types=True
    )
    
    print_target_summary(x, y, priorities, fiber_types)