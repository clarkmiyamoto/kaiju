#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Target-Robot Visualization Module

This module creates plots showing targets overlaid on the robot grid
for different target densities and configurations. This is separate from
the experiment results plotting functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional
from config_robots import load_apo_grid, get_robots_by_fiber_type
from targets import generate_targets
from target_utils import add_targets_to_robotgrid, clear_robotgrid_assignments


def plot_targets_on_robots(lam_per_area: float,
                          xlim: Tuple[float, float] = (-300, 300),
                          ylim: Tuple[float, float] = (-300, 300),
                          seed: int = 42,
                          output_dir: str = 'output',
                          show_assigned: bool = False,
                          title_suffix: str = '') -> str:
    """
    Create a plot showing targets overlaid on the robot grid.
    
    Args:
        lam_per_area: Target density parameter λ
        xlim: X-coordinate limits
        ylim: Y-coordinate limits
        seed: Random seed for reproducibility
        output_dir: Directory to save the plot
        show_assigned: Whether to show assignment status
        title_suffix: Additional text for plot title
        
    Returns:
        Path to saved plot file
    """
    # Set up random number generator
    rng = np.random.default_rng(seed)
    
    # Generate targets
    x, y, priorities, fiber_types = generate_targets(
        lam_per_area=lam_per_area,
        xlim=xlim,
        ylim=ylim,
        rng=rng,
        enable_fiber_types=True
    )
    
    # Load robot grid
    robotgrid = load_apo_grid(seed=rng.integers(0, 2**31))
    
    # Add targets to robotgrid
    add_targets_to_robotgrid(robotgrid, x, y, priorities, fiber_types)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot all robots uniformly (no fiber type distinction)
    robot_x = [robot.xPos for robot in robotgrid.robotDict.values() if not robot.isOffline]
    robot_y = [robot.yPos for robot in robotgrid.robotDict.values() if not robot.isOffline]
    ax.scatter(robot_x, robot_y, c='gray', s=20, alpha=0.6, 
              label=f'Robots ({len(robot_x)})', marker='o')
    
    # Plot all targets uniformly (no fiber type distinction)
    ax.scatter(x, y, s=60,
                           cmap='Blues', alpha=0.8, label=f'Targets ({len(x)})',
                           marker='^', edgecolors='darkblue', linewidth=1)
            
    
    # Set plot properties
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_title(f'Targets Overlaid on Robot Grid{title_suffix}\n'
                f'λ = {lam_per_area:.5f}, Total Targets: {len(x)}, Total Robots: {robotgrid.nRobots}', 
                fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add text box with statistics
    stats_text = f'Target Density: {lam_per_area:.5f}\n'
    stats_text += f'Total Targets: {len(x)}\n'
    stats_text += f'Total Robots: {robotgrid.nRobots}\n'
    stats_text += f'Active Robots: {len(robot_x)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'targets_robots_lambda_{lam_per_area:.5f}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def create_multiple_density_plots(lambda_values: list,
                                 xlim: Tuple[float, float] = (-300, 300),
                                 ylim: Tuple[float, float] = (-300, 300),
                                 seed: int = 42,
                                 output_dir: str = 'output') -> list:
    """
    Create plots for multiple target densities.
    
    Args:
        lambda_values: List of lambda values to plot
        xlim: X-coordinate limits
        ylim: Y-coordinate limits
        seed: Random seed for reproducibility
        output_dir: Directory to save plots
        
    Returns:
        List of saved plot file paths
    """
    saved_plots = []
    
    print(f"Creating plots for {len(lambda_values)} different target densities...")
    
    for i, lam in enumerate(lambda_values):
        print(f"  Plotting λ = {lam:.5f} ({i+1}/{len(lambda_values)})")
        try:
            filepath = plot_targets_on_robots(
                lam_per_area=lam,
                xlim=xlim,
                ylim=ylim,
                seed=seed,
                output_dir=output_dir,
                title_suffix=f' (λ = {lam:.5f})'
            )
            saved_plots.append(filepath)
        except Exception as e:
            print(f"    Error creating plot for λ = {lam:.5f}: {str(e)}")
    
    return saved_plots


def create_comparison_plot(lambda_values: list,
                          xlim: Tuple[float, float] = (-300, 300),
                          ylim: Tuple[float, float] = (-300, 300),
                          seed: int = 42,
                          output_dir: str = 'output') -> str:
    """
    Create a comparison plot showing multiple target densities in subplots.
    
    Args:
        lambda_values: List of lambda values to plot
        xlim: X-coordinate limits
        ylim: Y-coordinate limits
        seed: Random seed for reproducibility
        output_dir: Directory to save plot
        
    Returns:
        Path to saved comparison plot
    """
    n_plots = len(lambda_values)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    rng = np.random.default_rng(seed)
    
    for i, lam in enumerate(lambda_values):
        ax = axes_flat[i]
        
        # Generate targets
        x, y, priorities, fiber_types = generate_targets(
            lam_per_area=lam,
            xlim=xlim,
            ylim=ylim,
            rng=rng,
            enable_fiber_types=True
        )
        
        # Load robot grid
        robotgrid = load_apo_grid(seed=rng.integers(0, 2**31))
        
        # Plot robots (simplified - just show all robots as gray dots)
        robot_x = [robot.xPos for robot in robotgrid.robotDict.values() if not robot.isOffline]
        robot_y = [robot.yPos for robot in robotgrid.robotDict.values() if not robot.isOffline]
        ax.scatter(robot_x, robot_y, c='gray', s=8, alpha=0.4, marker='o')
        
        # Plot all targets uniformly (no fiber type distinction)
        ax.scatter(x, y, c='blue', s=20, alpha=0.7, marker='o')
        
        # Set subplot properties
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'λ = {lam:.5f}\nTargets: {len(x)}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Only show axis labels on leftmost and bottom subplots
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('X (mm)', fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel('Y (mm)', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Target Distribution vs Robot Grid\n(Different Target Densities)', fontsize=16)
    plt.tight_layout()
    
    # Save comparison plot
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'targets_robots_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


if __name__ == "__main__":
    # Example usage
    print("=== Target-Robot Visualization ===")
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define lambda values for different target densities
    lambda_values = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    
    print(f"Creating individual plots for {len(lambda_values)} densities...")
    individual_plots = create_multiple_density_plots(
        lambda_values=lambda_values,
        seed=42,
        output_dir=output_dir
    )
    
    print(f"\nCreated {len(individual_plots)} individual plots:")
    for plot_path in individual_plots:
        print(f"  {plot_path}")
    
    print(f"\nCreating comparison plot...")
    comparison_plot = create_comparison_plot(
        lambda_values=lambda_values,
        seed=42,
        output_dir=output_dir
    )
    print(f"Comparison plot saved to: {comparison_plot}")
    
    print(f"\nAll plots saved to: {output_dir}/")
    print("Visualization complete!")
