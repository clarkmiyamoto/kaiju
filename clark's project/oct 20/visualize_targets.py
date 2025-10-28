#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Target Visualization Script

This script visualizes a sample of targets for a given lambda value,
showing their positions, priorities, and fiber types.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

from config_robots import load_apo_grid, print_robot_summary
from targets import generate_targets, print_target_summary


def visualize_targets(lambda_value: float,
                     xlim: tuple = (-300, 300),
                     ylim: tuple = (-300, 300),
                     seed: int = 42,
                     show_robots: bool = True,
                     save_path: str = None):
    """
    Visualize targets for a given lambda value.
    
    Args:
        lambda_value: Target density (targets per unit^2)
        xlim: X-coordinate limits
        ylim: Y-coordinate limits
        seed: Random seed
        show_robots: Whether to show robot positions
        save_path: Path to save the plot
    """
    print(f"Generating targets with λ={lambda_value:.5f} per mm²...")
    
    # Generate targets
    x, y, priorities, fiber_types = generate_targets(
        lam_per_area=lambda_value,
        xlim=xlim,
        ylim=ylim,
        rng=np.random.default_rng(seed),
        enable_fiber_types=True
    )
    
    # Print target summary
    print_target_summary(x, y, priorities, fiber_types)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Load robot configuration if requested
    if show_robots:
        print("Loading robot configuration...")
        grid = load_apo_grid(seed=seed)
        
        # Plot robot positions
        robot_x = [robot.xPos for robot in grid.robotDict.values() if not robot.isOffline]
        robot_y = [robot.yPos for robot in grid.robotDict.values() if not robot.isOffline]
        ax.scatter(robot_x, robot_y, c='lightblue', s=20, alpha=0.6, 
                  label=f'Robots ({len(robot_x)})', marker='o', edgecolors='navy', linewidth=0.5)
    
    # Plot targets with different colors for fiber types
    boss_mask = fiber_types == 'BOSS'
    apogee_mask = fiber_types == 'APOGEE'
    
    # Plot BOSS targets
    if np.any(boss_mask):
        boss_x, boss_y = x[boss_mask], y[boss_mask]
        boss_priorities = priorities[boss_mask]
        
        # Color by priority
        scatter = ax.scatter(boss_x, boss_y, c=boss_priorities, s=60, 
                           cmap='Reds', alpha=0.8, label=f'BOSS targets ({len(boss_x)})',
                           marker='s', edgecolors='darkred', linewidth=1)
        
        # Add colorbar for BOSS priorities
        cbar1 = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar1.set_label('BOSS Priority (1-5)', rotation=270, labelpad=20)
    
    # Plot APOGEE targets
    if np.any(apogee_mask):
        apogee_x, apogee_y = x[apogee_mask], y[apogee_mask]
        apogee_priorities = priorities[apogee_mask]
        
        # Color by priority
        scatter = ax.scatter(apogee_x, apogee_y, c=apogee_priorities, s=60,
                           cmap='Blues', alpha=0.8, label=f'APOGEE targets ({len(apogee_x)})',
                           marker='^', edgecolors='darkblue', linewidth=1)
        
        # Add colorbar for APOGEE priorities
        cbar2 = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.15)
        cbar2.set_label('APOGEE Priority (1-5)', rotation=270, labelpad=20)
    
    # Customize plot
    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_title(f'Target Distribution (λ={lambda_value:.5f} targets/mm²)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Add statistics text
    stats_text = f"""Target Statistics:
Total Targets: {len(x)}
BOSS: {np.sum(boss_mask)} ({np.sum(boss_mask)/len(x)*100:.1f}%)
APOGEE: {np.sum(apogee_mask)} ({np.sum(apogee_mask)/len(x)*100:.1f}%)
Avg Priority: {np.mean(priorities):.1f}
Area: {(xlim[1]-xlim[0])*(ylim[1]-ylim[0]):.0f} mm²"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return x, y, priorities, fiber_types


def visualize_priority_distribution(priorities: np.ndarray, fiber_types: np.ndarray, save_path: str = None):
    """
    Visualize the priority distribution of targets.
    
    Args:
        priorities: Array of priority levels
        fiber_types: Array of fiber types
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall priority distribution
    priority_counts = np.bincount(priorities, minlength=6)[1:]  # Skip 0, get 1-5
    ax1.bar(range(1, 6), priority_counts, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('Priority Level')
    ax1.set_ylabel('Number of Targets')
    ax1.set_title('Overall Priority Distribution')
    ax1.set_xticks(range(1, 6))
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, count in enumerate(priority_counts):
        ax1.text(i+1, count + 0.5, str(count), ha='center', va='bottom')
    
    # Priority distribution by fiber type
    boss_mask = fiber_types == 'BOSS'
    apogee_mask = fiber_types == 'APOGEE'
    
    boss_priorities = priorities[boss_mask]
    apogee_priorities = priorities[apogee_mask]
    
    # Create grouped bar chart
    x = np.arange(1, 6)
    width = 0.35
    
    boss_counts = np.bincount(boss_priorities, minlength=6)[1:]
    apogee_counts = np.bincount(apogee_priorities, minlength=6)[1:]
    
    ax2.bar(x - width/2, boss_counts, width, label='BOSS', color='red', alpha=0.7)
    ax2.bar(x + width/2, apogee_counts, width, label='APOGEE', color='blue', alpha=0.7)
    
    ax2.set_xlabel('Priority Level')
    ax2.set_ylabel('Number of Targets')
    ax2.set_title('Priority Distribution by Fiber Type')
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Priority distribution plot saved to {save_path}")
    
    plt.show()


def visualize_spatial_density(x: np.ndarray, y: np.ndarray, 
                             xlim: tuple, ylim: tuple, 
                             save_path: str = None):
    """
    Visualize the spatial density of targets.
    
    Args:
        x, y: Target coordinates
        xlim, ylim: Coordinate limits
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot with density
    ax1.scatter(x, y, c='red', s=30, alpha=0.6)
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.set_title('Target Positions')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2D histogram/density plot
    h, xedges, yedges = np.histogram2d(x, y, bins=20, range=[xlim, ylim])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax2.imshow(h.T, origin='lower', extent=extent, cmap='Reds', aspect='equal')
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Y Position (mm)')
    ax2.set_title('Target Density Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Target Count', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spatial density plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Visualize target samples')
    
    parser.add_argument('--lambda', type=float, default=0.001,
                       help='Target density λ (default: 0.001)')
    parser.add_argument('--xlim', nargs=2, type=float, default=[-300, 300],
                       help='X-coordinate limits (default: -300 300)')
    parser.add_argument('--ylim', nargs=2, type=float, default=[-300, 300],
                       help='Y-coordinate limits (default: -300 300)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no-robots', action='store_true',
                       help='Hide robot positions')
    parser.add_argument('--output', type=str, default='target_visualization.png',
                       help='Output file path (default: target_visualization.png)')
    parser.add_argument('--priority-plot', type=str, default='priority_distribution.png',
                       help='Priority distribution plot path (default: priority_distribution.png)')
    parser.add_argument('--density-plot', type=str, default='spatial_density.png',
                       help='Spatial density plot path (default: spatial_density.png)')
    
    args = parser.parse_args()
    
    print("=== Target Visualization ===")
    print(f"Lambda: {getattr(args, 'lambda'):.5f} targets/mm²")
    print(f"Area: {args.xlim} × {args.ylim} mm")
    print(f"Seed: {args.seed}")
    print(f"Show robots: {not args.no_robots}")
    
    # Generate and visualize targets
    x, y, priorities, fiber_types = visualize_targets(
        lambda_value=getattr(args, 'lambda'),
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
        seed=args.seed,
        show_robots=not args.no_robots,
        save_path=args.output
    )
    
    # Generate additional visualizations
    print("\nGenerating priority distribution plot...")
    visualize_priority_distribution(priorities, fiber_types, args.priority_plot)
    
    print("\nGenerating spatial density plot...")
    visualize_spatial_density(x, y, tuple(args.xlim), tuple(args.ylim), args.density_plot)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
