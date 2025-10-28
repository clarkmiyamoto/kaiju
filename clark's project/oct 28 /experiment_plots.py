#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment Results Visualization Module

This module provides functions for creating various types of plots
for robot assignment experiment results, including comparison plots
and assignment visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Dict, Any, Tuple, Optional


def plot_assignment_results(robotgrid, targets, assignments, lambda_val, strategy, output_dir, is_scientific=None):
    """
    Plot targets and robots showing assignment status.
    
    Args:
        robotgrid: kaiju RobotGrid instance
        targets: Tuple of (x, y) target coordinates
        assignments: Dictionary of robot_id -> target_id assignments
        lambda_val: Lambda value for this experiment
        strategy: Strategy name ('greedy' or 'cp')
        output_dir: Output directory for saving plots
        is_scientific: Array of boolean values indicating which targets are scientific (optional)
    """
    x, y = targets
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Separate targets into assigned/unassigned and scientific/non-scientific
    assigned_scientific_x = []
    assigned_scientific_y = []
    assigned_non_scientific_x = []
    assigned_non_scientific_y = []
    unassigned_scientific_x = []
    unassigned_scientific_y = []
    unassigned_non_scientific_x = []
    unassigned_non_scientific_y = []
    
    # Get all assigned target IDs
    assigned_target_ids = set(assignments.values())
    
    for i, (target_x, target_y) in enumerate(zip(x, y)):
        is_target_scientific = is_scientific[i] if is_scientific is not None else False
        
        if i in assigned_target_ids:
            if is_target_scientific:
                assigned_scientific_x.append(target_x)
                assigned_scientific_y.append(target_y)
            else:
                assigned_non_scientific_x.append(target_x)
                assigned_non_scientific_y.append(target_y)
        else:
            if is_target_scientific:
                unassigned_scientific_x.append(target_x)
                unassigned_scientific_y.append(target_y)
            else:
                unassigned_non_scientific_x.append(target_x)
                unassigned_non_scientific_y.append(target_y)
    
    # Plot assigned scientific targets in dark blue
    if assigned_scientific_x:
        ax.scatter(assigned_scientific_x, assigned_scientific_y, s=60, c='darkblue', alpha=0.9, 
                  label=f'Assigned Scientific ({len(assigned_scientific_x)})', marker='^', 
                  edgecolors='navy', linewidth=1)
    
    # Plot assigned non-scientific targets in blue
    if assigned_non_scientific_x:
        ax.scatter(assigned_non_scientific_x, assigned_non_scientific_y, s=60, c='blue', alpha=0.8, 
                  label=f'Assigned Non-Scientific ({len(assigned_non_scientific_x)})', marker='^', 
                  edgecolors='darkblue', linewidth=1)
    
    # Plot unassigned scientific targets in orange
    if unassigned_scientific_x:
        ax.scatter(unassigned_scientific_x, unassigned_scientific_y, s=60, c='orange', alpha=0.8, 
                  label=f'Unassigned Scientific ({len(unassigned_scientific_x)})', marker='^', 
                  edgecolors='darkorange', linewidth=1)
    
    # Plot unassigned non-scientific targets in gray
    if unassigned_non_scientific_x:
        ax.scatter(unassigned_non_scientific_x, unassigned_non_scientific_y, s=60, c='gray', alpha=0.6, 
                  label=f'Unassigned Non-Scientific ({len(unassigned_non_scientific_x)})', marker='^', 
                  edgecolors='darkgray', linewidth=1)
    
    # Separate robots into assigned and unassigned
    assigned_robots_x = []
    assigned_robots_y = []
    unassigned_robots_x = []
    unassigned_robots_y = []
    
    for robot_id, robot in robotgrid.robotDict.items():
        if robot.isOffline:
            continue
            
        if robot_id in assignments:
            # Robot is assigned
            assigned_robots_x.append(robot.xPos)
            assigned_robots_y.append(robot.yPos)
        else:
            # Robot is unassigned
            unassigned_robots_x.append(robot.xPos)
            unassigned_robots_y.append(robot.yPos)
    
    # Plot assigned robots in light blue
    if assigned_robots_x:
        ax.scatter(assigned_robots_x, assigned_robots_y, c='lightblue', s=20, alpha=0.8,
                  label=f'Assigned Robots ({len(assigned_robots_x)})', marker='o')
    
    # Plot unassigned robots in red
    if unassigned_robots_x:
        ax.scatter(unassigned_robots_x, unassigned_robots_y, c='red', s=20, alpha=0.8,
                  label=f'Unassigned Robots ({len(unassigned_robots_x)})', marker='o')
    
    # Set plot properties
    xlim = (-300, 300)
    ylim = (-300, 300)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_title(f'{strategy.upper()} Algorithm Assignment Results\n'
                f'λ = {lambda_val:.5f}, Targets: {len(x)}, '
                f'Assigned: {len(assignments)}/{robotgrid.nRobots}', 
                fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add text box with statistics
    stats_text = f'Strategy: {strategy.upper()}\n'
    stats_text += f'Target Density: {lambda_val:.5f}\n'
    stats_text += f'Total Targets: {len(x)}\n'
    
    # Calculate totals for assigned/unassigned targets
    total_assigned = len(assigned_scientific_x) + len(assigned_non_scientific_x)
    total_unassigned = len(unassigned_scientific_x) + len(unassigned_non_scientific_x)
    
    stats_text += f'Assigned Targets: {total_assigned}\n'
    stats_text += f'Unassigned Targets: {total_unassigned}\n'
    
    # Add scientific targets information if available
    if is_scientific is not None:
        total_scientific = len(assigned_scientific_x) + len(unassigned_scientific_x)
        total_non_scientific = len(assigned_non_scientific_x) + len(unassigned_non_scientific_x)
        stats_text += f'Scientific Targets: {total_scientific}\n'
        stats_text += f'Non-Scientific Targets: {total_non_scientific}\n'
        stats_text += f'Assigned Scientific: {len(assigned_scientific_x)}\n'
        stats_text += f'Unassigned Scientific: {len(unassigned_scientific_x)}\n'
    
    stats_text += f'Total Robots: {robotgrid.nRobots}\n'
    stats_text += f'Assigned Robots: {len(assignments)}\n'
    stats_text += f'Unassigned Robots: {len(unassigned_robots_x)}\n'
    stats_text += f'Robot Assignment Rate: {len(assignments)/robotgrid.nRobots*100:.1f}%\n'
    stats_text += f'Target Assignment Rate: {total_assigned/len(x)*100:.1f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'assignment_results_{strategy}_lambda_{lambda_val:.5f}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def create_comparison_plots(df: pd.DataFrame, 
                         algorithms: List[str], 
                         scientific_percentage: float,
                         output_dir: str) -> str:
    """
    Create comparison plots showing algorithm performance.
    
    Args:
        df: DataFrame with experiment results
        algorithms: List of algorithm names
        scientific_percentage: Percentage of scientific targets (0.0 to 1.0)
        output_dir: Output directory for saving plots
        
    Returns:
        Path to saved comparison plot
    """
    if len(df) == 0:
        return None
        
    has_scientific = scientific_percentage > 0
    if has_scientific:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Number of assigned targets vs lambda
    if has_scientific:
        # Show total assigned (leftmost) and separate lines for scientific and non-scientific targets (middle)
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, algorithm in enumerate(algorithms):
            alg_data = df[df['strategy'] == algorithm]
            if len(alg_data) > 0:
                color = colors[i % len(colors)]
                # Total assigned targets (prefer sum of components if present, else fallback)
                if {'num_assigned_scientific', 'num_assigned_non_scientific'}.issubset(alg_data.columns):
                    total_assigned = alg_data['num_assigned_scientific'] + alg_data['num_assigned_non_scientific']
                else:
                    total_assigned = alg_data['num_assigned']
                ax0.plot(alg_data['lambda'], total_assigned,
                         marker='^', linestyle='-', color=color, label=f'{algorithm.upper()} Total', linewidth=2)
                # Scientific targets (solid line)
                ax1.plot(alg_data['lambda'], alg_data['num_assigned_scientific'], 
                        marker='o', linestyle='-', color=color, label=f'{algorithm.upper()} Scientific', linewidth=2)
                # Non-scientific targets (dotted line)
                ax1.plot(alg_data['lambda'], alg_data['num_assigned_non_scientific'], 
                        marker='s', linestyle=':', color=color, label=f'{algorithm.upper()} Non-Scientific', linewidth=2)
                ax1.set_yscale('log')
        ax0.set_xlabel('Lambda (λ)')
        ax0.set_ylabel('Number of Assigned Targets')
        ax0.set_title('Total Assigned Targets vs Target Density')
        ax0.legend()
        ax0.grid(True, alpha=0.3)
    else:
        # Show total assigned targets only
        for algorithm in algorithms:
            alg_data = df[df['strategy'] == algorithm]
            if len(alg_data) > 0:
                ax1.plot(alg_data['lambda'], alg_data['num_assigned'], 
                        marker='o', label=f'{algorithm.upper()} Algorithm', linewidth=2)
    
    ax1.set_xlabel('Lambda (λ)')
    ax1.set_ylabel('Number of Assigned Targets')
    ax1.set_title('Assigned Targets vs Target Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Computation time vs lambda
    for algorithm in algorithms:
        alg_data = df[df['strategy'] == algorithm]
        if len(alg_data) > 0:
            ax2.plot(alg_data['lambda'], alg_data['elapsed_time'], 
                    marker='s', label=f'{algorithm.upper()} Algorithm', linewidth=2)
    
    ax2.set_xlabel('Lambda (λ)')
    ax2.set_ylabel('Computation Time (seconds)')
    ax2.set_title('Computation Time vs Target Density')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    plot_path = os.path.join(output_dir, 'comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def print_summary_statistics(df: pd.DataFrame, algorithms: List[str]) -> None:
    """
    Print summary statistics for the experiment results.
    
    Args:
        df: DataFrame with experiment results
        algorithms: List of algorithm names
    """
    print(f"\n=== Summary Statistics ===")
    for algorithm in algorithms:
        alg_data = df[df['strategy'] == algorithm]
        if len(alg_data) > 0:
            print(f"\n{algorithm.upper()} Algorithm:")
            print(f"  Average assigned targets: {alg_data['num_assigned'].mean():.1f}")
            print(f"  Average computation time: {alg_data['elapsed_time'].mean():.3f} seconds")
            print(f"  Max assigned targets: {alg_data['num_assigned'].max()}")
            print(f"  Max computation time: {alg_data['elapsed_time'].max():.3f} seconds")
