#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Master Experiment Runner

This script runs experiments comparing greedy vs constraint programming
assignment algorithms with configurable parameters.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from typing import List, Dict, Any

from config_robots import load_apo_grid, get_robot_stats
from targets import generate_targets
from greedy_solver import greedy_assignment, greedy_assignment_with_collision_check
from cp_solver import cp_assignment, cp_assignment_simple


def run_experiment(lambda_value: float,
                  xlim: tuple,
                  ylim: tuple,
                  algorithm: str,
                  collision_buffer: float,
                  time_limit: float,
                  seed: int,
                  num_workers: int = 0,
                  verbose: bool = False) -> Dict[str, Any]:
    """
    Run a single experiment with given parameters.
    
    Args:
        lambda_value: Target density (targets per unit^2)
        xlim: X-coordinate limits
        ylim: Y-coordinate limits
        algorithm: 'greedy', 'cp', or 'both'
        collision_buffer: Collision distance threshold
        time_limit: CP solver time limit
        seed: Random seed
        num_workers: Number of worker threads for CP solver (0 = use all cores)
        verbose: Whether to print progress
        
    Returns:
        Dictionary with experiment results
    """
    if verbose:
        print(f"\nRunning experiment: λ={lambda_value:.5f}, algorithm={algorithm}")
    
    # Generate targets
    x, y, priorities, fiber_types = generate_targets(
        lam_per_area=lambda_value,
        xlim=xlim,
        ylim=ylim,
        rng=np.random.default_rng(seed),
        enable_fiber_types=True
    )
    
    results = {
        'lambda': lambda_value,
        'n_targets': len(x),
        'algorithm': algorithm,
        'collision_buffer': collision_buffer,
        'seed': seed
    }
    
    # Run greedy algorithm
    if algorithm in ['greedy', 'both']:
        if verbose:
            print("  Running greedy algorithm...")
        
        grid_greedy = load_apo_grid(seed=seed)
        start_time = time.time()
        
        greedy_priority, greedy_assignments = greedy_assignment(
            grid_greedy, x, y, priorities, fiber_types, verbose=False
        )
        
        greedy_time = time.time() - start_time
        
        results.update({
            'greedy_priority': greedy_priority,
            'greedy_assignments': len(greedy_assignments),
            'greedy_time': greedy_time
        })
    
    # Run CP algorithm
    if algorithm in ['cp', 'both']:
        if verbose:
            print("  Running CP algorithm...")
        
        grid_cp = load_apo_grid(seed=seed)
        start_time = time.time()
        
        cp_priority, cp_assignments, cp_solver_time = cp_assignment_simple(
            grid_cp, x, y, priorities, fiber_types, time_limit=time_limit, num_workers=num_workers, verbose=False
        )
        
        cp_total_time = time.time() - start_time
        
        results.update({
            'cp_priority': cp_priority,
            'cp_assignments': len(cp_assignments),
            'cp_solver_time': cp_solver_time,
            'cp_total_time': cp_total_time
        })
    
    return results


def run_lambda_sweep(lambda_min: float,
                    lambda_max: float,
                    num_points: int,
                    algorithm: str,
                    collision_buffer: float,
                    time_limit: float,
                    seed: int,
                    num_workers: int = 0,
                    xlim: tuple = (-300, 300),
                    ylim: tuple = (-300, 300),
                    verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run experiments across a range of lambda values.
    
    Args:
        lambda_min: Minimum lambda value
        lambda_max: Maximum lambda value
        num_points: Number of lambda values to test
        algorithm: 'greedy', 'cp', or 'both'
        collision_buffer: Collision distance threshold
        time_limit: CP solver time limit
        seed: Random seed
        num_workers: Number of worker threads for CP solver (0 = use all cores)
        xlim: X-coordinate limits
        ylim: Y-coordinate limits
        verbose: Whether to print progress
        
    Returns:
        List of experiment results
    """
    lambda_values = np.linspace(lambda_min, lambda_max, num_points)
    results = []
    
    if verbose:
        print(f"Running lambda sweep: {lambda_min:.5f} to {lambda_max:.5f} ({num_points} points)")
        print(f"Algorithm: {algorithm}, Collision buffer: {collision_buffer}mm")
    
    for i, lam in enumerate(lambda_values):
        if verbose:
            print(f"\nProgress: {i+1}/{num_points}")
        
        result = run_experiment(
            lambda_value=lam,
            xlim=xlim,
            ylim=ylim,
            algorithm=algorithm,
            collision_buffer=collision_buffer,
            time_limit=time_limit,
            seed=seed,
            num_workers=num_workers,
            verbose=verbose
        )
        
        results.append(result)
    
    return results


def plot_results(results: List[Dict[str, Any]], output_dir: str = "output"):
    """
    Generate plots from experiment results.
    
    Args:
        results: List of experiment results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Priority vs Lambda
    ax1 = axes[0, 0]
    if 'greedy_priority' in df.columns:
        ax1.plot(df['lambda'], df['greedy_priority'], 'bo-', label='Greedy', linewidth=2, markersize=6)
    if 'cp_priority' in df.columns:
        ax1.plot(df['lambda'], df['cp_priority'], 'ro-', label='CP Optimal', linewidth=2, markersize=6)
    ax1.set_xlabel('Lambda per Area (targets/mm²)')
    ax1.set_ylabel('Total Priority Gained')
    ax1.set_title('Priority Comparison: Greedy vs CP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Assignments vs Lambda
    ax2 = axes[0, 1]
    if 'greedy_assignments' in df.columns:
        ax2.plot(df['lambda'], df['greedy_assignments'], 'bo-', label='Greedy', linewidth=2, markersize=6)
    if 'cp_assignments' in df.columns:
        ax2.plot(df['lambda'], df['cp_assignments'], 'ro-', label='CP Optimal', linewidth=2, markersize=6)
    ax2.set_xlabel('Lambda per Area (targets/mm²)')
    ax2.set_ylabel('Number of Assignments')
    ax2.set_title('Assignment Count: Greedy vs CP')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Solver Time vs Lambda
    ax3 = axes[1, 0]
    if 'greedy_time' in df.columns:
        ax3.plot(df['lambda'], df['greedy_time'], 'bo-', label='Greedy', linewidth=2, markersize=6)
    if 'cp_solver_time' in df.columns:
        ax3.plot(df['lambda'], df['cp_solver_time'], 'ro-', label='CP Solver', linewidth=2, markersize=6)
    if 'cp_total_time' in df.columns:
        ax3.plot(df['lambda'], df['cp_total_time'], 'go-', label='CP Total', linewidth=2, markersize=6)
    ax3.set_xlabel('Lambda per Area (targets/mm²)')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Performance: Greedy vs CP')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Improvement vs Lambda
    ax4 = axes[1, 1]
    if 'greedy_priority' in df.columns and 'cp_priority' in df.columns:
        improvement = df['cp_priority'] - df['greedy_priority']
        ax4.plot(df['lambda'], improvement, 'mo-', linewidth=2, markersize=6)
        ax4.set_xlabel('Lambda per Area (targets/mm²)')
        ax4.set_ylabel('Priority Improvement (CP - Greedy)')
        ax4.set_title('CP Advantage over Greedy')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "assignment_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    plt.show()


def save_results(results: List[Dict[str, Any]], output_dir: str = "output"):
    """
    Save experiment results to CSV.
    
    Args:
        results: List of experiment results
        output_dir: Directory to save CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Print summary statistics
    print(f"\n=== Experiment Summary ===")
    print(f"Total experiments: {len(results)}")
    print(f"Lambda range: {df['lambda'].min():.5f} to {df['lambda'].max():.5f}")
    print(f"Target count range: {df['n_targets'].min()} to {df['n_targets'].max()}")
    
    if 'greedy_priority' in df.columns and 'cp_priority' in df.columns:
        improvement = df['cp_priority'] - df['greedy_priority']
        print(f"Average CP improvement: {improvement.mean():.2f} priority points")
        print(f"Maximum CP improvement: {improvement.max():.2f} priority points")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Run robot assignment experiments')
    
    # Experiment parameters
    parser.add_argument('--lambda-min', type=float, default=0.0001,
                       help='Minimum lambda value (default: 0.0001)')
    parser.add_argument('--lambda-max', type=float, default=0.002,
                       help='Maximum lambda value (default: 0.002)')
    parser.add_argument('--num-points', type=int, default=5,
                       help='Number of lambda values to test (default: 20)')
    parser.add_argument('--algorithm', choices=['greedy', 'cp', 'both'], default='both',
                       help='Algorithm to run (default: both)')
    
    # Solver parameters
    parser.add_argument('--collision-buffer', type=float, default=50.0,
                       help='Collision distance threshold in mm (default: 50.0)')
    parser.add_argument('--time-limit', type=float, default=30.0,
                       help='CP solver time limit in seconds (default: 30.0)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of worker threads for CP solver (default: 0 = use all cores)')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("=== Robot Assignment Experiment Runner ===")
    print(f"Lambda range: {args.lambda_min:.5f} to {args.lambda_max:.5f}")
    print(f"Number of points: {args.num_points}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Collision buffer: {args.collision_buffer}mm")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed}")
    
    # Run experiments
    results = run_lambda_sweep(
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        num_points=args.num_points,
        algorithm=args.algorithm,
        collision_buffer=args.collision_buffer,
        time_limit=args.time_limit,
        seed=args.seed,
        num_workers=args.num_workers,
        verbose=args.verbose
    )
    
    # Generate plots and save results
    plot_results(results, args.output)
    save_results(results, args.output)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
