import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from config_robots import load_apo_grid
from targets import sample_ppp, generate_targets
from strategies.multi_priority.greedy import Greedy
from strategies.multi_priority.cp import ORToolsStrictPriority
from target_utils import add_targets_to_robotgrid, clear_robotgrid_assignments

def run_greedy(robotgrid, targets, priorities, seed, is_scientific=None):
    strategy = Greedy(robotgrid=robotgrid, targets=targets, priorities=priorities, seed=seed, is_scientific=is_scientific)
    strategy.run_optimizer()
    return strategy.get_assignment_summary()

def run_ortools(robotgrid, targets, priorities, seed, time_limit):
    strategy = ORToolsStrictPriority(robotgrid=robotgrid, targets=targets, priorities=priorities, seed=seed, time_limit=time_limit)
    strategy.run_optimizer()
    return strategy.get_assignment_summary()

def run_experiment(lambda_: float, strategy: str, rng, time_limit: float = 30.0, plot_results: bool = False, output_dir: str = 'output', num_priorities: int = 5, scientific_percentage: float = 0.0):
    """
    Run a single experiment with given lambda value.
    
    Args:
        lambda_: Target density parameter
        strategy: 'greedy' or 'cp'
        rng: Random number generator
        time_limit: Time limit for CP solver
        plot_results: Whether to generate assignment plots
        output_dir: Output directory for plots
        num_priorities: Number of priority levels to generate
        scientific_percentage: Percentage of targets to mark as scientific (0.0 to 1.0, default: 0.0)
    Returns:
        Dictionary with experiment results
    """
    # Load robot grid
    robotgrid = load_apo_grid(seed=rng.integers(0, 2**31))
    
    # Generate targets with filtering
    x, y, priorities, fiber_types, is_scientific = generate_targets(
        lam_per_area=lambda_,
        xlim=(-300, 300),
        ylim=(-300, 300),
        rng=rng,
        enable_fiber_types=True,
        robotgrid=robotgrid,
        num_priorities=num_priorities,
        scientific_percentage=scientific_percentage
    )
    
    # Run the specified strategy
    if strategy == 'greedy':
        results = run_greedy(robotgrid, (x, y), priorities, rng.integers(0, 2**31), is_scientific)
    elif strategy == 'cp':
        results = run_ortools(robotgrid, (x, y), priorities, rng.integers(0, 2**31), time_limit)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Add experiment metadata
    results['lambda'] = lambda_
    results['strategy'] = strategy 
    results['num_targets'] = len(x)
    
    # Add scientific targets information if available
    if is_scientific is not None:
        results['num_scientific_targets'] = np.sum(is_scientific)
        results['num_non_scientific_targets'] = len(x) - np.sum(is_scientific)
        results['scientific_percentage'] = scientific_percentage
    else:
        results['num_scientific_targets'] = 0
        results['num_non_scientific_targets'] = len(x)
        results['scientific_percentage'] = 0.0
    
    # Generate assignment plot if requested
    if plot_results and 'assignments' in results:
        plot_file = plot_assignment_results(
            robotgrid=robotgrid,
            targets=(x, y),
            assignments=results['assignments'],
            lambda_val=lambda_,
            strategy=strategy,
            output_dir=output_dir
        )
        results['plot_file'] = plot_file
    
    return results


def plot_assignment_results(robotgrid, targets, assignments, lambda_val, strategy, output_dir):
    """
    Plot targets and robots showing assignment status.
    
    Args:
        robotgrid: kaiju RobotGrid instance
        targets: Tuple of (x, y) target coordinates
        assignments: Dictionary of robot_id -> target_id assignments
        lambda_val: Lambda value for this experiment
        strategy: Strategy name ('greedy' or 'cp')
        output_dir: Output directory for saving plots
    """
    x, y = targets
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Separate targets into assigned and unassigned
    assigned_targets_x = []
    assigned_targets_y = []
    unassigned_targets_x = []
    unassigned_targets_y = []
    
    # Get all assigned target IDs
    assigned_target_ids = set(assignments.values())
    
    for i, (target_x, target_y) in enumerate(zip(x, y)):
        if i in assigned_target_ids:
            assigned_targets_x.append(target_x)
            assigned_targets_y.append(target_y)
        else:
            unassigned_targets_x.append(target_x)
            unassigned_targets_y.append(target_y)
    
    # Plot assigned targets in blue
    if assigned_targets_x:
        ax.scatter(assigned_targets_x, assigned_targets_y, s=60, c='blue', alpha=0.8, 
                  label=f'Assigned Targets ({len(assigned_targets_x)})', marker='^', 
                  edgecolors='darkblue', linewidth=1)
    
    # Plot unassigned targets in gray
    if unassigned_targets_x:
        ax.scatter(unassigned_targets_x, unassigned_targets_y, s=60, c='gray', alpha=0.6, 
                  label=f'Unassigned Targets ({len(unassigned_targets_x)})', marker='^', 
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
    stats_text += f'Assigned Targets: {len(assigned_targets_x)}\n'
    stats_text += f'Unassigned Targets: {len(unassigned_targets_x)}\n'
    stats_text += f'Total Robots: {robotgrid.nRobots}\n'
    stats_text += f'Assigned Robots: {len(assignments)}\n'
    stats_text += f'Unassigned Robots: {len(unassigned_robots_x)}\n'
    stats_text += f'Robot Assignment Rate: {len(assignments)/robotgrid.nRobots*100:.1f}%\n'
    stats_text += f'Target Assignment Rate: {len(assigned_targets_x)/len(x)*100:.1f}%'
    
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

def parse_args():
    parser = argparse.ArgumentParser(description='Run robot assignment experiments')
    parser.add_argument('--lambda-min', type=float, default=0.0001,
                       help='Minimum lambda value (default: 0.0001)')
    parser.add_argument('--lambda-max', type=float, default=0.01,
                       help='Maximum lambda value (default: 0.002)')
    parser.add_argument('--num-points', type=int, default=10,
                       help='Number of lambda values to test (default: 20)')
    parser.add_argument('--num-priorities', type=int, default=2,
                       help='Number of priority levels (default: 2)')
    parser.add_argument('--scientific-percentage', type=float, default=0.0,
                       help='Percentage of targets to mark as scientific (0.0 to 1.0, default: 0.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    parser.add_argument('--algorithm', choices=['greedy', 'cp', 'all'], default='all',
                       help='Algorithm to run (default: all)')
    
    # CP solver parameters
    parser.add_argument('--time-limit', type=float, default=30.0,
                       help='CP solver time limit in seconds (default: 30.0)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of worker threads for CP solver (default: 0 = use all cores)')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='output_multi_priority',
                       help='Output directory (default: output)')
    parser.add_argument('--plot-assignments', action='store_true',
                       help='Generate assignment visualization plots')
    
    # Scientific targets configuration
    
    
    # Priority configuration
    
    return parser.parse_args()


if __name__ == '__main__':
    """Main function with argument parsing."""
    args = parse_args()
    
    print("=== Robot Assignment Experiment Runner ===")
    print(f"Lambda range: {args.lambda_min:.5f} to {args.lambda_max:.5f}")
    print(f"Number of priority levels: {args.num_priorities}")
    print(f"Number of points: {args.num_points}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Time limit: {args.time_limit} seconds")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed}")
    print(f"Target filtering: ENABLED (unreachable targets removed)")
    print(f"Scientific targets: {args.scientific_percentage*100:.1f}% of targets")
    if args.plot_assignments:
        print(f"Assignment plots: ENABLED")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set up random number generator
    rng = np.random.default_rng(args.seed)
    
    # Generate lambda values
    lambda_values = np.linspace(args.lambda_min, args.lambda_max, args.num_points)
    
    # Determine which algorithms to run
    algorithms = []
    if args.algorithm in ['greedy', 'all']:
        algorithms.append('greedy')
    if args.algorithm in ['greedy_furthest', 'all']:
        algorithms.append('greedy_furthest')
    if args.algorithm in ['cp', 'all']:
        algorithms.append('cp')
    
    # Run experiments
    all_results = []
    
    for lambda_val in lambda_values:
        print(f"\nRunning experiments for λ = {lambda_val:.5f}")
        
        for algorithm in algorithms:
            print(f"  Running {algorithm} algorithm...")
            try:
                results = run_experiment(lambda_val, algorithm, rng, args.time_limit, 
                                       plot_results=args.plot_assignments, output_dir=args.output, 
                                       num_priorities=args.num_priorities, scientific_percentage=args.scientific_percentage)
                all_results.append(results)
                print(f"    Assigned: {results['num_assigned']} targets")
                print(f"    Time: {results['elapsed_time']:.3f} seconds")
                print(f"    Status: {results['status']}")
                if args.plot_assignments and 'plot_file' in results:
                    print(f"    Plot saved: {results['plot_file']}")
            except Exception as e:
                print(f"    Error: {str(e)}")
                # Add error result
                all_results.append({
                    'lambda': lambda_val,
                    'strategy': algorithm,
                    'num_assigned': 0,
                    'elapsed_time': 0.0,
                    'status': f'error: {str(e)}',
                    'num_targets': 0
                })
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results to CSV
    csv_path = os.path.join(args.output, 'experiment_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate plots
    if len(df) > 0:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Number of assigned targets vs lambda
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
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Total assigned priority vs lambda
        for algorithm in algorithms:
            alg_data = df[df['strategy'] == algorithm]
            if len(alg_data) > 0:
                ax3.plot(alg_data['lambda'], alg_data['total_assigned_priority'], 
                        marker='^', label=f'{algorithm.upper()} Algorithm', linewidth=2)
        
        ax3.set_xlabel('Lambda (λ)')
        ax3.set_ylabel('Total Assigned Priority')
        ax3.set_title('Total Assigned Priority vs Target Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        plot_path = os.path.join(args.output, 'comparison_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_path}")
        
        # Show summary statistics
        print(f"\n=== Summary Statistics ===")
        for algorithm in algorithms:
            alg_data = df[df['strategy'] == algorithm]
            if len(alg_data) > 0:
                print(f"\n{algorithm.upper()} Algorithm:")
                print(f"  Average assigned targets: {alg_data['num_assigned'].mean():.1f}")
                print(f"  Average computation time: {alg_data['elapsed_time'].mean():.3f} seconds")
                print(f"  Average total assigned priority: {alg_data['total_assigned_priority'].mean():.1f}")
                print(f"  Max assigned targets: {alg_data['num_assigned'].max()}")
                print(f"  Max computation time: {alg_data['elapsed_time'].max():.3f} seconds")
                print(f"  Max total assigned priority: {alg_data['total_assigned_priority'].max():.1f}")
    
    print(f"\nExperiment completed!")