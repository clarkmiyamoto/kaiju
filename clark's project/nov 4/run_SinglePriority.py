import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from config_robots import load_apo_grid
from targets import sample_ppp, generate_targets
from strategies.single_priority.greedy import GreedyClosestStrategy, GreedyFurthestStrategy
from strategies.single_priority.cp_sat import ORToolsStrategy
from strategies.single_priority.cp_warmstart import CPWarmstartStrategy
from target_utils import add_targets_to_robotgrid, clear_robotgrid_assignments
from experiment_plots import plot_assignment_results, create_comparison_plots, print_summary_statistics

def run_greedy(robotgrid, targets, seed, is_scientific=None):
    strategy = GreedyClosestStrategy(robotgrid=robotgrid, targets=targets, seed=seed, is_scientific=is_scientific)
    strategy.run_optimizer()
    return strategy.get_assignment_summary()

def run_greedy_furthest(robotgrid, targets, seed, is_scientific=None):
    strategy = GreedyFurthestStrategy(robotgrid=robotgrid, targets=targets, seed=seed, is_scientific=is_scientific)
    strategy.run_optimizer()
    return strategy.get_assignment_summary()

def run_ortools(robotgrid, targets, seed, time_limit, is_scientific=None):
    strategy = ORToolsStrategy(robotgrid=robotgrid, targets=targets, seed=seed, time_limit=time_limit, is_scientific=is_scientific)
    strategy.run_optimizer()
    return strategy.get_assignment_summary()

def run_cp_warmstart(robotgrid, targets, seed, time_limit, is_scientific=None, use_closest_greedy=True):
    strategy = CPWarmstartStrategy(robotgrid=robotgrid, targets=targets, seed=seed, time_limit=time_limit, is_scientific=is_scientific, use_closest_greedy=use_closest_greedy)
    strategy.run_optimizer()
    return strategy.get_assignment_summary()


def parse_args():
    parser = argparse.ArgumentParser(description='Run robot assignment experiments')
    parser.add_argument('--lambda-min', type=float, default=0.0001,
                       help='Minimum lambda value (default: 0.0001)')
    parser.add_argument('--lambda-max', type=float, default=0.01,
                       help='Maximum lambda value (default: 0.002)')
    parser.add_argument('--num-points', type=int, default=10,
                       help='Number of lambda values to test (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    parser.add_argument('--algorithm', nargs='+', choices=['greedy', 'greedy_furthest', 'cp', 'cp_warmstart', 'all'], default=['all'],
                       help='Algorithm(s) to run (default: all). Can specify multiple: --algorithm greedy cp_warmstart')
    
    # CP solver parameters
    parser.add_argument('--time-limit', type=float, default=30.0,
                       help='CP solver time limit in seconds (default: 30.0)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of worker threads for CP solver (default: 0 = use all cores)')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='output_single_priority',
                       help='Output directory (default: output)')
    parser.add_argument('--plot-assignments', action='store_true',
                       help='Generate assignment visualization plots')
    
    # Scientific targets parameters
    parser.add_argument('--scientific-percentage', type=float, default=0.0,
                       help='Percentage of targets to mark as scientific (0.0 to 1.0, default: 0.0)')
    
    return parser.parse_args()

if __name__ == '__main__':
    """Main function with argument parsing."""
    args = parse_args()
    
    print("=== Robot Assignment Experiment Runner ===")
    print(f"Lambda range: {args.lambda_min:.5f} to {args.lambda_max:.5f}")
    print(f"Number of points: {args.num_points}")
    print(f"Algorithm(s): {args.algorithm}")
    print(f"Time limit: {args.time_limit} seconds")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed}")
    print(f"Scientific targets percentage: {args.scientific_percentage*100:.1f}%")
    print(f"Target filtering: ENABLED (unreachable targets removed)")
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
    
    # If 'all' is specified, run all algorithms
    if 'all' in args.algorithm:
        algorithms = ['greedy', 'greedy_furthest', 'cp', 'cp_warmstart']
    else:
        # Otherwise, run only the specified algorithms
        algorithms = args.algorithm
    
    # Run experiments
    all_results = []
    
    for lambda_val in lambda_values:
        print(f"\nRunning experiments for λ = {lambda_val:.5f}")
        
        # Generate robot grid and targets once per lambda value to ensure consistency
        robotgrid = load_apo_grid(seed=0)
        
        # Generate targets with filtering
        x, y, _, fiber_types, is_scientific = generate_targets(
            lam_per_area=lambda_val,
            xlim=(-300, 300),
            ylim=(-300, 300),
            rng=rng,
            enable_fiber_types=True,
            robotgrid=robotgrid,
            scientific_percentage=args.scientific_percentage
        )
        
        # Use the same seed for all strategies to ensure consistency
        strategy_seed = rng.integers(0, 2**31)
        
        for algorithm in algorithms:
            print(f"  Running {algorithm} algorithm...")
            try:
                # Clear any previous assignments
                clear_robotgrid_assignments(robotgrid)
                
                # Run the specified strategy with the same setup
                if algorithm == 'greedy':
                    results = run_greedy(robotgrid, (x, y), strategy_seed, is_scientific)
                elif algorithm == 'greedy_furthest':
                    results = run_greedy_furthest(robotgrid, (x, y), strategy_seed, is_scientific)
                elif algorithm == 'cp':
                    results = run_ortools(robotgrid, (x, y), strategy_seed, args.time_limit, is_scientific)
                elif algorithm == 'cp_warmstart':
                    results = run_cp_warmstart(robotgrid, (x, y), strategy_seed, args.time_limit, is_scientific, use_closest_greedy=True)
                else:
                    raise ValueError(f"Unknown strategy: {algorithm}")
                
                # Add experiment metadata
                results['lambda'] = lambda_val
                results['strategy'] = algorithm
                results['num_targets'] = len(x)
                
                # Add scientific targets information if available
                if is_scientific is not None:
                    results['num_scientific_targets'] = np.sum(is_scientific)
                    results['num_non_scientific_targets'] = len(x) - np.sum(is_scientific)
                    results['scientific_percentage'] = args.scientific_percentage
                    
                    # Calculate assigned scientific and non-scientific targets
                    assigned_scientific = 0
                    assigned_non_scientific = 0
                    assigned_target_ids = set(results['assignments'].values())
                    
                    # Target IDs correspond directly to indices in the original targets array
                    # (see target_utils.py line 47: target_id = i)
                    for target_id in assigned_target_ids:
                        if target_id < len(is_scientific) and is_scientific[target_id]:
                            assigned_scientific += 1
                        else:
                            assigned_non_scientific += 1
                    
                    results['num_assigned_scientific'] = assigned_scientific
                    results['num_assigned_non_scientific'] = assigned_non_scientific
                else:
                    results['num_scientific_targets'] = 0
                    results['num_non_scientific_targets'] = len(x)
                    results['scientific_percentage'] = 0.0
                    results['num_assigned_scientific'] = 0
                    results['num_assigned_non_scientific'] = results['num_assigned']
                
                # Generate assignment plot if requested
                if args.plot_assignments and 'assignments' in results:
                    plot_file = plot_assignment_results(
                        robotgrid=robotgrid,
                        targets=(x, y),
                        assignments=results['assignments'],
                        lambda_val=lambda_val,
                        strategy=algorithm,
                        output_dir=args.output,
                        is_scientific=is_scientific
                    )
                    results['plot_file'] = plot_file
                
                all_results.append(results)
                print(f"    Assigned: {results['num_assigned']} targets")
                print(f"    Scientific targets: {results['num_scientific_targets']}")
                print(f"    Non-scientific targets: {results['num_non_scientific_targets']}")
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
                    'num_targets': len(x) if 'x' in locals() else 0,
                    'num_scientific_targets': np.sum(is_scientific) if is_scientific is not None else 0,
                    'num_non_scientific_targets': len(x) - np.sum(is_scientific) if is_scientific is not None else len(x) if 'x' in locals() else 0,
                    'scientific_percentage': args.scientific_percentage,
                    'num_assigned_scientific': 0,
                    'num_assigned_non_scientific': 0
                })
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results to CSV
    csv_path = os.path.join(args.output, 'experiment_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate plots
    if len(df) > 0:
        plot_path = create_comparison_plots(df, algorithms, args.scientific_percentage, args.output)
        print(f"Plots saved to: {plot_path}")
        
        # Show summary statistics
        print_summary_statistics(df, algorithms)
    
    print(f"\nExperiment completed!")