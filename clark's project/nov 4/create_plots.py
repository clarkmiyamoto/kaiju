#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Target-Robot Visualization Script

This script provides an easy way to create custom visualizations
of targets overlaid on the robot grid.
"""

import argparse
import numpy as np
from visualize_targets_robots import plot_targets_on_robots, create_comparison_plot


def main():
    """Main function with argument parsing for custom visualizations."""
    parser = argparse.ArgumentParser(description='Create target-robot overlay visualizations')
    
    # Visualization parameters
    parser.add_argument('--lambda', type=float, dest='lambda_val', default=0.001,
                       help='Target density parameter λ (default: 0.001)')
    parser.add_argument('--lambda-range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='Range of lambda values for comparison plot (e.g., 0.0001 0.005)')
    parser.add_argument('--num-lambdas', type=int, default=5,
                       help='Number of lambda values for comparison plot (default: 5)')
    
    # Plot parameters
    parser.add_argument('--xlim', nargs=2, type=float, default=[-300, 300],
                       metavar=('MIN', 'MAX'), help='X-coordinate limits (default: -300 300)')
    parser.add_argument('--ylim', nargs=2, type=float, default=[-300, 300],
                       metavar=('MIN', 'MAX'), help='Y-coordinate limits (default: -300 300)')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--comparison-only', action='store_true',
                       help='Only create comparison plot, skip individual plots')
    
    args = parser.parse_args()
    
    print("=== Target-Robot Visualization Generator ===")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed}")
    print(f"Coordinate limits: X={args.xlim}, Y={args.ylim}")
    
    # Create output directory
    import os
    os.makedirs(args.output, exist_ok=True)
    
    if args.lambda_range:
        # Create comparison plot with specified range
        print(f"\nCreating comparison plot with λ range: {args.lambda_range[0]} to {args.lambda_range[1]}")
        lambda_values = np.linspace(args.lambda_range[0], args.lambda_range[1], args.num_lambdas)
        
        comparison_plot = create_comparison_plot(
            lambda_values=lambda_values,
            xlim=tuple(args.xlim),
            ylim=tuple(args.ylim),
            seed=args.seed,
            output_dir=args.output
        )
        print(f"Comparison plot saved to: {comparison_plot}")
        
        if not args.comparison_only:
            print(f"\nCreating individual plots...")
            from visualize_targets_robots import create_multiple_density_plots
            individual_plots = create_multiple_density_plots(
                lambda_values=lambda_values,
                xlim=tuple(args.xlim),
                ylim=tuple(args.ylim),
                seed=args.seed,
                output_dir=args.output
            )
            print(f"Created {len(individual_plots)} individual plots")
    
    else:
        # Create single plot
        print(f"\nCreating single plot with λ = {args.lambda_val}")
        filepath = plot_targets_on_robots(
            lam_per_area=args.lambda_val,
            xlim=tuple(args.xlim),
            ylim=tuple(args.ylim),
            seed=args.seed,
            output_dir=args.output,
            title_suffix=f' (λ = {args.lambda_val:.5f})'
        )
        print(f"Plot saved to: {filepath}")
    
    print(f"\nVisualization complete! Check the '{args.output}' directory for results.")


if __name__ == "__main__":
    main()
