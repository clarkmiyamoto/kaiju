# Greedy vs CP Algorithm Comparison

This project compares the performance of a **Greedy Algorithm** and a **Constraint Programming (CP)** algorithm for robot-target assignment in telescope fiber positioning systems.

## Overview

The comparison measures:
- **Number of assigned targets** vs target density (λ)
- **Computation time** vs target density (λ)

Targets are generated using a Poisson Point Process (PPP) with varying density parameters.

## Algorithms

### Greedy Algorithm
- **Strategy**: Randomized greedy assignment
- **Process**: 
  1. Shuffle robot order using random seed
  2. Each robot picks its closest unassigned valid target
  3. Check for collisions with already assigned robots
  4. Assign if collision-free

### CP Algorithm (OR-Tools)
- **Strategy**: Constraint Programming using CP-SAT solver
- **Process**:
  1. Set up binary variables for each robot-target pair
  2. Add constraints: one target per robot, one robot per target
  3. Add collision constraints
  4. Maximize number of assignments
  5. Solve using OR-Tools CP-SAT

## Usage

### Run Full Experiment
```bash
python run.py --algorithm both --num-points 10 --lambda-min 0.0001 --lambda-max 0.01
```

### Run Only Greedy Algorithm
```bash
python run.py --algorithm greedy --num-points 5
```

### Run Only CP Algorithm
```bash
python run.py --algorithm cp --time-limit 60 --num-points 5
```

### Command Line Options
- `--lambda-min`: Minimum lambda value (default: 0.0001)
- `--lambda-max`: Maximum lambda value (default: 0.01)
- `--num-points`: Number of lambda values to test (default: 10)
- `--algorithm`: Algorithm to run - 'greedy', 'cp', or 'both' (default: 'both')
- `--time-limit`: CP solver time limit in seconds (default: 30.0)
- `--output`: Output directory (default: 'output')
- `--seed`: Random seed (default: 42)

### Test Implementation
```bash
python test_implementation.py
```

## Output

The experiment generates:

1. **CSV Results**: `output/experiment_results.csv`
   - Columns: lambda, strategy, num_assigned, elapsed_time, status, num_targets

2. **Comparison Plots**: `output/comparison_plots.png`
   - Left plot: Number of assigned targets vs λ
   - Right plot: Computation time vs λ

3. **Console Output**: Summary statistics and progress

## File Structure

- `base_strategy.py`: Abstract base class for strategies
- `greedy.py`: Greedy algorithm implementation
- `cp.py`: CP algorithm implementation using OR-Tools
- `targets.py`: Target generation using Poisson Point Process
- `config_robots.py`: Robot grid configuration (APO telescope)
- `target_utils.py`: Utilities for integrating targets with robotgrid
- `run.py`: Main experiment runner
- `test_implementation.py`: Test script

## Dependencies

- `kaiju`: Telescope robot simulation library
- `ortools`: Google OR-Tools for constraint programming
- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `matplotlib`: Plotting

## Example Results

Typical results show:
- **Greedy**: Faster computation, fewer assignments at high density
- **CP**: Slower computation, more assignments (optimal solutions)

The trade-off between solution quality and computation time varies with target density.
