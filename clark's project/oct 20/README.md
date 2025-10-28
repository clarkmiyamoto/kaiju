# Robot Assignment Codebase

This codebase implements and compares different algorithms for assigning telescope robots to astronomical targets, with the goal of maximizing total priority while avoiding collisions.

## Overview

The codebase provides two main assignment algorithms:
- **Greedy Algorithm**: Assigns highest priority targets first to the nearest available robot
- **Constraint Programming (CP)**: Uses Google OR-Tools CP-SAT solver for optimal assignment

## File Structure

### Core Modules

- **`config_robots.py`** - Robot configuration and loading
  - `load_apo_grid()` - Load Apache Point Observatory robot configuration
  - `get_robot_stats()` - Get robot statistics by fiber type
  - `print_robot_summary()` - Print configuration summary

- **`targets.py`** - Target generation using Poisson Point Process
  - `generate_targets()` - Generate targets with priorities and fiber types
  - `sample_ppp()` - Sample from Poisson Point Process
  - `generate_targets_with_custom_priorities()` - Custom priority distributions

- **`collision_matrix.py`** - Collision detection and matrix building
  - `calculate_robot_reach()` - Calculate which targets each robot can reach
  - `build_collision_matrix()` - Build collision matrix for constraint programming
  - `analyze_collision_matrix()` - Analyze collision statistics

- **`greedy_solver.py`** - Greedy assignment algorithm
  - `greedy_assignment()` - Basic greedy assignment
  - `greedy_assignment_with_collision_check()` - Greedy with collision avoidance

- **`cp_solver.py`** - Constraint programming solver
  - `cp_assignment()` - Full CP solver with collision constraints
  - `cp_assignment_simple()` - Simplified CP solver (faster)

- **`runner.py`** - Master experiment runner with command-line interface

## Installation

```bash
pip install ortools kaiju matplotlib numpy pandas
```

## Usage

### Basic Usage

Run experiments comparing both algorithms:

```bash
python runner.py --algorithm both --num-points 10
```

### Advanced Usage

```bash
# Run only greedy algorithm with custom parameters
python runner.py --algorithm greedy --lambda-min 0.0001 --lambda-max 0.005 --num-points 15

# Run CP algorithm with longer time limit
python runner.py --algorithm cp --time-limit 600 --collision-buffer 75

# Run with custom output directory
python runner.py --output results --seed 123 --verbose
```

### Command Line Options

- `--lambda-min` - Minimum target density (default: 0.0001)
- `--lambda-max` - Maximum target density (default: 0.01)
- `--num-points` - Number of density values to test (default: 20)
- `--algorithm` - Algorithm to run: `greedy`, `cp`, or `both` (default: both)
- `--collision-buffer` - Collision distance threshold in mm (default: 50.0)
- `--time-limit` - CP solver time limit in seconds (default: 300.0)
- `--output` - Output directory for results (default: output)
- `--seed` - Random seed for reproducibility (default: 42)
- `--verbose` - Enable verbose output

### Programmatic Usage

```python
from config_robots import load_apo_grid
from targets import generate_targets
from greedy_solver import greedy_assignment
from cp_solver import cp_assignment_simple

# Load robot configuration
grid = load_apo_grid()

# Generate targets
x, y, priorities, fiber_types = generate_targets(
    lam_per_area=0.001,
    xlim=(-300, 300),
    ylim=(-300, 300),
    enable_fiber_types=True
)

# Run greedy assignment
greedy_priority, greedy_assignments = greedy_assignment(
    grid, x, y, priorities, fiber_types
)

# Run CP assignment
cp_priority, cp_assignments, solver_time = cp_assignment_simple(
    grid, x, y, priorities, fiber_types, time_limit=60.0
)

print(f"Greedy: {len(greedy_assignments)} assignments, priority: {greedy_priority}")
print(f"CP: {len(cp_assignments)} assignments, priority: {cp_priority}")
```

## Output

The runner generates:
- **Plots**: Comparison plots showing priority, assignments, and performance
- **CSV**: Detailed results for further analysis
- **Console**: Summary statistics and progress updates

## Algorithm Details

### Greedy Algorithm
1. Sort targets by priority (highest first)
2. For each target, find the nearest available robot with compatible fiber
3. Assign if no collision with previously assigned targets
4. Continue until all targets processed

### Constraint Programming
1. Create binary variables for each robot-target pair
2. Add constraints: one robot per target, one target per robot
3. Add collision avoidance constraints
4. Maximize total priority using CP-SAT solver

## Dependencies

- `ortools` - Google OR-Tools for constraint programming
- `kaiju` - Telescope robot simulation library
- `matplotlib` - Plotting
- `numpy` - Numerical computations
- `pandas` - Data analysis

## Examples

See the `__main__` sections in each module for example usage.