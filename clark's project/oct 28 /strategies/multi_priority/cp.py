import kaiju
from ortools.sat.python import cp_model
from ..base_strategy import Strategy
from typing import Dict, Tuple, Any, List
import numpy as np

class ORToolsStrictPriority(Strategy):
    def __init__(self, 
                 robotgrid: kaiju.robotGrid, 
                 targets: list[float], 
                 priorities: list[float],
                 seed: int,
                 time_limit: float,
                 num_workers: int = 0):
        super().__init__(robotgrid=robotgrid, targets=targets, seed=seed)
        
        # Validate priorities are positive integers
        if not all(isinstance(p, (int, float)) and p > 0 for p in priorities):
            raise ValueError("Priorities must be positive numbers")
        
        self.priorities = priorities
        self.time_limit = time_limit
        self.num_workers = num_workers
        
        # Create solver instance (will be reused for each priority level)
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = time_limit
        self.solver.parameters.num_workers = num_workers
    
    def get_total_assigned_priority(self) -> float:
        """Get total priority of assigned targets"""
        total_priority = 0.0
        for robot_id, target_id in self.assignments.items():
            target = self.robotgrid.targetDict[target_id]
            total_priority += target.priority
        return total_priority
    
    def get_assignment_summary(self) -> Dict[str, Any]:
        """Get summary of assignment results including priority information"""
        summary = super().get_assignment_summary()
        summary['total_assigned_priority'] = self.get_total_assigned_priority()
        return summary

    def optimize(self) -> Tuple[Dict[int, int], str]:
        """
        Solve the hierarchical CP-SAT assignment problem.
        Run separate CP models for each priority level in descending order.
        """
        try:
            # Get target rsids from robotgrid
            target_rsids = list(self.robotgrid.targetDict.keys())
            
            # Get unique priority levels in descending order (highest first)
            unique_priorities = sorted(set(self.priorities), reverse=True)
            
            all_assignments = {}
            assigned_targets = set()
            assigned_robots = set()
            overall_status = "optimal"
            
            print(f"Running hierarchical CP with {len(unique_priorities)} priority levels: {unique_priorities}")
            
            # Process each priority level
            for priority_level in unique_priorities:
                print(f"Processing priority level {priority_level}...")
                
                # Get targets at this priority level
                priority_targets = [rsid for i, rsid in enumerate(target_rsids) 
                                 if i < len(self.priorities) and self.priorities[i] == priority_level]
                
                if not priority_targets:
                    print(f"No targets at priority level {priority_level}, skipping...")
                    continue
                
                # Filter out already assigned targets
                available_targets = [t for t in priority_targets if t not in assigned_targets]
                
                if not available_targets:
                    print(f"All targets at priority level {priority_level} already assigned, skipping...")
                    continue
                
                # Get available robots (not yet assigned)
                available_robots = [robot_id for robot_id in self.robotgrid.robotDict.keys() 
                                  if robot_id not in assigned_robots]
                
                if not available_robots:
                    print(f"No available robots for priority level {priority_level}")
                    break
                
                print(f"  Available targets: {len(available_targets)}, Available robots: {len(available_robots)}")
                
                # Run CP model for this priority level
                level_assignments, level_status = self.run_priority_level_cp(
                    available_targets, available_robots, priority_level
                )
                
                # Update tracking sets
                for robot_id, target_id in level_assignments.items():
                    all_assignments[robot_id] = target_id
                    assigned_targets.add(target_id)
                    assigned_robots.add(robot_id)
                    # Actually assign the robot to target in robotgrid
                    self.robotgrid.assignRobot2Target(robot_id, target_id)
                
                print(f"  Assigned {len(level_assignments)} targets at priority {priority_level}")
                
                # Update overall status
                if level_status not in ["optimal", "feasible"]:
                    overall_status = level_status
                
            print(f"Total assignments: {len(all_assignments)}")
            return all_assignments, overall_status
                
        except Exception as e:
            return {}, f"error: {str(e)}"

    def run_priority_level_cp(self, available_targets: List[int], available_robots: List[int], priority_level: float) -> Tuple[Dict[int, int], str]:
        """
        Run CP-SAT for a single priority level.
        
        Args:
            available_targets: List of target rsids available at this priority level
            available_robots: List of robot ids available at this priority level
            priority_level: The priority level being processed
            
        Returns:
            Tuple of (assignments_dict, status_string)
        """
        try:
            # Create a new model for this priority level
            model = cp_model.CpModel()
            
            # Setup variables and constraints for this priority level
            wwrt, wwtr, ww_list = self.setup_priority_level_constraints(
                model, available_targets, available_robots
            )
            
            # Add constraints to ensure ALL targets at this priority level are assigned
            # (since these are "scientific targets that must be guaranteed")
            self.add_guaranteed_assignment_constraints(model, available_targets, wwtr)
            
            # Set objective to maximize number of assignments at this priority level
            objective_terms = []
            for robot_id in wwrt:
                for target_id in wwrt[robot_id]:
                    objective_terms.append(wwrt[robot_id][target_id])
            
            if objective_terms:
                model.Maximize(cp_model.LinearExpr.Sum(objective_terms))
            
            # Solve the model
            status = self.solver.Solve(model)
            
            # Extract assignments
            assignments = {}
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                for robot_id in wwrt:
                    for target_id in wwrt[robot_id]:
                        if self.solver.Value(wwrt[robot_id][target_id]) == 1:
                            assignments[robot_id] = target_id
                
                status_str = "optimal" if status == cp_model.OPTIMAL else "feasible"
                return assignments, status_str
            else:
                return {}, f"solver_failed: {status}"
                
        except Exception as e:
            return {}, f"error: {str(e)}"

    def setup_priority_level_constraints(self, model: cp_model.CpModel, available_targets: List[int], available_robots: List[int]) -> Tuple[Dict, Dict, List]:
        """
        Setup CP-SAT constraints for a single priority level.
        
        Args:
            model: CP model to add constraints to
            available_targets: List of target rsids available at this priority level
            available_robots: List of robot ids available at this priority level
            
        Returns:
            Tuple of (wwrt, wwtr, ww_list) variable dictionaries
        """
        # Add variables; one for each robot-target pair
        wwrt = dict()  # robotID -> rsid -> BoolVar
        wwtr = dict()  # rsid -> robotID -> BoolVar
        
        for robot_id in available_robots:
            r = self.robotgrid.robotDict[robot_id]
            for rsid in interlist(r.validTargetIDs, available_targets):
                name = f'ww[{robot_id}][{rsid}]'
                if rsid not in wwtr:
                    wwtr[rsid] = dict()
                if robot_id not in wwrt:
                    wwrt[robot_id] = dict()
                wwrt[robot_id][rsid] = model.NewBoolVar(name)
                wwtr[rsid][robot_id] = wwrt[robot_id][rsid]

        # List of all robot-target pairs
        ww_list = [wwrt[y][x] for y in wwrt for x in wwrt[y]]

        # Constrain to use only one target per robot
        for robot_id in wwrt:
            rlist = [wwrt[robot_id][c] for c in wwrt[robot_id]]
            if rlist:  # Only add constraint if robot has valid targets
                robot_sum = cp_model.LinearExpr.Sum(rlist)
                model.Add(robot_sum <= 1)

        # Constrain to use only one robot per target
        for rsid in wwtr:
            tlist = [wwtr[rsid][r] for r in wwtr[rsid]]
            if tlist:  # Only add constraint if target has valid robots
                target_sum = cp_model.LinearExpr.Sum(tlist)
                model.Add(target_sum <= 1)

        # Add collision constraints
        self.add_collision_constraints(model, wwrt, available_robots)

        return wwrt, wwtr, ww_list

    def add_guaranteed_assignment_constraints(self, model: cp_model.CpModel, available_targets: List[int], wwtr: Dict):
        """
        Add constraints to ensure ALL targets at this priority level are assigned.
        These are "scientific targets that must be guaranteed".
        """
        for rsid in available_targets:
            if rsid in wwtr:
                # Sum of all robots assigned to this target must be >= 1
                target_assignments = [wwtr[rsid][robot_id] for robot_id in wwtr[rsid]]
                if target_assignments:
                    model.Add(cp_model.LinearExpr.Sum(target_assignments) >= 1)

    def add_collision_constraints(self, model: cp_model.CpModel, wwrt: Dict, available_robots: List[int]):
        """
        Add collision constraints for the available robots.
        """
        # Find potential collisions among available robots
        collisions = []
        for robot_id1 in available_robots:
            r1 = self.robotgrid.robotDict[robot_id1]
            for rsid1 in r1.validTargetIDs:
                self.robotgrid.assignRobot2Target(robot_id1, rsid1)
                for robot_id2 in r1.robotNeighbors:
                    if robot_id2 in available_robots:  # Only check collisions with available robots
                        r2 = self.robotgrid.robotDict[robot_id2]
                        for rsid2 in r2.validTargetIDs:
                            if rsid1 != rsid2:
                                self.robotgrid.assignRobot2Target(robot_id2, rsid2)
                                if self.robotgrid.isCollidedWithAssigned(robot_id1)[0]:
                                    collisions.append((robot_id1, rsid1, robot_id2, rsid2))
                                self.robotgrid.homeRobot(robot_id2)
                self.robotgrid.homeRobot(robot_id1)

        # Add constraints that collisions can't occur
        for robot_id1, rsid1, robot_id2, rsid2 in collisions:
            if (robot_id1 in wwrt and rsid1 in wwrt[robot_id1] and 
                robot_id2 in wwrt and rsid2 in wwrt[robot_id2]):
                ww1 = wwrt[robot_id1][rsid1]
                ww2 = wwrt[robot_id2][rsid2]
                collision_sum = cp_model.LinearExpr.Sum([ww1, ww2])
                model.Add(collision_sum <= 1)


def interlist(list1, list2):
    return(list(set(list1).intersection(list2)))