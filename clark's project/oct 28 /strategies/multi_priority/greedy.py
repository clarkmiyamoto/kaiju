from ..base_strategy import Strategy
import kaiju
import numpy as np
from typing import Dict, Tuple, Any

class Greedy(Strategy):

    def __init__(self, 
                 robotgrid: kaiju.robotGrid, 
                 targets: list[float], 
                 priorities: list[float],
                 is_scientific: list[bool],
                 seed: int,
                 ):
        super().__init__(robotgrid=robotgrid, targets=targets, seed=seed)
        self.priorities = priorities
        self.is_scientific = is_scientific
    
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
        Priority-based target-first greedy assignment algorithm with scientific target prioritization.
        
        Scientific targets are processed first within each priority level, then non-scientific targets.
        Within each group (scientific/non-scientific), targets are processed by priority (highest first), 
        then randomly within each priority.
        Each target is assigned to the closest available robot that doesn't cause a collision.
        """
        try:
            # Set up random number generator for reproducible shuffling
            rng = np.random.default_rng(self.seed)
            
            # Group targets by priority level and scientific status
            target_ids = list(self.robotgrid.targetDict.keys())
            priority_groups = {}
            
            for tid in target_ids:
                priority = self.robotgrid.targetDict[tid].priority
                if priority not in priority_groups:
                    priority_groups[priority] = {'scientific': [], 'non_scientific': []}
                
                # Check if target is scientific
                is_scientific = False
                if self.is_scientific is not None:
                    # Create mapping from target ID to scientific status
                    # Target IDs in robotgrid may not be sequential due to filtering
                    target_ids_list = list(self.robotgrid.targetDict.keys())
                    if tid in target_ids_list:
                        target_index = target_ids_list.index(tid)
                        if target_index < len(self.is_scientific):
                            is_scientific = self.is_scientific[target_index]
                
                if is_scientific:
                    priority_groups[priority]['scientific'].append(tid)
                else:
                    priority_groups[priority]['non_scientific'].append(tid)
            
            # Sort by priority (high to low) and shuffle within each priority group
            sorted_priorities = sorted(priority_groups.keys(), reverse=True)
            ordered_targets = []
            
            for priority in sorted_priorities:
                # First add scientific targets for this priority
                scientific_targets = priority_groups[priority]['scientific']
                rng.shuffle(scientific_targets)
                ordered_targets.extend(scientific_targets)
                
                # Then add non-scientific targets for this priority
                non_scientific_targets = priority_groups[priority]['non_scientific']
                rng.shuffle(non_scientific_targets)
                ordered_targets.extend(non_scientific_targets)
            
            # Track assignments
            assignments = {}
            assigned_robots = set()
            
            # For each target in priority order, find closest available robot
            for target_id in ordered_targets:
                target = self.robotgrid.targetDict[target_id]
                
                # Find all robots that can reach this target
                reachable_robots = []
                
                for robot_id, robot in self.robotgrid.robotDict.items():
                    # Skip offline or already assigned robots
                    if robot.isOffline or robot_id in assigned_robots:
                        continue
                    
                    # Check if this robot can reach the target
                    if target_id in robot.validTargetIDs:
                        # Calculate distance from robot to target
                        target_pos = np.array([target.xWok, target.yWok])
                        robot_pos = np.array([robot.xPos, robot.yPos])
                        distance = np.linalg.norm(target_pos - robot_pos)
                        reachable_robots.append((distance, robot_id))
                
                # Sort robots by distance (closest first)
                reachable_robots.sort(key=lambda x: x[0])
                
                # Try to assign target to closest robot without collision
                for distance, robot_id in reachable_robots:
                    # Check for collisions with already assigned robots
                    collision_check = self.robotgrid.wouldCollideWithAssigned(robot_id, target_id)
                    collided, fiducial_collided, gfa_collided, assigned_robots_colliding = collision_check
                    
                    if not collided:
                        # Safe to assign - found our robot!
                        assignments[robot_id] = target_id
                        assigned_robots.add(robot_id)
                        self.robotgrid.assignRobot2Target(robot_id, target_id)
                        break
                
                # If no robot could be assigned without collision, target remains unassigned
            
            return assignments, "success"
            
        except Exception as e:
            return {}, f"error: {str(e)}"
