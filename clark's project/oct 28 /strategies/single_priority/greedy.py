from ..base_strategy import Strategy
import kaiju
import numpy as np
from typing import Dict, Tuple

class GreedyClosestStrategy(Strategy):

    def __init__(self, 
                 robotgrid: kaiju.robotGrid, 
                 targets: list[float], 
                 seed: int,
                 is_scientific: list[bool] = None):
        super().__init__(robotgrid=robotgrid, targets=targets, seed=seed, is_scientific=is_scientific)
        # Precompute mapping from target ID to scientific status to avoid repeated list/index work
        target_ids = list(self.robotgrid.targetDict.keys())
        if is_scientific is not None:
            n = min(len(is_scientific), len(target_ids))
            self._science_by_tid = {tid: bool(is_scientific[i]) for i, tid in enumerate(target_ids[:n])}
        else:
            self._science_by_tid = {}
    
    def optimize(self) -> Tuple[Dict[int, int], str]:
        """
        Randomized greedy assignment algorithm with scientific target prioritization.
        Scientific targets are assigned first, then non-scientific targets.
        Each robot picks its closest unassigned valid target in random order.
        """
        try:
            import heapq
            # Set up random number generator for reproducible shuffling
            rng = np.random.default_rng(self.seed)
            
            # Get list of robot IDs and shuffle them
            robot_ids = list(self.robotgrid.robotDict.keys())
            rng.shuffle(robot_ids)
            
            assignments = {}
            assigned_targets = set()
            
            # Bind hot lookups to locals
            robotgrid = self.robotgrid
            robotDict = robotgrid.robotDict
            targetDict = robotgrid.targetDict
            wouldCollide = robotgrid.wouldCollideWithAssigned
            assign = robotgrid.assignRobot2Target
            science_by_tid = self._science_by_tid
            
            # For each robot in random order, find closest unassigned valid target
            for robot_id in robot_ids:
                robot = robotDict[robot_id]
                
                # Skip offline robots
                if robot.isOffline:
                    continue
                
                # Build two min-heaps keyed by squared distance
                sci_heap = []
                non_sci_heap = []
                
                rx, ry = robot.xPos, robot.yPos
                push = heapq.heappush
                
                for target_id in robot.validTargetIDs:
                    # Skip if target already assigned
                    if target_id in assigned_targets:
                        continue
                    
                    target = targetDict[target_id]
                    dx = target.xWok - rx
                    dy = target.yWok - ry
                    dist2 = dx * dx + dy * dy
                    if science_by_tid.get(target_id, False):
                        push(sci_heap, (dist2, target_id))
                    else:
                        push(non_sci_heap, (dist2, target_id))
                
                # Pop best from scientific first, then non-scientific
                for heap in (sci_heap, non_sci_heap):
                    while heap:
                        _, tid = heapq.heappop(heap)
                        collided, _, _, _ = wouldCollide(robot_id, tid)
                        if not collided:
                            assignments[robot_id] = tid
                            assigned_targets.add(tid)
                            assign(robot_id, tid)
                            break
                    else:
                        continue
                    break
            
            return assignments, "success"
            
        except Exception as e:
            return {}, f"error: {str(e)}"


class GreedyFurthestStrategy(Strategy):

    def __init__(self, 
                 robotgrid: kaiju.robotGrid, 
                 targets: list[float], 
                 seed: int,
                 is_scientific: list[bool] = None):
        super().__init__(robotgrid=robotgrid, targets=targets, seed=seed, is_scientific=is_scientific)
        # Precompute mapping from target ID to scientific status to avoid repeated list/index work
        target_ids = list(self.robotgrid.targetDict.keys())
        if is_scientific is not None:
            n = min(len(is_scientific), len(target_ids))
            self._science_by_tid = {tid: bool(is_scientific[i]) for i, tid in enumerate(target_ids[:n])}
        else:
            self._science_by_tid = {}
    
    def optimize(self) -> Tuple[Dict[int, int], str]:
        """
        Randomized greedy assignment algorithm with scientific target prioritization.
        Scientific targets are assigned first, then non-scientific targets.
        Each robot picks its furthest unassigned valid target in random order.
        """
        try:
            import heapq
            # Set up random number generator for reproducible shuffling
            rng = np.random.default_rng(self.seed)
            
            # Get list of robot IDs and shuffle them
            robot_ids = list(self.robotgrid.robotDict.keys())
            rng.shuffle(robot_ids)
            
            assignments = {}
            assigned_targets = set()
            
            # Bind hot lookups to locals
            robotgrid = self.robotgrid
            robotDict = robotgrid.robotDict
            targetDict = robotgrid.targetDict
            wouldCollide = robotgrid.wouldCollideWithAssigned
            assign = robotgrid.assignRobot2Target
            science_by_tid = self._science_by_tid
            
            # For each robot in random order, find furthest unassigned valid target
            for robot_id in robot_ids:
                robot = robotDict[robot_id]
                
                # Skip offline robots
                if robot.isOffline:
                    continue
                
                # Build two heaps using negative squared distance so that smallest pops = furthest
                sci_heap = []
                non_sci_heap = []
                
                rx, ry = robot.xPos, robot.yPos
                push = heapq.heappush
                
                for target_id in robot.validTargetIDs:
                    # Skip if target already assigned
                    if target_id in assigned_targets:
                        continue
                    
                    target = targetDict[target_id]
                    dx = target.xWok - rx
                    dy = target.yWok - ry
                    neg_dist2 = -(dx * dx + dy * dy)
                    if science_by_tid.get(target_id, False):
                        push(sci_heap, (neg_dist2, target_id))
                    else:
                        push(non_sci_heap, (neg_dist2, target_id))
                
                # Pop furthest from scientific first, then non-scientific
                for heap in (sci_heap, non_sci_heap):
                    while heap:
                        _, tid = heapq.heappop(heap)
                        collided, _, _, _ = wouldCollide(robot_id, tid)
                        if not collided:
                            assignments[robot_id] = tid
                            assigned_targets.add(tid)
                            assign(robot_id, tid)
                            break
                    else:
                        continue
                    break
            
            return assignments, "success"
            
        except Exception as e:
            return {}, f"error: {str(e)}"