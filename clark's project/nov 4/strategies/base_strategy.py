import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import kaiju

class Strategy:

    def __init__(self, 
                 robotgrid: kaiju.robotGrid, 
                 targets: list[float], 
                 seed: int,
                 is_scientific: list[bool] = None):
        self.robotgrid = robotgrid
        self.targets = targets
        self.seed = seed
        self.is_scientific = is_scientific
        self.elapsed_time: float = None # Units: seconds
        self.assignments: Dict[int, int] = {}  # robot_id -> target_id
        self.status: str = "unknown"

    def run_optimizer(self):
        '''
        The optimizer has been set, run the assignment problem!
        '''
        start_time = time.time()
        self.assignments, self.status = self.optimize()
        end_time = time.time()

        self.elapsed_time = end_time - start_time # in seconds

        return self.assignments

    def get_num_assigned(self) -> int:
        """Get number of assigned targets"""
        return len(self.assignments)

    def get_assignment_summary(self) -> Dict[str, Any]:
        """Get summary of assignment results"""
        return {
            'num_assigned': self.get_num_assigned(),
            'elapsed_time': self.elapsed_time,
            'status': self.status,
            'assignments': self.assignments
        }

    @abstractmethod
    def optimize(self) -> Tuple[Dict[int, int], str]:
        """Return (assignments_dict, status_string)"""
        pass







    