import kaiju
from ortools.sat.python import cp_model
from ..base_strategy import Strategy
from typing import Dict, Tuple

class ORToolsStrategy(Strategy):
    def __init__(self, 
                 robotgrid: kaiju.robotGrid, 
                 targets: list[float], 
                 seed: int,
                 time_limit: float,
                 num_workers: int = 0,
                 is_scientific: list[bool] = None):
        super().__init__(robotgrid=robotgrid, targets=targets, seed=seed, is_scientific=is_scientific)

        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = time_limit
        self.solver.parameters.num_workers = num_workers

    def optimize(self) -> Tuple[Dict[int, int], str]:
        """
        Solve the CP-SAT assignment problem and extract assignments.
        """
        try:
            # Get target rsids from robotgrid
            target_rsids = list(self.robotgrid.targetDict.keys())
            
            # Setup constraints
            model, wwrt, wwtr, ww_list = self.setup_cp_assignment_constraints(target_rsids)
            
            # Set objective to maximize number of assignments with scientific priority
            self.setup_scientific_priority_objective(wwrt, ww_list)
            
            # Solve the model
            status = self.solver.Solve(self.model)
            
            # Extract assignments
            assignments = {}
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                for robot_id in wwrt:
                    for target_id in wwrt[robot_id]:
                        if self.solver.Value(wwrt[robot_id][target_id]) == 1:
                            assignments[robot_id] = target_id
                            # Actually assign the robot to target in robotgrid
                            self.robotgrid.assignRobot2Target(robot_id, target_id)
                
                status_str = "optimal" if status == cp_model.OPTIMAL else "feasible"
                return assignments, status_str
            else:
                return {}, f"solver_failed: {status}"
                
        except Exception as e:
            return {}, f"error: {str(e)}"

    def setup_cp_assignment_constraints(self, 
                                        rsids, 
                                        check_collisions: bool =True):
        """Setup core CP-SAT constraints for robot-target assignment
        
        Parameters
        ----------
        rsids : list or ndarray
            List of target rsids to consider
        check_collisions : bool
            If True, add collision constraints (default True)
        
        Returns
        -------
        model : cp_model.CpModel
            Model with constraints added
        wwrt : dict
            robotID -> rsid -> BoolVar
        wwtr : dict
            rsid -> robotID -> BoolVar
        ww_list : list
            Flattened list of all BoolVars
        """
        # Add variables; one for each robot-target pair
        # Make a dictionary to organize them as wwrt[robotID][rsid],
        # and one to organize them as wwtr[rsid][robotID], and
        # also a flattened list
        wwrt = dict()
        wwtr = dict()
        for robotID in self.robotgrid.robotDict:
            r = self.robotgrid.robotDict[robotID]
            for rsid in interlist(r.validTargetIDs, rsids):
                name = 'ww[{r}][{c}]'.format(r=robotID, c=rsid)
                if(rsid not in wwtr):
                    wwtr[rsid] = dict()
                if(robotID not in wwrt):
                    wwrt[robotID] = dict()
                wwrt[robotID][rsid] = self.model.NewBoolVar(name)
                wwtr[rsid][robotID] = wwrt[robotID][rsid]

        # List of all robot-target pairs
        ww_list = [wwrt[y][x] for y in wwrt for x in wwrt[y]]

        # Constrain to use only one target per robot
        wwsum_robot = dict()
        for robotID in wwrt:
            rlist = [wwrt[robotID][c] for c in wwrt[robotID]]
            wwsum_robot[robotID] = cp_model.LinearExpr.Sum(rlist)
            self.model.Add(wwsum_robot[robotID] <= 1)

        # Constrain to use only one robot per target
        wwsum_target = dict()
        for rsid in wwtr:
            tlist = [wwtr[rsid][r] for r in wwtr[rsid]]
            wwsum_target[rsid] = cp_model.LinearExpr.Sum(tlist)
            self.model.Add(wwsum_target[rsid] <= 1)

        # Do not allow collisions
        if(check_collisions):
            # Find potential collisions
            collisions = []
            for robotID1 in self.robotgrid.robotDict:
                r1 = self.robotgrid.robotDict[robotID1]
                for rsid1 in r1.validTargetIDs:
                    self.robotgrid.assignRobot2Target(robotID1, rsid1)
                    for robotID2 in r1.robotNeighbors:
                        r2 = self.robotgrid.robotDict[robotID2]
                        for rsid2 in r2.validTargetIDs:
                            if(rsid1 != rsid2):
                                self.robotgrid.assignRobot2Target(robotID2, rsid2)
                                if(self.robotgrid.isCollidedWithAssigned(robotID1)[0]):
                                    collisions.append((robotID1,
                                                    rsid1,
                                                    robotID2,
                                                    rsid2))
                                self.robotgrid.homeRobot(robotID2)
                    self.robotgrid.homeRobot(robotID1)

            # Now add constraint that collisions can't occur
            for robotID1, rsid1, robotID2, rsid2 in collisions:
                ww1 = wwrt[robotID1][rsid1]
                ww2 = wwrt[robotID2][rsid2]
                tmp_collision = cp_model.LinearExpr.Sum([ww1, ww2])
                self.model.Add(tmp_collision <= 1)

        return self.model, wwrt, wwtr, ww_list

    def is_target_scientific(self, target_id: int) -> bool:
        """
        Determine if a target is scientific based on the is_scientific list.
        
        Parameters
        ----------
        target_id : int
            Target ID to check
            
        Returns
        -------
        bool
            True if target is scientific, False otherwise
        """
        if self.is_scientific is None:
            return False
        
        # Create mapping from target ID to scientific status
        # Target IDs in robotgrid may not be sequential due to filtering
        target_ids_list = list(self.robotgrid.targetDict.keys())
        if target_id in target_ids_list:
            target_index = target_ids_list.index(target_id)
            if target_index < len(self.is_scientific):
                return self.is_scientific[target_index]
        return False

    def setup_scientific_priority_objective(self, wwrt: Dict, ww_list: list):
        """
        Setup objective function that prioritizes scientific targets.
        
        This creates a weighted objective where scientific targets have higher weight
        than non-scientific targets, ensuring they are prioritized in the assignment.
        
        Parameters
        ----------
        wwrt : dict
            robotID -> rsid -> BoolVar dictionary
        ww_list : list
            Flattened list of all BoolVars
        """
        if self.is_scientific is None:
            # No scientific priorities, use simple maximization
            self.model.Maximize(cp_model.LinearExpr.Sum(ww_list))
            return
        
        # Create weighted objective terms
        objective_terms = []
        
        for robot_id in wwrt:
            for target_id in wwrt[robot_id]:
                var = wwrt[robot_id][target_id]
                
                if self.is_target_scientific(target_id):
                    # Scientific targets get higher weight (2.0)
                    # This ensures they are prioritized over non-scientific targets
                    objective_terms.append(cp_model.LinearExpr.Term(var, 2.0))
                else:
                    # Non-scientific targets get standard weight (1.0)
                    objective_terms.append(cp_model.LinearExpr.Term(var, 1.0))
        
        if objective_terms:
            self.model.Maximize(cp_model.LinearExpr.Sum(objective_terms))
        else:
            # Fallback to simple maximization if no terms
            self.model.Maximize(cp_model.LinearExpr.Sum(ww_list))

def interlist(list1, list2):
    return(list(set(list1).intersection(list2)))