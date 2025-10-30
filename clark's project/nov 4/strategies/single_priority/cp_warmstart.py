import kaiju
import random
from ortools.sat.python import cp_model
from ..base_strategy import Strategy
from .greedy import GreedyClosestStrategy
from typing import Dict, Tuple
from tqdm import tqdm

class CPWarmstartStrategy(Strategy):
    """
    CP-SAT strategy that uses greedy algorithm result as warmstart.
    
    This strategy first runs a greedy algorithm to get an initial assignment,
    then uses a random fraction of that assignment as hints for the CP-SAT solver
    to find an improved solution. The hint_fraction parameter allows tuning between
    using the greedy solution as a strong starting point (1.0) and letting the 
    CP-SAT solver explore more freely (0.0).
    """
    
    def __init__(self, 
                 robotgrid: kaiju.robotGrid, 
                 targets: list[float], 
                 seed: int,
                 time_limit: float,
                 num_workers: int = 0,
                 is_scientific: list[bool] = None,
                 use_closest_greedy: bool = True,
                 hint_fraction: float = 0.1):
        """
        Initialize CP-SAT strategy with greedy warmstart.
        
        Parameters
        ----------
        robotgrid : kaiju.robotGrid
            The robot grid containing robots and targets
        targets : list[float]
            List of target values
        seed : int
            Random seed for reproducible results
        time_limit : float
            Time limit for CP-SAT solver in seconds
        num_workers : int, optional
            Number of workers for CP-SAT solver (0 = auto), by default 0
        is_scientific : list[bool], optional
            List indicating which targets are scientific, by default None
        use_closest_greedy : bool, optional
            Whether to use closest greedy strategy, by default True
        hint_fraction : float, optional
            Fraction of greedy assignments to use as hints (0.0 to 1.0), by default 1.0
            - 1.0: Use all greedy assignments as hints (strong guidance)
            - 0.5: Use half of greedy assignments as hints (balanced)
            - 0.0: Use no hints (let CP-SAT explore freely)
        """
        super().__init__(robotgrid=robotgrid, targets=targets, seed=seed, is_scientific=is_scientific)
        
        self.time_limit = time_limit
        self.num_workers = num_workers
        self.use_closest_greedy = use_closest_greedy
        self.hint_fraction = hint_fraction
        
        # Initialize CP-SAT model and solver
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = time_limit
        self.solver.parameters.num_workers = num_workers
        
        # Cache for collision detection to avoid recomputing
        self._collision_cache = None

    def optimize(self) -> Tuple[Dict[int, int], str]:
        """
        Solve using greedy warmstart + CP-SAT optimization.
        """
        try:
            # Step 1: Get greedy assignment as warmstart
            greedy_assignments = self._get_greedy_warmstart()
            
            # Step 2: Setup CP-SAT model with constraints
            target_rsids = list(self.robotgrid.targetDict.keys())
            model, wwrt, wwtr, ww_list = self.setup_cp_assignment_constraints(target_rsids)
            
            # Step 3: Set objective with scientific priority
            self.setup_scientific_priority_objective(wwrt, ww_list)
            
            # Step 4: Add warmstart hints to the solver
            self._add_warmstart_hints(wwrt, greedy_assignments)
            
            # Step 5: Solve the model
            status = self.solver.Solve(self.model)
            
            # Step 6: Extract assignments
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
                # If CP-SAT fails, fall back to greedy result
                return greedy_assignments, f"cp_sat_failed_fallback_to_greedy: {status}"
                
        except Exception as e:
            # If anything fails, try to return greedy result
            try:
                greedy_assignments = self._get_greedy_warmstart()
                return greedy_assignments, f"error_fallback_to_greedy: {str(e)}"
            except:
                return {}, f"error: {str(e)}"

    def _get_greedy_warmstart(self) -> Dict[int, int]:
        """
        Get greedy assignment to use as warmstart.
        """
        if self.use_closest_greedy:
            greedy_strategy = GreedyClosestStrategy(
                robotgrid=self.robotgrid,
                targets=self.targets,
                seed=self.seed,
                is_scientific=self.is_scientific
            )
        else:
            from .greedy import GreedyFurthestStrategy
            greedy_strategy = GreedyFurthestStrategy(
                robotgrid=self.robotgrid,
                targets=self.targets,
                seed=self.seed,
                is_scientific=self.is_scientific
            )
        
        # Run greedy optimization
        assignments, _ = greedy_strategy.optimize()
        return assignments

    def _add_warmstart_hints(self, wwrt: Dict, greedy_assignments: Dict[int, int]):
        """
        Add warmstart hints to the CP-SAT solver based on greedy assignments.
        
        This method uses model.add_hint to provide a random fraction of the greedy 
        algorithm's assignments as hints to the CP-SAT solver. This allows tuning
        between using the greedy solution as a strong starting point (hint_fraction=1.0)
        and letting the CP-SAT solver explore more freely (hint_fraction=0.0).
        
        Parameters
        ----------
        wwrt : Dict
            robotID -> rsid -> BoolVar dictionary containing the assignment variables
        greedy_assignments : Dict[int, int]
            robotID -> targetID mapping from greedy algorithm
        """
        # Set random seed for reproducible hint selection
        random.seed(self.seed)
        
        # Convert greedy assignments to list for random sampling
        greedy_list = list(greedy_assignments.items())
        
        # Calculate how many hints to add based on hint_fraction
        num_hints_to_add = int(len(greedy_list) * self.hint_fraction)
        
        # Randomly sample which greedy assignments to use as hints
        selected_assignments = random.sample(greedy_list, num_hints_to_add) if num_hints_to_add > 0 else []
        
        # Add hints for selected greedy assignments
        for robot_id, target_id in selected_assignments:
            # Check if this robot-target pair exists in our variables
            if robot_id in wwrt and target_id in wwrt[robot_id]:
                # Hint that this assignment should be True (1)
                self.model.add_hint(wwrt[robot_id][target_id], 1)
            else:
                # This shouldn't happen if greedy algorithm is working correctly,
                # but log it for debugging
                print(f"Warning: Greedy assignment robot {robot_id} -> target {target_id} not found in CP-SAT variables")
        
        # For unassigned robots, only add negative hints if hint_fraction > 0
        # This prevents over-constraining when hint_fraction is low
        if self.hint_fraction > 0:
            for robot_id in wwrt:
                if robot_id not in greedy_assignments:
                    # This robot wasn't assigned by greedy, hint all its variables to False
                    for target_id in wwrt[robot_id]:
                        self.model.add_hint(wwrt[robot_id][target_id], 0)

    def _get_precomputed_collisions(self, rsids):
        """Precompute all potential collisions once and cache them"""
        if not hasattr(self, '_collision_cache') or self._collision_cache is None:
            self._collision_cache = self._compute_all_collisions(rsids)
        return self._collision_cache

    def _compute_all_collisions(self, rsids):
        """Compute all potential robot-target collision pairs"""
        collisions = []
        
        # Only check robot pairs that are neighbors (much smaller set)
        for robotID1 in tqdm(self.robotgrid.robotDict):
            r1 = self.robotgrid.robotDict[robotID1]
            for rsid1 in interlist(r1.validTargetIDs, rsids):
                # Temporarily assign robot1 to target1
                self.robotgrid.assignRobot2Target(robotID1, rsid1)
                
                # Only check neighboring robots (not all robots)
                for robotID2 in r1.robotNeighbors:
                    r2 = self.robotgrid.robotDict[robotID2]
                    for rsid2 in interlist(r2.validTargetIDs, rsids):
                        if rsid1 != rsid2:
                            # Temporarily assign robot2 to target2
                            self.robotgrid.assignRobot2Target(robotID2, rsid2)
                            
                            # Check collision
                            if self.robotgrid.isCollidedWithAssigned(robotID1)[0]:
                                collisions.append((robotID1, rsid1, robotID2, rsid2))
                            
                            # Reset robot2
                            self.robotgrid.homeRobot(robotID2)
                
                # Reset robot1
                self.robotgrid.homeRobot(robotID1)
        
        return collisions

    def _invalidate_collision_cache(self):
        """Call this when robotgrid or targets change"""
        self._collision_cache = None

    def setup_cp_assignment_constraints(self, 
                                        rsids, 
                                        check_collisions: bool = True):
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
        if(check_collisions and self.time_limit > 0.1):  # Skip collision detection for very short time limits
            # Use precomputed collision matrix instead of nested loops
            collisions = self._get_precomputed_collisions(rsids)
            
            # Add constraint that collisions can't occur
            for robotID1, rsid1, robotID2, rsid2 in collisions:
                # Check if these robot-target pairs exist in our variables
                if (robotID1 in wwrt and rsid1 in wwrt[robotID1] and 
                    robotID2 in wwrt and rsid2 in wwrt[robotID2]):
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
    """Helper function to find intersection of two lists"""
    return list(set(list1).intersection(list2))