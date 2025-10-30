import kaiju
import numpy as np
from config_robots import load_apo_grid
from targets import sample_ppp
from tqdm import tqdm

'''
def wouldCollideWithAssigned(robotID: int, targID: int) -> bool:
'''

if __name__ == "__main__":
    robotgrid = load_apo_grid()
    
    # Sample targets using Poisson Point Process
    lam_per_area = 0.002  # Rate parameter (targets per mm^2)
    xlim = (-300, 300)  # X limits in mm
    ylim = (-300, 300)  # Y limits in mm
    rng = np.random.default_rng(42)  # Seed for reproducibility
    
    x_coords, y_coords = sample_ppp(lam_per_area, xlim, ylim, rng)
    
    print(f"Sampled {len(x_coords)} targets")
    
    # Add targets to the robotgrid
    for i in range(len(x_coords)):
        target_id = i
        xyz_wok = [x_coords[i], y_coords[i], 0.0]  # z=0 for flat field
        fiber_type = kaiju.cKaiju.BossFiber  # Use BOSS fiber
        priority = 1  # Default priority
        
        robotgrid.addTarget(
            targetID=target_id,
            xyzWok=xyz_wok,
            fiberType=fiber_type,
            priority=priority
        )
    
    print(f"Added {len(robotgrid.targetDict)} targets to robotgrid")
    
    # Assign some targets to robots to create collision scenarios
    # Get some robot IDs and target IDs
    robot_ids = list(robotgrid.robotDict.keys()) # Test with first 50 robots
    target_ids = list(robotgrid.targetDict.keys())  # Test with first 100 targets
    
    print(f"\nTesting with {len(robot_ids)} robots and {len(target_ids)} targets")
    
    # Assign some targets to create interesting collision scenarios
    assignments_made = []
    for i, robot_id in enumerate(robot_ids):  # Assign to first 20 robots
        robot = robotgrid.robotDict[robot_id]
        # Try to assign a valid target
        if len(robot.validTargetIDs) > 0:
            # Pick a valid target for this robot
            valid_targets = list(robot.validTargetIDs)
            if i < len(valid_targets):
                target_id = valid_targets[i % len(valid_targets)]
                robot.assignedTargetID = target_id
                assignments_made.append((robot_id, target_id))
                print(f"Assigned target {target_id} to robot {robot_id}")
    
    print(f"\nMade {len(assignments_made)} assignments")
    
    # Now test wouldCollideWithAssigned vs wouldCollideWithAssignedFast
    print("\n" + "="*60)
    print("Testing wouldCollideWithAssigned vs wouldCollideWithAssignedFast")
    print("="*60)
    
    test_count = 0
    mismatch_count = 0
    
    # Test multiple robot-target combinations
    for robot_id in tqdm(robot_ids):
        for target_id in target_ids:
            # Only test valid assignments
            robot = robotgrid.robotDict[robot_id]
            if target_id not in robot.validTargetIDs:
                continue
            
            # Call both methods
            result_original = robotgrid.wouldCollideWithAssigned(robot_id, target_id)
            result_fast = robotgrid.wouldCollideWithAssignedFast(robot_id, target_id)
            
            test_count += 1
            
            # Check if results match
            if result_original != result_fast:
                mismatch_count += 1
                print(f"MISMATCH! Robot {robot_id}, Target {target_id}:")
                print(f"  wouldCollideWithAssigned: {result_original}")
                print(f"  wouldCollideWithAssignedFast: {result_fast}")
            
            # Assert they should be the same
            assert result_original == result_fast, \
                f"Methods returned different results for robot {robot_id}, target {target_id}: " \
                f"original={result_original}, fast={result_fast}"
    
    print(f"\n✓ Tested {test_count} robot-target combinations")
    print(f"✓ All results matched! (0 mismatches out of {test_count} tests)")
    print(f"✓ wouldCollideWithAssigned and wouldCollideWithAssignedFast are equivalent")