#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Target Integration Utilities

This module provides helper functions to integrate generated targets
with the kaiju robotgrid system.
"""

import kaiju
import numpy as np
from typing import Tuple


def add_targets_to_robotgrid(robotgrid, 
                           x: np.ndarray, 
                           y: np.ndarray, 
                           priorities: np.ndarray = None,
                           fiber_types: np.ndarray = None) -> None:
    """
    Add generated targets to the robotgrid.
    
    Args:
        robotgrid: kaiju RobotGrid instance (RobotGridNominal doesn't need initialization)
        x: Array of x coordinates
        y: Array of y coordinates  
        priorities: Array of priority levels (default: all priority 1)
        fiber_types: Array of fiber types ('BOSS' or 'APOGEE', default: all BOSS)
    """
    # RobotGridNominal doesn't require initialization check
    # The base RobotGrid class would need initialization, but RobotGridNominal is pre-configured
    
    # Set defaults
    if priorities is None:
        priorities = np.ones(len(x), dtype=int)
    if fiber_types is None:
        fiber_types = np.array(['BOSS'] * len(x))
    
    # Convert fiber type strings to kaiju enum
    fiber_type_map = {
        'BOSS': kaiju.cKaiju.BossFiber,
        'APOGEE': kaiju.cKaiju.ApogeeFiber
    }
    
    # Add each target
    for i in range(len(x)):
        target_id = i  # Use index as target ID
        xyz_wok = [x[i], y[i], 0.0]  # z=0 for flat field
        fiber_type = fiber_type_map[fiber_types[i]]
        priority = int(priorities[i])
        
        robotgrid.addTarget(
            targetID=target_id,
            xyzWok=xyz_wok,
            fiberType=fiber_type,
            priority=priority
        )


def clear_robotgrid_assignments(robotgrid: kaiju.robotGrid) -> None:
    """
    Clear all robot-target assignments from the robotgrid.
    
    Args:
        robotgrid: kaiju RobotGrid instance
    """
    # Home all robots to clear assignments
    for robot_id in robotgrid.robotDict:
        robotgrid.homeRobot(robot_id)


def get_target_summary(robotgrid: kaiju.robotGrid) -> dict:
    """
    Get summary statistics about targets in the robotgrid.
    
    Args:
        robotgrid: kaiju RobotGrid instance
        
    Returns:
        Dictionary with target statistics
    """
    total_targets = len(robotgrid.targetDict)
    assigned_targets = 0
    boss_targets = 0
    apogee_targets = 0
    
    for target in robotgrid.targetDict.values():
        if target.isAssigned():
            assigned_targets += 1
        
        if target.fiberType == kaiju.cKaiju.BossFiber:
            boss_targets += 1
        elif target.fiberType == kaiju.cKaiju.ApogeeFiber:
            apogee_targets += 1
    
    return {
        'total_targets': total_targets,
        'assigned_targets': assigned_targets,
        'unassigned_targets': total_targets - assigned_targets,
        'boss_targets': boss_targets,
        'apogee_targets': apogee_targets
    }
