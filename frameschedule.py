"""
Frame Schedule
"""


import numpy as np
from numpy.typing import NDArray
from .utils import auxiliary as aux


class FrameSchedule:
    """
    Class representing the schedule of frames in dynamic PET imaging.

    Attributes:
    - mid_points (NDArray): Time at the midpoint of each frame
    - durations (NDArray): Duration of each frame
    - start_points (NDArray): Start time of each frame
    - unit (str): Unit of time (e.g., seconds, minutes)
    - num_frames (int): Total number of frames
    """
    
    def __init__(self, mid_points: NDArray, durations: NDArray, start_points: NDArray, unit: str):
        self.mid_points = mid_points
        self.durations = durations         
        self.start_points = start_points
        self.unit = unit
        self.num_frames = len(mid_points)
        
    @classmethod
    def from_midpoints(cls, 
                       array: list[float] = None, 
                       filepath: str = None):
        """
        Create a FrameSchedule from an array of midpoints or a file.

        Parameters:
        - array (list of float): List of frame mid-points.
        - filepath (str): Path to a CSV file with one column of midpoints.

        Returns:
        - FrameSchedule object
        """
        
        if array is not None and filepath is not None:
            raise ValueError("Only one of array or filepath should be provided, not both.")
        
        elif array is not None:
            mid_points = array
            unit = ''
            
        elif filepath is not None:
            mid_points, _, unit = aux.read_from_csv_onecol(filepath)
        
        else:
            raise ValueError("Either array or filepath must be provided.")
        
        durations = []
        start_points = []
        
        cur_start = 0.0
        for mid in mid_points:
            
            start_points.append(cur_start)
            
            duration = 2 * (mid - cur_start)
            durations.append(duration)
            
            nxt_start = cur_start + duration
            
            cur_start = nxt_start
        
        mid_points = np.array(mid_points)
        durations = np.array(durations)
        start_points = np.array(start_points)
        
        return cls(mid_points, durations, start_points, unit)

    
    @classmethod
    def from_durations(cls, 
                       array: list[float] = None,
                       filepath: str = None):
        """
        Create a FrameSchedule from an array of durations or a file.

        Parameters:
        - array (list of float): List of frame durations.
        - filepath (str): Path to a CSV file with one column of durations.

        Returns:
        - FrameSchedule object
        """
        
        if array is not None and filepath is not None:
            raise ValueError("Only one of array or filepath should be provided, not both.")
        
        elif array is not None:
            durations = array
            unit = ''
            
        elif filepath is not None:
            durations, _, unit = aux.read_from_csv_onecol(filepath)
        
        else:
            raise ValueError("Either array or filepath must be provided.")
        
        
        mid_points = []
        start_points = []
        
        cur_start = 0.0
        for duration in durations:
            
            start_points.append(cur_start)
            
            nxt_start = cur_start + duration
            
            mid = (cur_start + nxt_start) / 2.0
            mid_points.append(mid)
            
            cur_start = nxt_start
        
        mid_points = np.array(mid_points)
        durations = np.array(durations)
        start_points = np.array(start_points)
        
        return cls(mid_points, durations, start_points, unit)

    