"""
Frame Schedule
"""


import numpy as np
from numpy.typing import NDArray
from .tool import aux


class FrameSchedule:
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

    
    
    
    