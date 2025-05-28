"""
edge_detection.py

This module provides a function to perform edge detection on images using a MATLAB implementation 
of the edge3-approxcanny algorithm. The function uses the MATLAB Engine API for Python to call 
a MATLAB script from Python.

Requirements:
- MATLAB must be installed with Python integration (matlab.engine module).
- The 'approxcanny' MATLAB function must be available in the specified MATLAB directory.

Author: Zeyu Zhou
Date: 2025-05-20
"""

import matlab.engine


def matlab_approxcanny(
    thresh: float,
    sigma: float, 
    infilepath: str,
    outfilepath: str,
    matlab_dir: str
) -> None:
    """
    Applies the edge3-approxcanny edge detection algorithm using MATLAB.

    Parameters:
    -----------
    thresh : float
        High sensitivity threshold for the Canny algorithm. The low threshold is set as 0.4 * thresh.
    sigma : float
        Standard deviation for the Gaussian smoothing filter.
    infilepath : str
        Full path to the input image file.
    outfilepath : str
        Full path to save the output image with detected edges.
    matlab_dir : str
        Path to the directory containing the MATLAB 'approxcanny' function.

    Returns:
    --------
    None
    """
    # Start MATLAB engine and add the directory containing the function
    eng = matlab.engine.start_matlab()
    eng.addpath(matlab_dir)

    # Call the MATLAB function
    eng.approxcanny(infilepath, outfilepath, thresh, sigma, nargout=0)

    # Optionally, stop the MATLAB engine if needed
    # eng.quit()

    return None
            
        
            
            
            
            