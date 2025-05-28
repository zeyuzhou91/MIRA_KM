"""
smooth.py

Smoothing functions using Gaussian filters.

Includes:
- gaussian_filter_3D_local(): Apply Gaussian smoothing locally on a 3D image.
- gaussian_filter_3D(): Apply Gaussian smoothing on a 3D image with specified paths.

Author: Zeyu Zhou
Date: 2025-05-21
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import copy




def gaussian_filter_3D_local(
        sigma: float,
        ipimg: str, 
        opimg: str) -> None:
    """
    Apply a Gaussian filter to a 3D medical image using a specified standard deviation (sigma).
    This version assumes the input and output files are both local.
    
    Parameters:
    - sigma: Standard deviation for Gaussian kernel.
    - ipimg: Input NIfTI file path.
    - opimg: Output NIfTI file path.
    
    Returns:
    - None. Writes the smoothed image to disk.
    """
    
    # load input image
    ip = nib.load(ipimg)
    
    # make a copy of the image 
    op_data = copy.deepcopy(ip.get_fdata())
    
    # Apply Gaussian filter with the given sigma
    op_data = gaussian_filter(op_data, sigma=sigma)
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip.affine, ip.header)
            
    # Save the output image
    nib.save(op, opimg)
    
    return None
    
    

def gaussian_filter_3D(
        sigma: float,
        ippath: str, 
        oppath: str) -> None:
    """
    Apply a Gaussian filter to a 3D medical image using a specified standard deviation (sigma).
    General version for use in pipelines where file paths are specified.
    
    Parameters:
    - sigma: Standard deviation for Gaussian kernel.
    - ippath: Input NIfTI file path.
    - oppath: Output NIfTI file path.
    
    Returns:
    - None. Writes the smoothed image to disk.
    """
    
    # load input image
    ip = nib.load(ippath)
    
    # make a copy of the image 
    op_data = copy.deepcopy(ip.get_fdata())
    
    
    # Apply Gaussian filter with the given sigma
    op_data = gaussian_filter(op_data, sigma=sigma)
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip.affine, ip.header)
            
    # Save the output image
    nib.save(op, oppath)
    
    return None
            
        
            
            
            
            