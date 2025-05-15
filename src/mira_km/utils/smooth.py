"""
Smoothing functions. 
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
    Apply a Gaussian filter to the input 3D image. Do it locally. 
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
    Apply a Gaussian filter to the input 3D image. 
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
            
        
            
            
            
            