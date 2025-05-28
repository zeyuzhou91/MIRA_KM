"""
math.py

Utility functions for voxel-wise image manipulation.

Includes:
- Arithmetic operations (multiply, add, average)
- Thresholding and classification
- Clipping with or without mask
- Statistical queries (max value, percentile)
- Gaussian FWHM/sigma conversions

Dependencies:
- Nibabel, NumPy, SciPy, FSL

Author: Zeyu Zhou
Date: 2025-05-21
"""

import os
import numpy as np
import nibabel as nib
import copy
import subprocess



def multiply_local(
        value: int | float, 
        ipimg: str,
        opimg: str) -> None:
    """
    Multiply a NIfTI image by a constant value and save the result in the same directory.
    
    Parameters:
    - value: Scalar to multiply.
    - ipimg: Path to input image.
    - opimg: Path to output image.
    """
    
    # load input image
    ip = nib.load(ipimg)
    
    # make a copy of the image 
    op_data = copy.deepcopy(ip.get_fdata())
    
    # Multiply
    op_data = op_data * value
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip.affine, ip.header)
            
    # Save the output image
    nib.save(op, opimg)
    
    return None



def multiply(
        value: int | float, 
        ippath: str,
        oppath: str) -> None:
    """
    Multiply a NIfTI image by a constant value.
    
    Parameters:
    - value: Scalar multiplier.
    - ippath: Input image path.
    - oppath: Output image path.
    """
    
    # load input image
    ip = nib.load(ippath)
    
    # make a copy of the image 
    op_data = copy.deepcopy(ip.get_fdata())
    
    # Multiply
    op_data = op_data * value
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip.affine, ip.header)
            
    # Save the output image
    nib.save(op, oppath)
    
    return None



def thresholding_local(
        lb: float,
        ub: float,
        ipimg: str,
        opimg: str) -> None:
    """
    Clip voxel values in a NIfTI image to be within [lb, ub] bounds.
    
    Parameters:
    - lb: Lower bound.
    - ub: Upper bound.
    - ipimg: Input image path.
    - opimg: Output image path.
    """

    # load input image
    ip = nib.load(ipimg)
    
    # make a copy of the image 
    op_data = copy.deepcopy(ip.get_fdata())
    
    # thresholding
    op_data = np.clip(op_data, lb, ub)
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip.affine, ip.header)
            
    # Save the output image
    nib.save(op, opimg)
    
    return None
            
        
def thresholding(
        lb: float,
        ub: float,
        ippath: str,
        oppath: str) -> None:
    """
    Deprecated duplicate of `clip()`. Threshold a NIfTI image using lower and upper bounds.
    
    DUPLICATE WITH function clip. TO DELETE. 
    
    Parameters:
    - lb: Lower bound.
    - ub: Upper bound.
    - ippath: Input image path.
    - oppath: Output image path.
    """

    # load input image
    ip = nib.load(ippath)
    
    # make a copy of the image 
    op_data = copy.deepcopy(ip.get_fdata())
    
    # thresholding
    op_data = np.clip(op_data, lb, ub)
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip.affine, ip.header)
    nib.save(op, oppath)
    
            
    # Save the output image
    return None   
        


def threshold_and_classify(
        threshold: float,
        up_class: int,
        down_class: int,
        ippath: str,
        oppath: str) -> None:
    """
    Assign voxels a binary label based on threshold comparison.
    
    Parameters:
    - threshold: Threshold value.
    - up_class: Value for voxels >= threshold.
    - down_class: Value for voxels < threshold.
    - ippath: Input image path.
    - oppath: Output image path.
    """

    # load input image
    ip = nib.load(ippath)
    
    # make a copy of the image 
    op_data = copy.deepcopy(ip.get_fdata())
    
    # above threshold: up_class
    # below threshold: down_class
    op_data = np.where(op_data >= threshold, up_class, down_class)
    
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip.affine, ip.header)
            
    # Save the output image
    nib.save(op, oppath)
    
    return None 



def clip(
        infile_path: str,
        outfile_path: str,
        lb: float,
        ub: float,
        mask_path: str | None = None) -> None:
    """
    Clip voxel values in a NIfTI image to [lb, ub], optionally within a binary mask.
    
    Parameters:
    - infile_path: Path to input image.
    - outfile_path: Output path.
    - lb: Lower bound.
    - ub: Upper bound.
    - mask_path: Optional mask (values outside mask are not clipped).
    """

    # load input image
    infile = nib.load(infile_path)
    
    # make a copy of the image 
    in_data = copy.deepcopy(infile.get_fdata())
    
    # clip 
    in_clip_data = np.clip(in_data, lb, ub)
    
    
    if mask_path is None:
        
        out_data = in_clip_data
        
    else:
        
        # load mask image
        maskfile = nib.load(mask_path)
    
        # make a copy of the mask
        mask = copy.deepcopy(maskfile.get_fdata()).astype(int)
    
        # complement of mask
        mask_compl = 1 - mask
        
        # clip the mask region, do not clip the mask complement region
        out_data = (in_clip_data * mask + in_data * mask_compl).astype(in_data.dtype)
            
    # Make the output image
    outfile = nib.Nifti1Image(out_data, infile.affine, infile.header)
            
    # Save the output image
    nib.save(outfile, outfile_path)
    
    return None 



    

def max_value_in_image(ippath: str) -> float:
    """
    Return the maximum voxel value in a NIfTI image.
    
    Parameters:
    - ippath: Path to image file.
    
    Returns:
    - Maximum value (float).
    """

    # load input image
    ip = nib.load(ippath)
    
    op_data = ip.get_fdata()
    
    max_v = np.max(op_data)
    
    return max_v


def percentile_value_in_image(q: float, 
                              ippath: str) -> float:
    """
    Compute a percentile value from voxel intensities in a NIfTI image.
    
    Parameters:
    - q: Percentile to compute (0–100).
    - ippath: Input image path.
    
    Returns:
    - The q-th percentile value.
    """

    # load input image
    ip = nib.load(ippath)
    
    op_data = ip.get_fdata()
    
    r = np.percentile(op_data.flatten(), q)
    
    return r



        
def add(
        ippath1: str,
        ippath2: str,
        oppath: str) -> None:
    """
    Add two NIfTI images element-wise.
    
    Parameters:
    - ippath1: First input image.
    - ippath2: Second input image.
    - oppath: Output image path.
    """
    
    ip1 = nib.load(ippath1)
    ip1_data = copy.deepcopy(ip1.get_fdata())
    
    ip2 = nib.load(ippath2)
    ip2_data = copy.deepcopy(ip2.get_fdata())
    
    op_data = ip1_data + ip2_data
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip1.affine, ip1.header)
            
    # Save the output image
    nib.save(op, oppath)
    
    return None



def average(
        ippaths: list[str],
        oppath: str) -> None:
    """
    Compute the voxel-wise average of a list of NIfTI images using FSL.
    
    Parameters:
    - ippaths: List of input paths.
    - oppath: Output path.
    """
    
    command = ['fslmaths']
    
    for (i, ippath) in enumerate(ippaths):
        
        if i == 0:
            command += [ippath]
        else:
            command += ['-add', ippath]
            
    n = len(ippaths)
    
    command += ['-div', str(n), oppath]
    
    subprocess.run(command)
    
    return None



def gaussian_fwhm2sigma(fwhm: float) -> float:
    """
    Convert Gaussian full width at half maximum (FWHM) to standard deviation (sigma).
    
    Parameters:
    - fwhm: Full width at half maximum.
    
    Returns:
    - sigma: Standard deviation.
    """
    
    
    r = 2 * np.sqrt(2 * np.log(2))
    
    sigma = fwhm / r
    
    return sigma


def gaussian_sigma2fwhm(sigma: float) -> float:
    
    """
    Convert Gaussian standard deviation (sigma) to full width at half maximum (FWHM).
    
    Parameters:
    - sigma: Standard deviation.
    
    Returns:
    - fwhm: Full width at half maximum.
    """
    
    r = 2 * np.sqrt(2 * np.log(2))
    
    fwhm = r * sigma
    
    return fwhm

            