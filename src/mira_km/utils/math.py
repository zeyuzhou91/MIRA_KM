"""
Math functions. 

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
    Multiply the image by a given value.
    
    Do it locally, read the local ipimg and produce the opimg to the same folder. 
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
    Multiply the image by a given value.
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
    Threshold the image by the given lower-bound and higher-bound. 

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
    Threshold the image by the given lower-bound and higher-bound. 

    DUPLICATE WITH function clip. TO DELETE. 
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
    Set all values above the threshold to up_class, below the threshold to
    down_class. 
    
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
    Clip the image by the given lower-bound and higher-bound. 

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
    Find the max value in a given image. 

    """

    # load input image
    ip = nib.load(ippath)
    
    op_data = ip.get_fdata()
    
    max_v = np.max(op_data)
    
    return max_v


def percentile_value_in_image(q: float, 
                              ippath: str) -> float:
    """
    Find the q-th percentile of all pixel/voxel values in an image. 
    
    q: in [0, 100]

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
    Add two images. They must be of the same dimension. 

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
    Find average of N images. They must be of the same dimension. 

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
    fwhm: full width at half maximum
    sigma: standard deviation
    """
    
    
    r = 2 * np.sqrt(2 * np.log(2))
    
    sigma = fwhm / r
    
    return sigma


def gaussian_sigma2fwhm(sigma: float) -> float:
    """
    fwhm: full width at half maximum
    sigma: standard deviation
    """
    
    r = 2 * np.sqrt(2 * np.log(2))
    
    fwhm = r * sigma
    
    return fwhm

            