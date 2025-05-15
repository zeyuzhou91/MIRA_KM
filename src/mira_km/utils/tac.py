"""
Functions related to time-activity curves (TAC).
"""

import os
import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from . import aux  


class TAC_DATA:
    def __init__(self):
        
        self.avg = None
        self.tot = None
        self.std = None
        self.num_voxels = None
        self.unit = None
        
        self.histogram = None
        


def extract_tac_from_PETimg(
        PETimg_path: str, 
        ROImask_path: str,
        opfile_path: str | None = None, 
        PETimg_unit: str | None = None) -> (NDArray, str):
    """
    Extract ROI tac from the PET image and the corresponding mask. The roi object's 
    attributes will be updated. 
    
    Parameters
    ----------
    PETimg_path : file path. Path of the PET image, ending in .nii or .nii.gz
                  Can be either 3D (one frame) or 4D (multi-frame).
    ROImask_path : file path. Path of the ROI mask in PET domain, ending in .nii or .nii.gz
    opfile_path : output file path
    opfile_name : (optional) name of the output filename. 
    PETimg_unit : (optional) unit of the PET image. 

    Returns
    -------
    avg_intensity : average intensity of the ROI at each frame
    unit : output unit of the tac

    """
    
    # For a given PET image, apply a binary mask and generate the ROI information. 
    
    # Load the input PET image
    PETimg = nib.load(PETimg_path)
    PETimg_data = PETimg.get_fdata()
    
    # Load the mask
    ROImask = nib.load(ROImask_path)
    ROImask_data = ROImask.get_fdata()
    
    
    
    result = TAC_DATA()
    result.num_voxels = int(np.count_nonzero(ROImask_data))
    
    if len(PETimg.shape) == 3:
        # one frame (i.e. single 3D image)
        
        num_frames = 1
        
        assert PETimg.shape == ROImask.shape, f"""The input image {PETimg_path} (shape = {PETimg_data.shape}) and 
        the mask {ROImask_path} (shape = mask_data.shape) should have the same dimension."""
        
        # outputs an array containing only the masked voxel values in PETimg
        ROI_reduced = np.extract(ROImask_data, PETimg_data)
        
        if PETimg_unit == 'Bq/mL':
            ROI_reduced /= 1000.0 # from Bq/mL to kBq/mL
                
        result.tot = np.array([np.sum(ROI_reduced)])
        result.avg = np.array([np.mean(ROI_reduced)])
        result.std = np.array([np.std(ROI_reduced)])
        
        
        
    elif len(PETimg.shape) == 4:
        # multi-frame 
        
        num_frames = PETimg.shape[-1]
        
        assert PETimg.shape[0:3] == ROImask.shape, f"""Each frame of the input image {PETimg_path} (shape = {PETimg.shape[0:3]}) and 
        the mask {ROImask_path} (shape = ROImask_data.shape) should have the same dimension."""
    
        result.tot = np.zeros(num_frames)
        result.avg = np.zeros(num_frames)
        result.std = np.zeros(num_frames)     
        result.histogram = ()
        for i in range(num_frames):
            
            # the ith frame
            frame = PETimg_data[..., i]
            
            # outputs an array containing only the masked voxel values in this frame
            frame_ROI_reduced = np.extract(ROImask_data, frame)
            
            if PETimg_unit == 'Bq/mL':
                frame_ROI_reduced /= 1000.0 # from Bq/mL to kBq/mL
            
            result.tot[i] = np.sum(frame_ROI_reduced)
            result.avg[i] = np.mean(frame_ROI_reduced)
            result.std[i] = np.std(frame_ROI_reduced)
            result.histogram += (np.histogram(frame_ROI_reduced, bins=range(26)),)
           
    
    if PETimg_unit == 'kBq/mL':
        #aux.write_to_csv_onecol(avg, 'average', 'kBq/mL', opfile_path)
        if opfile_path is not None:
            aux.write_to_csv_threecols(result.avg, 'average', 'kBq/mL', 
                                       result.std, 'std', 'kBq/mL', 
                                       result.num_voxels * np.ones(num_frames, dtype=int), 'num_voxels', '#',
                                       opfile_path)
        result.unit = PETimg_unit
        
    elif PETimg_unit == 'Bq/mL':
        if opfile_path is not None:
            aux.write_to_csv_threecols(result.avg, 'average', 'kBq/mL', 
                                     result.std, 'std', 'kBq/mL', 
                                     result.num_voxels * np.ones(num_frames, dtype=int), 'num_voxels', '#',
                                     opfile_path)
        result.unit = 'kBq/mL'
        
    elif PETimg_unit == 'unitless':
        if opfile_path is not None:
            aux.write_to_csv_threecols(result.avg, 'average', 'unitless', 
                                     result.std, 'std', 'unitless', 
                                     result.num_voxels * np.ones(num_frames, dtype=int), 'num_voxels', '#',
                                     opfile_path)
        result.unit = 'unitless'

    return result
            
            


def extract_tac_from_csv(filepath: str) -> (NDArray, str):
    """
    Extract ROI tac from a given csv file. The tac0's attributes will be updated. 
    
    Parameters
    ----------
    filepath : csv file path. 
    
    Returns
    -------
    ys : tac values 
    unit : unit of ys
    
    """
    
    #ys, header, unit = aux.read_from_csv_onecol(filepath)
    
    avg, header1, unit1, std, header2, unit2, num_voxels_list, header3, unit3 =  aux.read_from_csv_threecols(filepath)
    
    avg = np.array(avg)
    std = np.array(std)
    
    if unit1 == 'kBq/mL':
        pass

    elif unit1 == 'Bq/mL':
        avg = avg / 1000.0
        std = std / 1000.0
        unit1 = 'kBq/mL'

    elif unit1 == 'unitless':
        pass
    
    num_voxels = int(num_voxels_list[0])
    return avg, std, num_voxels, unit1
    

            