"""
partial_volume_correction.py

Functions related to Partial Volume Correction (PVC) for PET imaging.
Supports both 3D and 4D data, using the PETPVC tool via Nipype.

Includes:
- seg_3Dto4D(): Converts 3D segmentation to 4D for PVC use.
- pvc(): Applies PVC to single or multi-frame PET images.

Author: Zeyu Zhou
Date: 2025-05-21
"""

import os
import nibabel as nib
from pathlib import Path
import glob
import shutil
from nipype.interfaces.petpvc import PETPVC
from .pet_image_processing import split_4D_into_frames, concatenate_frames


def seg_3Dto4D(
        ippath: str,
        oppath: str) -> None:
    """
    Convert a 3D segmentation file to a 4D format using pvc_make4d.
    This is required for frame-wise partial volume correction (PVC).
    
    Parameters:
    - ippath: Input 3D segmentation file path.
    - oppath: Output 4D segmentation file path.
    
    Returns:
    - None. Saves the output directly.
    """    
    
    os.system(f'pvc_make4d -i {ippath} -o {oppath}')
        
    return None



def pvc(in_file: str,
        mask_file: str,
        out_file: str,
        algo: str,
        fwhm: float) -> None:
    """
    Apply Partial Volume Correction (PVC) using the PETPVC interface from Nipype.
    Supports both 3D and 4D PET images. Uses the specified segmentation mask.
    
    Parameters:
    - in_file: Input PET image (.nii or .nii.gz), either 3D or 4D.
    - mask_file: Segmentation or anatomical mask to guide PVC.
    - out_file: Output file path for PVC-corrected image.
    - algo: PVC algorithm identifier (e.g., 'gtm', 'labbe').
    - fwhm: Full-width-half-maximum (FWHM) value for Gaussian kernel in mm.
    
    Returns:
    - None. Writes output image(s) directly to disk.
    """
    
    print('###################### PVCing ##########################')
    
    p = PETPVC()
    
    img = nib.load(in_file)
    hdr = img.header
    dim = hdr['dim'][0]
    #print('Dimension: ', dim)
    
    if dim == 3: # PVC for a single image
        
        p.inputs.in_file   = in_file
        p.inputs.mask_file = mask_file
        p.inputs.out_file  = out_file
        p.inputs.pvc = algo
        p.inputs.fwhm_x = fwhm
        p.inputs.fwhm_y = fwhm
        p.inputs.fwhm_z = fwhm
        outs = p.run()
        
    elif dim == 4: # PVC for each individual images/frames
    
        num_frames = hdr['dim'][4]
        
        # create a temporary PVC folder
        in_file_dir = Path(in_file).parent
        PVC_tmp_dir = os.path.join(in_file_dir, 'PVC_tmp')
        os.makedirs(PVC_tmp_dir, exist_ok=True)
        
        # split the input 4D image into 3D images by time
        split_4D_into_frames(infile = in_file, 
                              out_dir = PVC_tmp_dir, 
                              outfile_basename = 'nopvc_Frame')
        
        PVCed_frame_list = []
        for i in range(num_frames):
            print('Frame ', i)
            
            str_i = str(i)
            if len(str_i) == 1:
                str_i = '000' + str_i
            elif len(str_i) == 2:
                str_i = '00' + str_i
            elif len(str_i) == 3:
                str_i = '0' + str_i
            elif len(str_i) == 4:
                pass
            
            nopvc_frame_path = os.path.join(PVC_tmp_dir, f'nopvc_Frame{str_i}.nii.gz')
            pvc_frame_path = os.path.join(PVC_tmp_dir, f'pvc_Frame{str_i}.nii.gz')
            PVCed_frame_list.append(pvc_frame_path)
            
            if not os.path.exists(pvc_frame_path):
                p.inputs.in_file   = nopvc_frame_path
                p.inputs.mask_file = mask_file
                p.inputs.out_file  = pvc_frame_path
                p.inputs.pvc = algo
                p.inputs.fwhm_x = fwhm
                p.inputs.fwhm_y = fwhm
                p.inputs.fwhm_z = fwhm
                outs = p.run()
        
        concatenate_frames(frame_list = PVCed_frame_list, 
                           outfile = out_file)
        
        
        # delete the temporary PVC folder
        if os.path.exists(PVC_tmp_dir):
            shutil.rmtree(PVC_tmp_dir)
        
        
        
    return None
    



