"""
motion_correction.py

Functions related to motion correction for PET image frames.

Includes:
- mc_wrt_frame1(): Motion correction using edge amplification strategy.
- mc_wrt_frame2(): Motion correction using normalization scaling strategy.
- mc_through_matlab_canny_for_sequential_frames(): Sequential frame correction using Canny-based alignment.

Author: Zeyu Zhou
Date: 2025-05-21
"""


import os
import numpy as np
import shutil
from . import smooth
from . import math as km_math
from . import edge_detection as ed

            




def mc_wrt_frame1(
        refid: int,
        toreg_ids: list[int],
        basename: str,
        PET_dir: str,
        del_middle_folder: bool, 
        matlab_dir: str, 
        approxcanny_thresh: float,
        gaussian_filter_sigma: float,
        q_percentile: float) -> None:
    """
    Perform motion correction relative to a reference frame using enhanced images.
    Enhancement = smoothed nonnegative image + amplified edge image (scaled by percentile).
    
    Parameters:
    - refid: Index of the reference frame.
    - toreg_ids: List of frame indices to be registered.
    - basename: Frame base filename (e.g., 'Frame' for 'Frame10.nii.gz').
    - PET_dir: Directory containing the input frames.
    - del_middle_folder: Whether to delete the intermediate folder.
    - matlab_dir: Path to directory containing approxcanny.m.
    - approxcanny_thresh: High threshold for edge detection (low is 0.4×high).
    - gaussian_filter_sigma: Sigma for Gaussian smoothing.
    - q_percentile: Percentile value used to scale edge maps.
    
    Returns:
    - None. Writes corrected files and optionally deletes intermediate folder.
    """
    
    # Create a folder to store mc middle files
    mc_middle_dir = os.path.join(PET_dir, "mc_middle_files")
    os.makedirs(mc_middle_dir, exist_ok=True)
    
    # Creating enhanced images
    print("=====================================================================================")
    print("Creating enhanced images")
    print("Output: _enhanced.nii.gz files in mc_middle_files")
    print("=====================================================================================")
    all_ids = [refid] + toreg_ids
    for i in all_ids:
        print(f"{basename} {i}")
        
        # Creating edges
        infile = os.path.join(PET_dir, f"{basename}{i}.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_edges.nii.gz")
        ed.matlab_approxcanny(thresh = approxcanny_thresh,
                                   sigma = gaussian_filter_sigma,
                                   infilepath = infile,
                                   outfilepath = outfile,
                                   matlab_dir = matlab_dir)
        
        # Smoothing by Gaussian
        infile = os.path.join(PET_dir, f"{basename}{i}.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_smooth.nii.gz")
        smooth.gaussian_filter_3D(sigma = gaussian_filter_sigma, 
                                  ippath = infile, 
                                  oppath = outfile)
        
        # Thresholding below
        infile = os.path.join(mc_middle_dir, f"{basename}{i}_smooth.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_nonnegative.nii.gz")
        km_math.thresholding(lb = 0.0,
                             ub = np.inf,
                             ippath = infile,
                             oppath = outfile)
        
        # Finding q-percentile value in image
        infile = os.path.join(PET_dir, f"{basename}{i}.nii.gz")
        q_value = km_math.percentile_value_in_image(q = q_percentile,
                                                    ippath = infile)
        
        # Amplifying edges
        infile = os.path.join(mc_middle_dir, f"{basename}{i}_edges.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_edges_amplified.nii.gz")
        km_math.multiply(
                value = q_value,
                ippath = infile,
                oppath = outfile)
        
        # Creating enhanced images: thresholded image + edge image
        infile1 = os.path.join(mc_middle_dir, f"{basename}{i}_nonnegative.nii.gz")
        infile2 = os.path.join(mc_middle_dir, f"{basename}{i}_edges_amplified.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_enhanced.nii.gz")
        km_math.add(
            ippath1 = infile1,
            ippath2 = infile2,
            oppath = outfile)
    
    # Registration    
    print("========================================================")
    print("Registration ...")
    print("Output: _regto?.nii.gz and .mat files in mc_middle_files")
    print("========================================================")
    reffile = os.path.join(mc_middle_dir, f"{basename}{refid}_enhanced.nii.gz")
    for i in toreg_ids:
        print(f"{basename} {i} to {basename} {refid}")
        infile = os.path.join(mc_middle_dir, f"{basename}{i}_enhanced.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_enhanced_regto{refid}.nii.gz")
        matfile = os.path.join(mc_middle_dir, f"{basename}{i}to{refid}.mat")
        os.system(f'flirt -in {infile} -ref {reffile} -out {outfile} -omat {matfile} -dof 6')
    
    # Apply transformation matrices to original images
    print("=======================================================")
    print("Applying transformation matrices to original images ...")
    print("Output: _regto?.nii.gz files in original/raw image folder")
    print("=======================================================")
    # reffile is used to determine the size of the outfile volume, but the contents of reffile are NOT used
    reffile = os.path.join(PET_dir, f'{basename}{refid}.nii.gz')
    for i in toreg_ids:
        print(f"{basename} {i}")
        infile = os.path.join(PET_dir, f'{basename}{i}.nii.gz')
        outfile = os.path.join(PET_dir, f'{basename}{i}_mc.nii.gz')
        transmat = os.path.join(mc_middle_dir, f'{basename}{i}to{refid}.mat')

        os.system(f'flirt -in {infile} -ref {reffile} -out {outfile} -init {transmat} -applyxfm')
        
        
    if del_middle_folder:
        shutil.rmtree(mc_middle_dir)
    
    return None



def mc_wrt_frame2(
        refid: int,
        toreg_ids: list[int],
        basename: str,
        PET_dir: str,
        del_middle_folder: bool, 
        matlab_dir: str, 
        approxcanny_thresh: float,
        gaussian_filter_sigma: float,
        q_percentile: float) -> None:
    """
    Perform motion correction relative to a reference frame using a scaled smoothing strategy.
    Enhancement = normalized smoothed image + edge image.
    
    Parameters:
    - refid: Index of the reference frame.
    - toreg_ids: List of frame indices to be registered.
    - basename: Frame base filename (e.g., 'Frame' for 'Frame10.nii.gz').
    - PET_dir: Directory containing the input frames.
    - del_middle_folder: Whether to delete the intermediate folder.
    - matlab_dir: Path to directory containing approxcanny.m.
    - approxcanny_thresh: High threshold for edge detection.
    - gaussian_filter_sigma: Sigma for Gaussian smoothing.
    - q_percentile: Percentile value used to normalize image intensity.
    
    Returns:
    - None. Writes corrected files and optionally deletes intermediate folder.
    """
    
    # Create a folder to store mc middle files
    mc_middle_dir = os.path.join(PET_dir, "mc_middle_files")
    os.makedirs(mc_middle_dir, exist_ok=True)
    
    # Creating enhanced images
    print("=====================================================================================")
    print("Creating enhanced images")
    print("Output: _enhanced.nii.gz files in mc_middle_files")
    print("=====================================================================================")
    all_ids = [refid] + toreg_ids
    for i in all_ids:
        print(f"{basename} {i}")
        
        # Creating edges
        infile = os.path.join(PET_dir, f"{basename}{i}.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_edges.nii.gz")
        ed.matlab_approxcanny(thresh = approxcanny_thresh,
                                   sigma = gaussian_filter_sigma,
                                   infilepath = infile,
                                   outfilepath = outfile,
                                   matlab_dir = matlab_dir)
        
        # Finding q-percentile value in image
        infile = os.path.join(PET_dir, f"{basename}{i}.nii.gz")
        q_value = km_math.percentile_value_in_image(q = q_percentile,
                                                    ippath = infile)        
        
        # Scaling the original image
        infile = os.path.join(PET_dir, f"{basename}{i}.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_scaled.nii.gz")
        km_math.multiply(
                value = 1.0 / q_value,
                ippath = infile,
                oppath = outfile)
        
        # Smoothing by Gaussian
        infile = os.path.join(mc_middle_dir, f"{basename}{i}_scaled.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_smooth.nii.gz")
        smooth.gaussian_filter_3D(sigma = gaussian_filter_sigma, 
                                  ippath = infile, 
                                  oppath = outfile)

        # Thresholding below
        infile = os.path.join(mc_middle_dir, f"{basename}{i}_smooth.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_nonnegative.nii.gz")
        km_math.thresholding(lb = 0.0,
                             ub = np.inf,
                             ippath = infile,
                             oppath = outfile)
        
        # Creating enhanced images: scaled smoothed nonnegative image + edge image
        infile1 = os.path.join(mc_middle_dir, f"{basename}{i}_nonnegative.nii.gz")
        infile2 = os.path.join(mc_middle_dir, f"{basename}{i}_edges.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_enhanced.nii.gz")
        km_math.add(
            ippath1 = infile1,
            ippath2 = infile2,
            oppath = outfile)
    
    # Registration    
    print("========================================================")
    print("Registration ...")
    print("Output: _regto?.nii.gz and .mat files in mc_middle_files")
    print("========================================================")
    reffile = os.path.join(mc_middle_dir, f"{basename}{refid}_enhanced.nii.gz")
    for i in toreg_ids:
        print(f"{basename} {i} to {basename} {refid}")
        infile = os.path.join(mc_middle_dir, f"{basename}{i}_enhanced.nii.gz")
        outfile = os.path.join(mc_middle_dir, f"{basename}{i}_enhanced_regto{refid}.nii.gz")
        matfile = os.path.join(mc_middle_dir, f"{basename}{i}to{refid}.mat")
        os.system(f'flirt -in {infile} -ref {reffile} -out {outfile} -omat {matfile} -dof 6')
    
    # Apply transformation matrices to original images
    print("=======================================================")
    print("Applying transformation matrices to original images ...")
    print("Output: _regto?.nii.gz files in original/raw image folder")
    print("=======================================================")
    # reffile is used to determine the size of the outfile volume, but the contents of reffile are NOT used
    reffile = os.path.join(PET_dir, f'{basename}{refid}.nii.gz')
    for i in toreg_ids:
        print(f"{basename} {i}")
        infile = os.path.join(PET_dir, f'{basename}{i}.nii.gz')
        outfile = os.path.join(PET_dir, f'{basename}{i}_regto{refid}.nii.gz')
        transmat = os.path.join(mc_middle_dir, f'{basename}{i}to{refid}.mat')

        os.system(f'flirt -in {infile} -ref {reffile} -out {outfile} -init {transmat} -applyxfm')
        
        
    if del_middle_folder:
        shutil.rmtree(mc_middle_dir)
    
    return None



# This function does not work as expected, as it causes intra-frame errors to 
# accumulate. Nor is this a common practice in literature. DO NOT USE IT. 
# Use the _wrt_frame function above
def mc_through_matlab_canny_for_sequential_frames(
        startid: int,
        endid: int,
        basename: str,
        PET_dir: str,
        del_middle_folder: bool, 
        matlab_dir: str, 
        approxcanny_thresh: float,
        gaussian_filter_sigma: float) -> None:
    
    """
    Apply sequential motion correction to a series of frames using Canny edge alignment.
    Each frame is registered to its immediate predecessor. 
    
    [This function does not work as expected, as it causes intra-frame errors to 
    accumulate. Nor is this a common practice in literature. DO NOT USE IT. 
    Use the _wrt_frame function above.]
    
    Steps: for each frame 
        1. Create the edges of this frame and the previous (motion corrected) frame
        2. Register/motion correct this frame's edges with respect to the previous frame's edges,  producing a transformation matrix
        3. Apply the transformation matrix to the frame
    
    Parameters:
    - startid: Starting frame index.
    - endid: Ending frame index.
    - basename: Frame base filename (e.g., 'Frame' for 'Frame10.nii.gz').
    - PET_dir: Path to image directory.
    - del_middle_folder: If True, delete temporary folder after execution.
    - matlab_dir: Directory containing approxcanny.m.
    - approxcanny_thresh: Threshold for edge detection.
    - gaussian_filter_sigma: Sigma for Gaussian smoothing.
    
    Returns:
    - None. Writes motion-corrected images. 
    """  
    
    # Create a folder to store mc middle files
    mc_middle_dir = os.path.join(PET_dir, "mc_middle_files")
    os.makedirs(mc_middle_dir, exist_ok=True)
    
    # Make a copy of the first frame as a mc version
    infile = os.path.join(PET_dir, f"{basename}{startid}.nii.gz")
    outfile = os.path.join(PET_dir, f"{basename}{startid}.mc.nii.gz")
    shutil.copyfile(infile, outfile)
    
    for i in range(startid+1, startid+3):
        print("============================")
        print(f"Motion correcting {basename} {i} ...")
        print("============================")
        
        # Create edges of the previous (motion corrected) frame
        infilename = f"{basename}{i-1}.mc.nii.gz"
        outfilename = f"{basename}{i-1}_edges.mc.nii.gz"
        print(f"{infilename} -> mc_middle_dir/{outfilename} ... ")
        infile = os.path.join(PET_dir, infilename)
        outfile = os.path.join(mc_middle_dir, outfilename)
        ed.matlab_approxcanny(thresh = approxcanny_thresh,
                                   sigma = gaussian_filter_sigma,
                                   infilepath = infile,
                                   outfilepath = outfile,
                                   matlab_dir = matlab_dir)
        
        # Create edges of the current (un-motion corrected) frame
        infilename = f"{basename}{i}.nii.gz"
        outfilename = f"{basename}{i}_edges.nii.gz"
        print(f"{infilename} -> mc_middle_dir/{outfilename} ... ")
        infile = os.path.join(PET_dir, infilename)
        outfile = os.path.join(mc_middle_dir, outfilename)
        ed.matlab_approxcanny(thresh = approxcanny_thresh,
                                   sigma = gaussian_filter_sigma,
                                   infilepath = infile,
                                   outfilepath = outfile,
                                   matlab_dir = matlab_dir)
        
        # Registrition
        reffilename = f"{basename}{i-1}_edges.mc.nii.gz"
        infilename = f"{basename}{i}_edges.nii.gz"
        outfilename = f"{basename}{i}_edges_tmp.mc.nii.gz"
        matfilename = f"{basename}{i}to{i-1}.mat"
        print(f"Registering mc_middle_dir/{infilename} onto mc_middle_dir/{reffilename} to produce mc_middle_dir/{outfilename}")
        print(f"Transformation matrix: mc_middle_dir/{matfilename}")
        reffile = os.path.join(mc_middle_dir, reffilename)
        infile = os.path.join(mc_middle_dir, infilename)
        outfile = os.path.join(mc_middle_dir, outfilename)
        matfile = os.path.join(mc_middle_dir, matfilename)
        os.system(f'flirt -in {infile} -ref {reffile} -out {outfile} -omat {matfile} -dof 6')
        
        # Apply transformation to un-motion corrected frame
        # Apply transformation matrices to original images
        # reffile is used to determine the size of the outfile volume, but the contents of reffile are NOT used
        reffilename = f'{basename}{i-1}.mc.nii.gz'
        infilename = f'{basename}{i}.nii.gz'
        outfilename = f'{basename}{i}.mc.nii.gz'
        matfilename = f'{basename}{i}to{i-1}.mat'
        print(f"Applying mc_middle_dir/{matfilename} to {infilename} to produce {outfilename}")       
        reffile = os.path.join(PET_dir, reffilename)
        infile = os.path.join(PET_dir, infilename)
        outfile = os.path.join(PET_dir, outfilename)
        matfile = os.path.join(mc_middle_dir, matfilename)
        os.system(f'flirt -in {infile} -ref {reffile} -out {outfile} -init {matfile} -applyxfm')
            
        
    if del_middle_folder:
        shutil.rmtree(mc_middle_dir)
    
    return None
    