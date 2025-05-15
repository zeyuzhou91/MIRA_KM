"""
Functions related to motion correction.
"""


import os
import numpy as np
import shutil
from . import smooth
from . import math as km_math
from . import feature 
from . import smooth

            

# def mc_through_smoothing_wrt_frame(
#         refid: int,
#         toreg_ids: list[int],
#         basename: str,
#         PET_dir: str,
#         del_middle_folder: bool, 
#         thresholding: bool | None = None,
#         threshold_lb: float | None = None,
#         threshold_ub: float | None = None) -> None:
    
#     """
#     Do motion correction to a list of image frames wrt to a given frame. Assume 
#     the images are noisy. So smoothing is needed before motion correction. 
    
#     Steps:
#         1. Smooth the images by Gaussian filter.
#         2. Motion correct the smoothed images, producing transformation matrices. 
#         3. Apply the transformation matrices to the original images. 
    
#     refid: ID of the reference frame
#     toreg_ids: list of IDs of the frames to be registered
#     basename: basename of the frame filenames. e.g. for Frame10.nii.gz, basename is 'Frame'
#     PET_dir: path of the directory where the PET images are stored
#     del_middle_folder: if the mc_middle_files folder will be deleted
#     thresholding (optional): if thresholding will be applied or not
#     threshold_lb (optional): thresholding lower-bound
#     threshold_ub (optional): thresholding upper-bound
#     """
    
#     # Tunable parameters
#     gf_SIGMA = 3    
    
#     # Create a folder to store mc middle files
#     mc_middle_dir = os.path.join(PET_dir, "mc_middle_files")
#     os.makedirs(mc_middle_dir, exist_ok=True)
    
#     # Gaussian filtering
#     print("============================")
#     print("Applying Gaussian filter ...")
#     print("Output: _gf.nii.gz files in mc_middle_files")
#     print("============================")
#     all_ids = [refid] + toreg_ids
#     for i in all_ids:
#         print(f"{basename} {i}")
#         infile = os.path.join(PET_dir, f"{basename}{i}.nii.gz")
#         outfile = os.path.join(mc_middle_dir, f"{basename}{i}_gf.nii.gz")
#         smooth.gaussian_filter_3D(sigma = gf_SIGMA, 
#                                   ippath = infile, 
#                                   oppath = outfile)
    
#     # Thresholding (if requested)
#     if thresholding:
#         print("============================")
#         print("Thresholding ...")
#         print("Output: _gf.nii.gz files in mc_middle_files")
#         print("============================")
#         for i in all_ids:
#             print(f"{basename} {i}")
#             infile = os.path.join(mc_middle_dir, f"{basename}{i}_gf.nii.gz")
#             outfile = os.path.join(mc_middle_dir, f"{basename}{i}_gf.nii.gz")
#             km_math.thresholding(lb = threshold_lb,
#                                   ub = threshold_ub,
#                                   ippath = infile,
#                                   oppath = outfile)
    
#     # Registration    
#     print("============================")
#     print("Registration ...")
#     print("Output: _regto?.nii.gz and .mat files in mc_middle_files")
#     print("============================")
#     reffile = os.path.join(mc_middle_dir, f"{basename}{refid}_gf.nii.gz")
#     for i in toreg_ids:
#         print(f"{basename} {i} to {basename} {refid}")
#         infile = os.path.join(mc_middle_dir, f"{basename}{i}_gf.nii.gz")
#         outfile = os.path.join(mc_middle_dir, f"{basename}{i}_regto{refid}.nii.gz")
#         matfile = os.path.join(mc_middle_dir, f"{basename}{i}to{refid}.mat")
#         os.system(f'flirt -in {infile} -ref {reffile} -out {outfile} -omat {matfile} -dof 6')
    
    
#     # Apply transformation matrices to original images
#     print("=======================================================")
#     print("Applying transformation matrices to original images ...")
#     print("Output: _regto?.nii.gz files in original/raw image folder")
#     print("=======================================================")
#     # reffile is used to determine the size of the outfile volume, but the contents of reffile are NOT used
#     reffile = os.path.join(PET_dir, f'{basename}{refid}.nii.gz')
#     for i in toreg_ids:
#         print(f"{basename} {i}")
#         infile = os.path.join(PET_dir, f'{basename}{i}.nii.gz')
#         outfile = os.path.join(PET_dir, f'{basename}{i}_regto{refid}.nii.gz')
#         transmat = os.path.join(mc_middle_dir, f'{basename}{i}to{refid}.mat')

#         os.system(f'flirt -in {infile} -ref {reffile} -out {outfile} -init {transmat} -applyxfm')
        
        
#     if del_middle_folder:
#         shutil.rmtree(mc_middle_dir)
    
#     return None
    


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
    TO-DO
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
        feature.matlab_approxcanny(thresh = approxcanny_thresh,
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
    TO-DO
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
        feature.matlab_approxcanny(thresh = approxcanny_thresh,
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



# This function does not work as expected, for some reason
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
    Do motion correction to a sequence of frames by using Matlab's (approximate)
    Canny edge detection algorithm.    
    
    Steps: for each frame 
        1. Create the edges of this frame and the previous (motion corrected) frame
        2. Register/motion correct this frame's edges with respect to the previous frame's edges,  producing a transformation matrix
        3. Apply the transformation matrix to the frame
    
    refid: ID of the reference frame
    toreg_ids: list of IDs of the frames to be registered
    basename: basename of the frame filenames. e.g. for Frame10.nii.gz, basename is 'Frame'
    PET_dir: path of the directory where the PET images are stored
    del_middle_folder: if the mc_middle_files folder will be deleted
    matlab_dir: directory path of the Matlab folder, which contains the approxcanny.m file
    approxcanny_thresh: high sensitivity threshold of the Canny algorithm. the low sensitivity threshold is
              set as 0.4*thresh
    gaussian_filter_sigma: standard deviation of the Gaussian smoothing filter. 
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
        feature.matlab_approxcanny(thresh = approxcanny_thresh,
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
        feature.matlab_approxcanny(thresh = approxcanny_thresh,
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
    