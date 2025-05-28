"""
pet_image_processing.py

This module provides utilities for processing PET imaging data, particularly
related to image/frame conversion, concatenation, averaging, and reorganization.

Dependencies:
- FSL and FreeSurfer command-line tools (e.g., `mri_convert`, `mri_concat`, `fslmaths`, `fslsplit`)

Author: Zeyu Zhou
Date: 2025-05-20
"""

import os
import shutil
from pathlib import Path
import subprocess
import glob
from typing import Callable, Any
from . import filesystem_utils as fu




def freesurfer_convert(ippath: str,
                       oppath: str) -> None:
    """
    Convert an image using FreeSurfer's mri_convert.

    Parameters:
    -----------
    ippath : str
        Input file path.
    oppath : str
        Output file path.
    """
    
    command = ['mri_convert',
               '-i', ippath,
               '-o', oppath]
    
    subprocess.run(command)
    
    return None





def concatenate_frames(frame_list: list[str],
                       outfile: str) -> None:
    
    """
    Concatenate multiple frames into a 4D volume using FreeSurfer's mri_concat.
    
    Parameters:
    -----------
    frame_list : list of str
        Ordered list of frame file paths to concatenate.
    outfile : str
        Path to the output 4D file.
    """
    
    command = ['mri_concat']
    
    for frame in frame_list:
        command += [frame]
    
    command += ['--o', outfile]
    
    subprocess.run(command)
    
    return None



def generate_mean_frame(infile: str,
                        outfile: str) -> None:
    """
    Generate mean frame from a 4D dynamic image using FreeSurfer's mri_concat.
    
    Parameters:
    -----------
    infile : str
        Input 4D image path.
    outfile : str
        Output mean frame file path.
    """
        
    command = ['mri_concat', infile, '--mean',
               '--o', outfile]
    
    
    subprocess.run(command)
    
    return None



def weighted_mean_frame_fsl(infile_list: list[str],
                            infile_weights: list[float],
                            outfile: str) -> None:
    """
    Compute the weighted mean of multiple PET frames using FSL's fslmaths.
    
    Parameters:
    -----------
    infile_list : list of str
        List of input frame files.
    infile_weights : list of float
        List of weights for each input file.
    outfile : str
        Output file path for the weighted mean.
    """
    
    
    if len(infile_list) != len(infile_weights):
        raise ValueError('Length of infile_list must equal to length of infile_weights.')
    
    
    # calculate weighted images
    infile_weighted_list = []
    tot_weight = 0.0
    for (infile, weight) in zip(infile_list, infile_weights):
        
        infile_fullname = Path(infile).name
        infile_parent = Path(infile).parent
        
        infile_name, ext = split_filename(infile_fullname)
        infile_weighted_name = infile_name + '_weighted'
        infile_weighted_fullname = infile_weighted_name + ext
        infile_weighted_path = os.path.join(infile_parent, infile_weighted_fullname)
        
        command = ['fslmaths', infile, '-mul', str(weight), infile_weighted_path]
        subprocess.run(command)
        
        infile_weighted_list.append(infile_weighted_path)
        tot_weight += weight
        
    # sum up the weighted images
    command = ['fslmaths']
    for (j, infile_weighted) in enumerate(infile_weighted_list):
        if j == 0:
            command += [infile_weighted]
        else:
            command += ['-add', infile_weighted]
        
    command += ['-div', str(tot_weight), outfile]    
    subprocess.run(command)
    
    fu.delete_file_list(infile_weighted_list)
    
    return None





def group_images_by_frames(img_dir: str, 
                           frame_size: int, 
                           img_suffix: str,
                           image_files: list[str] | None = None):
    """
    Group image files into subdirectories (Frame0, Frame1, ...) based on frame size.
    
    Parameters:
    -----------
    img_dir : str
        Directory containing the image files.
    frame_size : int
        Number of images per frame.
    img_suffix : str
        Suffix to filter image files.
    image_files : list of str, optional
        Pre-filtered and sorted list of image files. If None, it will be generated from img_dir.
    """        
    
    if image_files is not None:
        pass
    else:
        # Get a list of all image files in the input folder
        image_files = glob.glob(os.path.join(img_dir, f'*{img_suffix}'))
        
        # Sort the images by their labels in ascending order
        image_files.sort()
        #print('len(image_files) =', len(image_files))
    
    # for f in image_files:
    #     print(f)

    # Calculate the number of frames needed
    num_frames = (len(image_files) + frame_size - 1) // frame_size
    
    print(f'num_frames: {num_frames}')

    # Iterate through each frame
    for i in range(num_frames):
        start_index = i * frame_size
        end_index = min((i + 1) * frame_size, len(image_files)) 
        print(f'start: {start_index}, end: {end_index}')

        # Create a sub-folder for the frame
        frame_path = os.path.join(img_dir, f'Frame{i}')
        os.makedirs(frame_path, exist_ok=True)

        # Move the corresponding images to the sub-folder
        for j in range(start_index, end_index):
            source_path = os.path.join(img_dir, image_files[j])
            dest_path = os.path.join(frame_path, image_files[j])
            shutil.move(source_path, dest_path)
            
    return None




def create_frames(frames_dir: str, 
                  img_suffix: str | None = None,
                  img_prefix: str | None = None,
                  sort_key: Callable[[str], Any] | None = None):
    """
    Convert each 'Frame' subdirectory in frames_dir into a single NIfTI volume using FreeSurfer.
    
    Parameters:
    -----------
    frames_dir : str
        Directory containing 'Frame*' subfolders.
    img_suffix : str, optional
        Only include files ending with this suffix.
    img_prefix : str, optional
        Only include files starting with this prefix.
    sort_key : callable, optional
        Custom key function for sorting filenames.
    """
    
    # Get a list of the frame directories
    frame_list = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d)) and d.startswith('Frame')]
    
    num_frames = len(frame_list)
    
    for i in range(num_frames):
        
        print("====================")
        print(f'Frame {i}')
        print("====================")
        
        framefolder_path = os.path.join(frames_dir, f'Frame{i}')
        
        # Find the first image for this frame
        if img_suffix is not None:
            image_files = [f for f in os.listdir(framefolder_path) if f.endswith(img_suffix)]
        elif img_prefix is not None:
            image_files = [f for f in os.listdir(framefolder_path) if f.startswith(img_prefix)]
        else:
            image_files = [f for f in os.listdir(framefolder_path)]
            
        image_files.sort(key = sort_key)
        first_image = image_files[0]
        
        # Create the path for this first image
        dicom_path = os.path.join(framefolder_path, first_image)
        
        # Create the path for the output PET frame in .nii
        framenii_path = os.path.join(frames_dir, f'Frame{i}.nii.gz')
        
        # Convert the dicom images into a .nii frame
        command = ['mri_convert', 
                   '-it', 'dicom', 
                   dicom_path, framenii_path]
        
        subprocess.run(command)

    return None



def generate_mean_image(infiles: list[str],
                        outfile: str) -> None:
    """
    Generate the mean image from a list of input images using FreeSurfer's mri_average.
    
    Parameters:
    -----------
    infiles : list of str
        List of image paths.
    outfile : str
        Output file path.
    """

    command = ['mri_average'] + infiles + [outfile]
    
    subprocess.run(command)
    
    return None


def split_4D_into_frames(infile: str,
                         out_dir: str,
                         outfile_basename: str) -> None:
    """
    Split a 4D image into individual frames using FSL's fslsplit.
    
    Parameters:
    -----------
    infile : str
        Path to the 4D image.
    out_dir : str
        Output directory to store the frames.
    outfile_basename : str
        Base name for output files.
    """

    outfile_basename_full = os.path.join(out_dir, outfile_basename)
    
    command = ['fslsplit', infile, outfile_basename_full]
    
    subprocess.run(command)
    
    return None




def split_filename(filename):
    """
    Split a filename into base name and combined suffix (handles multi-part suffixes).

    Parameters:
    -----------
    filename : str
        Full filename (e.g., 'image.nii.gz')

    Returns:
    --------
    name : str
        Base name without extensions.
    ext : str
        Combined extension (e.g., '.nii.gz')
    """    
    
    p = Path(filename)
    # Get all suffixes (e.g., ['.tar', '.gz'] for 'my.archive.tar.gz')
    ext = "".join(p.suffixes)
    # Remove all suffixes by repeatedly calling with_suffix('')
    while p.suffix:
        p = p.with_suffix('')
    return p.name, ext