"""
File processing

"""

import os
import shutil
from pathlib import Path
import subprocess
import glob
from . import file_handling as fh

from typing import Callable, Any



def freesurfer_convert(ippath: str,
                       oppath: str) -> None:
    
    
    command = ['mri_convert',
               '-i', ippath,
               '-o', oppath]
    
    subprocess.run(command)
    
    return None





def concatenate_frames(frame_list: list[str],
                       outfile: str) -> None:
    
    """
    Frames must be concatenated in order from frame_list.
    
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
    infile: must be a dynamic (4D) image.
    
    """
        
    command = ['mri_concat', infile, '--mean',
               '--o', outfile]
    
    
    subprocess.run(command)
    
    return None



def weighted_mean_frame_fsl(infile_list: list[str],
                            infile_weights: list[float],
                            outfile: str) -> None:

    """
    Calculate weighted mean of frames. 
    
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
    
    fh.delete_file_list(infile_weighted_list)
    
    return None




def group_images_by_frames_old_todel(img_dir, frame_size, img_suffix):
    
    ## TO UPDATE: do with glob

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(img_dir) if f.endswith(img_suffix)]
    
    # Sort the images by their labels in ascending order
    image_files.sort()
    #print('len(image_files) =', len(image_files))

    # Calculate the number of frames needed
    num_frames = (len(image_files) + frame_size - 1) // frame_size

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




def group_images_by_frames(img_dir: str, 
                           frame_size: int, 
                           img_suffix: str,
                           image_files: list[str] | None = None):
        
    
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


    command = ['mri_average'] + infiles + [outfile]
    
    subprocess.run(command)
    
    return None


def split_4D_into_frames(infile: str,
                         out_dir: str,
                         outfile_basename: str) -> None:
    """
    Split an input 4D image into dynamic frames.
    """

    outfile_basename_full = os.path.join(out_dir, outfile_basename)
    
    command = ['fslsplit', infile, outfile_basename_full]
    
    subprocess.run(command)
    
    return None




def split_filename(filename):
    p = Path(filename)
    # Get all suffixes (e.g., ['.tar', '.gz'] for 'my.archive.tar.gz')
    ext = "".join(p.suffixes)
    # Remove all suffixes by repeatedly calling with_suffix('')
    while p.suffix:
        p = p.with_suffix('')
    return p.name, ext