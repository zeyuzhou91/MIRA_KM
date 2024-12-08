"""
File processing

"""

import os
import shutil
import subprocess



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



def group_images_by_frames(img_dir, frame_size, img_suffix):
    
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



def create_frames(frames_dir, img_suffix):
    
    # Get a list of the frame directories
    frame_list = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d)) and d.startswith('Frame')]
    
    num_frames = len(frame_list)
    
    for i in range(num_frames):
        
        framefolder_path = os.path.join(frames_dir, f'Frame{i}')
        
        # Find the first image for this frame
        image_files = [f for f in os.listdir(framefolder_path) if f.endswith(img_suffix)]
        image_files.sort()
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