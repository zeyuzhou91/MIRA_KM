import sys
# add the path where kinetic_modeling lives
sys.path.append('/Users/zeyuzhou/Documents')

import os
import shutil
import numpy as np
from pathlib import Path
import glob
import argparse

import MIRA_KM.tool.file_handling as fh
from MIRA_KM.subject import Cohort
from MIRA_KM.tool.coreg import fsl_coreg
from MIRA_KM.tool.file import concatenate_frames, generate_mean_frame, split_4D_into_frames

    

def motion_correct(motion_correct_dir_name: str,
                   PETimg_name: str,
                   framemean_name: str,
                   PETimg_mc_name: str,
                   framemean_mc_name: str,
                   PET_dir: str,
                   print_message: str):
    
    print(print_message)
    
    PETimg_path = os.path.join(PET_dir, PETimg_name)
    framemean_path = os.path.join(PET_dir, framemean_name)
    PETimg_mc_path = os.path.join(PET_dir, PETimg_mc_name)
    framemean_mc_path = os.path.join(PET_dir, framemean_mc_name)
    
    if (not os.path.exists(PETimg_path)) or (not os.path.exists(framemean_path)):
        print('PET image data incomplete. Cannot do motion correction.')
        return None
    
    if os.path.exists(PETimg_mc_path) and os.path.exists(framemean_mc_path):
        print('PET motion correction already done. Skip.')
        return None
    
    mc_dir = os.path.join(PET_dir, motion_correct_dir_name)
    os.makedirs(mc_dir, exist_ok=True)
    
    # split the input 4D image into 3D images by time
    split_4D_into_frames(infile = PETimg_path, 
                         out_dir = mc_dir, 
                         outfile_basename = 'Frame')
    
    frame_list = glob.glob(os.path.join(mc_dir, 'Frame*.nii.gz'))  # non-ordered
    num_frames = len(frame_list)
    frame_list = []
    for i in range(num_frames):
            
        str_i = str(i)
        if len(str_i) == 1:
            str_i = '000' + str_i
        elif len(str_i) == 2:
            str_i = '00' + str_i
        elif len(str_i) == 3:
            str_i = '0' + str_i
        elif len(str_i) == 4:
            pass
            
        frame_path = os.path.join(mc_dir, f'Frame{str_i}.nii.gz')
        frame_list.append(frame_path)
    
    #print(frame_list)
    
    mc_frame_list = []

    for (i, frame) in enumerate(frame_list):
        
        print(f'Frame {i}')
        
        mc_frame = os.path.join(mc_dir, f'mc_Frame{i}.nii.gz')
        mc_frame_list.append(mc_frame)
        
        if not os.path.exists(mc_frame):

            fsl_coreg(move_path = frame, 
                      ref_path = framemean_path,
                      out_path = mc_frame,
                      dof = 6)
                

    concatenate_frames(frame_list = mc_frame_list, 
                       outfile = PETimg_mc_path)
    print('Frame concatenation done.')
    
    generate_mean_frame(infile = PETimg_mc_path, 
                        outfile = framemean_mc_path)
    print('Mean frame generated.')

    if os.path.exists(mc_dir):
        fh.delete_dir(mc_dir)

    return None



parser = argparse.ArgumentParser(description='GTM segmentation')
parser.add_argument("--num_threads", type=int, metavar='integer', help='Number of threads in parallel processing.', default=1)
parser.add_argument("--thread_idx", type=int, metavar='integer', help='The thread index. 1<= thread_idx <= num_threads.', default=1)
args = parser.parse_args()


cwd = os.getcwd()
#print('cwd =', cwd)
cur_dir = Path(cwd)    
root_dir = cur_dir.parent
os.chdir(root_dir)

subj_list = glob.glob("SHFDG*")
subj_list.sort()

# #priority_cohort = Cohort.from_excel(filepath = os.path.join(root_dir, 'docs/subj_info_priority_all(2024-12-17).xlsx'))
# priority_cohort = Cohort.from_excel(filepath = os.path.join(root_dir, 'docs/subj_info_prepost_priority(2025-01-06).xlsx'))
# #priority_cohort.print_info()
# priority_subj_list = priority_cohort.all_subj_names()
    

N = len(subj_list)
n = N // args.num_threads + 1
for (ii, subj) in enumerate(subj_list):
    
    if args.num_threads == 1:
        pass
    else:
        THREAD_IDX = args.thread_idx - 1
        in_range = False
        if (args.thread_idx < args.num_threads) and (THREAD_IDX * n <= ii <= (THREAD_IDX+1) * n - 1):
            in_range = True
        elif (args.thread_idx == args.num_threads) and (THREAD_IDX * n <= ii <= N-1): 
            in_range = True
        if not in_range:
            continue

        
    print(ii, subj)
    
    # if subj != 'SHFDG079_MRI100123_SUB1512211433_DATE20160331_SESSION2':
    #     continue

    #subj_index = int(subj.split('_')[0][-3:])
    #print(subj_index)

    subj_dir = os.path.join(root_dir, subj)
    PET_dir = os.path.join(subj_dir, 'PET')
    
    motion_correct(motion_correct_dir_name = 'mc_tmp',
                   PETimg_name = 'pet.nomc.nii.gz',
                   framemean_name = 'framemean.nii.gz',
                   PETimg_mc_name = 'pet.mc.nii.gz',
                   framemean_mc_name = 'framemean.mc.nii.gz',
                   PET_dir = PET_dir,
                   print_message = 'Motion Correction: 1st pass')
    
    motion_correct(motion_correct_dir_name = 'mc2_tmp',
                   PETimg_name = 'pet.mc.nii.gz',
                   framemean_name = 'framemean.mc.nii.gz',
                   PETimg_mc_name = 'pet.mc2.nii.gz',
                   framemean_mc_name = 'framemean.mc2.nii.gz',
                   PET_dir = PET_dir,
                   print_message = 'Motion Correction: 2nd pass')

    




