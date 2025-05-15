"""
Partial volume correction
"""

import os
import nibabel as nib
from pathlib import Path
import glob
import shutil
from nipype.interfaces.petpvc import PETPVC
from .file import split_4D_into_frames, concatenate_frames


def seg_3Dto4D(
        ippath: str,
        oppath: str) -> None:
    
    os.system(f'pvc_make4d -i {ippath} -o {oppath}')
    
    # # load input image
    # ip = nib.load(ippath)
    
    # ip_data = ip.get_fdata().astype(int)
    
    # all_IDs = list(set(ip_data.flatten()))
    # all_IDs.sort()
    # print(all_IDs)

    
    # if len(all_IDs) < 4:
        
    #     op_data = None
    
    #     for ID in all_IDs:
            
    #         print(ID)
            
    #         mask = np.isin(ip_data, [ID]).astype(int)
                        
    #         if op_data is None:
    #             # 3D to 4D
    #             op_data = mask[..., np.newaxis]
    #         else:
    #             op_data = np.concatenate((op_data, mask[..., np.newaxis]), axis=-1)
    
    # else:
        
    #     # divide into 4 chunks for quicker processing
        
    #     op_data1 = None
    #     op_data2 = None
    #     op_data3 = None
    #     op_data4 = None
        
    #     N = len(all_IDs) // 4
        
    #     for ID in all_IDs:
            
    #         print(ID)
    #         mask = np.isin(ip_data, [ID]).astype(int)
            
            
    #         if 0 <= ID <= N-1:
                
    #             if op_data1 is None:
    #                 op_data1 = mask[..., np.newaxis]
    #             else:
    #                 op_data1 = np.concatenate((op_data1, mask[..., np.newaxis]), axis=-1)
        
    #         elif N <= ID <= 2*N-1:
                
    #             if op_data2 is None:
    #                 op_data2 = mask[..., np.newaxis]
    #             else:
    #                 op_data2 = np.concatenate((op_data2, mask[..., np.newaxis]), axis=-1)
        
    #         elif 2*N <= ID <= 3*N-1:
                
    #             if op_data3 is None:
    #                 op_data3 = mask[..., np.newaxis]
    #             else:
    #                 op_data3 = np.concatenate((op_data3, mask[..., np.newaxis]), axis=-1)
            
    #         else:
               
    #             if op_data4 is None:
    #                 op_data4 = mask[..., np.newaxis]
    #             else:
    #                 op_data4 = np.concatenate((op_data4, mask[..., np.newaxis]), axis=-1)
               
    #     op_data = np.concatenate((op_data1, op_data2, op_data3, op_data4), axis=-1)       
    
    # # print(type(op_data))
    # print(op_data.shape)
    
    # # Make the output image
    # op = nib.Nifti1Image(op_data, ip.affine, ip.header)
    
    # print('finished making')
            
    # # Save the output image
    # nib.save(op, oppath)
    
    # print('finished saving')
    
    return None



def pvc(in_file: str,
        mask_file: str,
        out_file: str,
        algo: str,
        fwhm: float) -> None:
    
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
    



