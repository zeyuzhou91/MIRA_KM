

import sys
# add the path where kinetic_modeling lives
sys.path.append('/Users/zeyuzhou/Documents')

import os
import numpy as np
import matplotlib.pyplot as plt

from kinetic_modeling.arterial import BloodInput
from kinetic_modeling.core import Environment, FrameSchedule, TAC, ROI

from kinetic_modeling.model.twotcm import TwoTCM_Model
import setup 
    


if __name__ == "__main__":
    
    cwd = os.getcwd()
    print('cwd =', cwd)
    
    env = Environment('env_path_names.txt')
    rois_file = os.path.join(cwd, "rois.txt")
    delete_existing_masks_first = False
    delete_existing_tacs_first = False
    model_name = '2TCM'
    
    root_dir = os.path.dirname(cwd)
    
    for directory in os.listdir(root_dir):
        if (directory.startswith('VAT')) and os.path.isdir(os.path.join(root_dir, directory)):
            
            subj = directory            
            _ , ID, name_initials, AorB, scan_date = subj.split('_')
        
            if subj != 'VAT_3_KF_A_20211215':
            #if subj != 'VAT_4_ND_A_20211216':
                continue
            
            print(subj)
            print(name_initials, AorB, scan_date)
            
            env.reset_all()
            setup.setup_env(env, root_dir, subj)
            # env.print_all()
            
            # Create modeling output folder

            output_dir = os.path.join(env.path_dict['kmresults_dir'], model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            fs = FrameSchedule.from_midpoints(filepath = env.path_dict['framemidpointfile_path'])
            # print(f'fs.mid_points: {fs.mid_points}')
            # print(f'fs.durations: {fs.durations}')
            # print(f'fs.start_points: {fs.start_points}')
            # print(fs.unit)
        
            assert os.path.exists(env.path_dict['blood_dir']), f"The {model_name} requires blood data, which is not found for subject {subj}."
            
            bloodinput = BloodInput.from_file(env.path_dict['blood_ptac_path'],
                                              env.path_dict['blood_pif_path'],
                                              env.path_dict['blood_p2wb_ratio_path'],
                                              t = fs.mid_points)
            
            #bloodinput.plot(trange=[0,100])
            
            tacs = TAC(is_ref = False,
                       t = fs.mid_points,
                       t_unit = fs.unit)
            tacs.read_rois_and_masks_from_file(file_path = rois_file, 
                                               delete_existing_masks_first = delete_existing_masks_first,
                                               env = env)
            tacs.extract_tac(delete_existing_tacs_first = delete_existing_tacs_first,
                             env = env)

            for i in range(tacs.num_elements):
                if tacs.rois[i].name == 'CorticalWM':
                    CorticalWMErode = ROI.from_copy(tacs.rois[i])
                    CorticalWMErode.rename('CorticalWMErode')                    
                    CorticalWMErode.erode_PET_mask(depth = 2)
            
            tacs.add_ROI_with_tac(new_roi = CorticalWMErode, 
                                  delete_existing_tac_first = delete_existing_tacs_first,
                                  env = env)
            
            model = TwoTCM_Model(binput = bloodinput, 
                                  tacs = tacs, 
                                  fitting_func_type = "without_VB")
            
            model.plot_tac(add_input = False,
                            op_dir = output_dir)
            # model.fit()
            # # model.fit(p0 = (0.2, 0.05, 0.05, 0.01))
            
            # model.print_fitting_results()
            # model.plot_tac_with_fitting(add_input = False, 
            #                             op_dir = output_dir)
            # model.export_fitting_results(op_dir = output_dir)
