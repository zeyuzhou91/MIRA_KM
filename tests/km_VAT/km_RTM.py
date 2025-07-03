import sys
# add the path where kinetic_modeling lives
sys.path.append('/Users/zeyuzhou/Documents')

import os
from kinetic_modeling.core import Environment, FrameSchedule, TAC

from kinetic_modeling.model.rtm import RTM_Model
import setup 
    


if __name__ == "__main__":
    
    cwd = os.getcwd()
    print('cwd =', cwd)
    
    env = Environment('env_path_names.txt')
    rois_file = os.path.join(cwd, "rois_without_wm.txt")
    ref_roi_file = os.path.join(cwd, "roi_ref_wm.txt")
    
    root_dir = os.path.dirname(cwd)
    
    for directory in os.listdir(root_dir):
        if (directory.startswith('VAT')) and os.path.isdir(os.path.join(root_dir, directory)):
            
            subj = directory            
            _ , ID, name_initials, AorB, scan_date = subj.split('_')
        
            if subj != 'VAT_3_KF_A_20211215':
                continue
            
            print(subj)
            print(name_initials, AorB, scan_date)
            
            env.reset_all()
            setup.setup_env(env, root_dir, subj)
            # env.print_all()
            
            # Create modeling output folder
            model_name = 'RTM'
            output_dir = os.path.join(env.path_dict['kmresults_dir'], model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            fs = FrameSchedule.from_midpoints(filepath = env.path_dict['framemidpointfile_path'])
            # print(f'fs.mid_points: {fs.mid_points}')
            # print(f'fs.durations: {fs.durations}')
            # print(f'fs.start_points: {fs.start_points}')
            # print(fs.unit)
                                
            PETimg_path = os.path.join(env.path_dict['PET_dir'], 'pet.nii.gz')
                        
            reftac = TAC(t = fs.mid_points,
                         t_unit = fs.unit)
            reftac.read_rois_from_file(ref_roi_file, env)
            for i in range(reftac.num_elements):
                if reftac.rois[i].name == 'CorticalWM':
                    reftac.rois[i].erode_masks()
            reftac.extract_tac(PETimg_path = PETimg_path, 
                               PETimg_unit = 'Bq/mL', 
                               tacfile_name_extension = '_avgIntensity.csv', 
                               env = env) 
            
            tacs = TAC(t = fs.mid_points,
                       t_unit = fs.unit)
            tacs.read_rois_from_file(rois_file, env)
            tacs.extract_tac(PETimg_path = PETimg_path, 
                             PETimg_unit = 'Bq/mL', 
                             tacfile_name_extension = '_avgIntensity.csv', 
                             env = env)
            
            
            model = RTM_Model(reftac = reftac, 
                              tacs = tacs)
            
            model.fit()            
            model.print_fitting_results()
            model.plot_tac_and_fitted_tac(opfile_path = os.path.join(output_dir, f'{model_name}.png'))
            model.export_fitting_results(opfile_path = os.path.join(output_dir, f'{model_name}.csv'))
            





