import sys
# add the path where kinetic_modeling lives
sys.path.append('/Users/zeyuzhou/Documents')

import os
from kinetic_modeling.core import Environment, FrameSchedule, TAC, ROI
from kinetic_modeling.subject import Cohort

from kinetic_modeling.model.srtm import SRTM_Model
import setup 
    


if __name__ == "__main__":
    
    cwd = os.getcwd()
    print('cwd =', cwd)
    
    env = Environment('env_variables.txt')
    rois_file = os.path.join(cwd, "rois_without_wm_sample.txt")
    ref_roi_file = os.path.join(cwd, "roi_ref_wm.txt")
    delete_existing_mask_first = False
    delete_existing_tac_first = False
    model_name = 'SRTM'
    
    root_dir = os.path.dirname(cwd)
    
    cohort = Cohort.from_excel(filepath = os.path.join(root_dir, 'docs/subject_info.xlsx'))
    #cohort.print_info()
    #cohort.write_to_excel(filepath = os.path.join(root_dir, 'docs/subject_info_rewrite.xlsx'))
    
    for directory in os.listdir(root_dir):
        if (directory.startswith('VAT')) and os.path.isdir(os.path.join(root_dir, directory)):
            
            subj = directory            
            _ , ID, name_initials, AorB, scan_date = subj.split('_')
        
            if subj != 'VAT_3_KF_A_20211215':
            #if subj != 'VAT_6_JG_B_20231010':
                continue
        
            print(subj)
            print(name_initials, AorB, scan_date)
                        
            env.reset_all()
            setup.setup_env(env, root_dir, subj)
            # env.print_all()
            
            # if not os.path.exists(env.path_dict['PET_dir']):
            #     continue
            
            # Create modeling output folder
            output_dir = os.path.join(env.path_dict['kmresults_dir'], model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            fs = FrameSchedule.from_midpoints(filepath = env.path_dict['framemidpointfile_path'])
            # print(f'fs.mid_points: {fs.mid_points}')
            # print(f'fs.durations: {fs.durations}')
            # print(f'fs.start_points: {fs.start_points}')
            # print(fs.unit)
            
            
            reftac = TAC(is_ref = True,
                          t= fs.mid_points,
                          t_unit = fs.unit)
            
            reftac.read_rois_and_masks_from_file(file_path = ref_roi_file, 
                                                  delete_existing_mask_first = delete_existing_mask_first,
                                                  env = env)            
            reftac.extract_tac(delete_existing_tac_first = delete_existing_tac_first,
                                env = env)
            
            for i in range(reftac.num_elements):
                if reftac.rois[i].name == 'CorticalWM':
                    CorticalWMErode = ROI.from_copy(reftac.rois[i])
                    CorticalWMErode.rename('CorticalWMErode4')                    
                    CorticalWMErode.erode_mask(depth = 4, env = env)
            reftac.add_ROI_with_tac(new_roi = CorticalWMErode, 
                                    delete_existing_tac_first = delete_existing_tac_first,
                                    env = env)
            reftac.remove_ROI_with_tac(name = 'CorticalWM')
        
        
            tacs = TAC(is_ref = False, 
                        t = fs.mid_points,
                        t_unit = fs.unit)
            tacs.read_rois_and_masks_from_file(file_path = rois_file, 
                                                delete_existing_mask_first = delete_existing_mask_first,
                                                env = env)
            tacs.extract_tac(delete_existing_tac_first = delete_existing_tac_first,  
                             env = env)
            
            model = SRTM_Model(reftac = reftac, 
                                tacs = tacs,
                                fitting_func_type = 'R1k2BP')
            
            # model = SRTM_Model(reftac = reftac, 
            #                    tacs = tacs,
            #                    fitting_func_type = 'R1k2pk2a')
            
            model.plot_tac(add_input = True,
                           op_dir = output_dir)
            model.fit()
            #model.fit(initial_guess = (1.0, 0.05, 0.1, 2))
            
            model.print_fitting_results()
            model.plot_tac_with_fitting(add_input = True, 
                                        op_dir = output_dir)
            model.export_fitting_results(op_dir = output_dir)
            


    

    


