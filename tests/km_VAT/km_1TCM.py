
import sys
# add the path where kinetic_modeling lives
sys.path.append('/Users/zeyuzhou/Documents')

import os

from kinetic_modeling.arterial import BloodInput
from kinetic_modeling.core import Environment, FrameSchedule, TAC

from kinetic_modeling.model.onetcm import OneTCM_Model
import setup 
    




if __name__ == "__main__":
    
    cwd = os.getcwd()
    print('cwd =', cwd)
    
    env = Environment('env_path_names.txt')
    rois_file = os.path.join(cwd, "rois.txt")
    
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
            model_name = '1TCM'
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
            
            PETimg_path = os.path.join(env.path_dict['PET_dir'], 'pet.nii.gz')
            
            tacs = TAC(t = fs.mid_points,
                       t_unit = fs.unit)
            tacs.read_rois_from_file(rois_file, env)


            for i in range(tacs.num_elements):
                if tacs.rois[i].name == 'CorticalWM':
                    tacs.rois[i].erode_masks()
                    
            tacs.extract_tac(PETimg_path = PETimg_path, 
                            PETimg_unit = 'Bq/mL', 
                            tacfile_name_extension = '_avgIntensity.csv', 
                            env = env)
            
            model = OneTCM_Model(binput = bloodinput, 
                                 tacs = tacs, 
                                 fitting_func_type = "without_VB")
            model.fit()
            
            model.print_fitting_results()
            model.plot_tac_and_fitted_tac(op_dir = output_dir)
            model.export_fitting_results(op_dir = output_dir)


        



    


