import sys
# add the path where kinetic_modeling lives
sys.path.append('/Users/zeyuzhou/Documents')

import os
import numpy as np
from pathlib import Path
import glob

from kinetic_modeling.core import Environment, FrameSchedule, TAC, ROI
from kinetic_modeling.subject import Cohort

from kinetic_modeling.model.loganrtm import LoganRTM_Model
from kinetic_modeling.model.srtm import SRTM_Model
import setup 
    




if __name__ == "__main__":
    
    cwd = os.getcwd()
    print('cwd =', cwd)
        
    env = Environment('env_variables.txt')
    rois_file = os.path.join(cwd, "rois_without_wm_complete.txt")
    #rois_file = os.path.join(cwd, "rois_without_wm_sample.txt")
    ref_roi_file = os.path.join(cwd, "roi_ref_wm.txt")
    delete_existing_mask_first = False
    delete_existing_tac_first = False
    PVCorNOPVC = ''
    #PVCorNOPVC = 'pvc_IY'
    PETimg = 'petpvc.nii.gz'
    #PETimg = 'pet2mr.nii.gz'
    model_name = 'LoganRTM'
    
    
    cur_dir = Path(cwd)    
    root_dir = cur_dir.parent
    os.chdir(root_dir)
    subj_list = glob.glob("VAT_*")
    
    
    cohort = Cohort.from_excel(filepath = os.path.join(root_dir, 'docs/subject_info.xlsx'))
    cohort.add_header('k2p')
    tacs_tmp = TAC()
    tacs_tmp.read_rois_from_file(file_path = rois_file)
    rois_tmp = tacs_tmp.roi_names()
    cohort.add_header(rois_tmp)
    #cohort.print_info()
    
    
    for subj in subj_list:
                       
        _ , ID, name_initials, AorB, scan_date = subj.split('_')
    
        # if subj in ['VAT_3_KF_A_20211215', 'VAT_2_MA_A_20211116']:
        # if subj in ['VAT_2_MA_B_20230512', 'VAT_4_ND_A_20211216']:    
        # if subj != 'VAT_1_MH_A_20211012':
        # if subj != 'VAT_6_JG_B_20231010':
        #     continue
        
        print(subj)
        print(name_initials, AorB, scan_date)
        
        env.reset_all()
        setup.setup_env(env, root_dir, subj)
        env.path_dict['PETimg_path'] = os.path.join(env.path_dict['PET_dir'], PETimg)
        # env.print_all()
        
        # Create modeling output folder
        output_dir = os.path.join(env.path_dict['kmresults_dir'], model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        
        if not os.path.exists(env.path_dict['framemidpointfile_path']):
            print('framemidpointfile does not exist')
            continue
        fs = FrameSchedule.from_midpoints(filepath = env.path_dict['framemidpointfile_path'])
        # print(f'fs.mid_points: {fs.mid_points}')
        # print(f'fs.durations: {fs.durations}')
        # print(f'fs.start_points: {fs.start_points}')
        # print(fs.unit)
                            
        reftac = TAC(is_ref = True,
                      t= fs.mid_points,
                      t_unit = fs.unit)
        reftac.construct_from_rois_file(file_path = ref_roi_file, 
                                        delete_existing_mask_first = delete_existing_mask_first, 
                                        delete_existing_tac_first = delete_existing_tac_first, 
                                        env = env)
        
        for i in range(reftac.num_elements):
            if reftac.rois[i].name == 'CorticalWM':
                CorticalWMErode = ROI.from_copy(reftac.rois[i])
                CorticalWMErode.rename('CorticalWMErode5')                    
                CorticalWMErode.erode_mask(depth = 5, env = env)
                
                # CorticalWMErode.rename('CorticalWMErodeODonell2024')                    
                # CorticalWMErode.erode_mask_ODonell2024(env = env,
                #                                         fwhm = 9.0,
                #                                         threshold = 0.95)
        reftac.add_ROI_with_tac(new_roi = CorticalWMErode, 
                                delete_existing_tac_first = delete_existing_tac_first,
                                env = env)
        reftac.remove_ROI_with_tac(name = 'CorticalWM')
        
        
        tacs = TAC(is_ref = False, 
                    t = fs.mid_points,
                    t_unit = fs.unit)
        tacs.construct_from_rois_file(file_path = rois_file, 
                                      delete_existing_mask_first = delete_existing_mask_first, 
                                      delete_existing_tac_first = delete_existing_tac_first, 
                                      env = env)
        
        srtm = SRTM_Model(reftac = reftac, 
                          tacs = tacs,
                          fitting_func_type = 'R1k2BP')
        srtm.plot_tac(tissues = ['all'], 
                      add_input = True,
                      op_dir = output_dir,
                      op_name_suffix = f'_{PVCorNOPVC}.png')
        if subj == 'VAT_4_ND_A_20211216':
            srtm.fit(special_tissues = ['Amygdala'],
                     special_p0 = [2.5, 0.1, 1.0],
                     debug = True)
        else:
            srtm.fit()
        
        # # srtm.print_fitting_results()
        srtm.plot_tac_with_fitting(tissues = ['all'], 
                                    add_input = True,
                                    op_dir = output_dir,
                                    op_name_suffix = f'_{PVCorNOPVC}.png')
        srtm.export_fitting_results(op_dir = output_dir,
                                    op_name_suffix = f'_{PVCorNOPVC}.csv')
        
        k2p = srtm.get_parameter('k2p')
        BPND_srtm = srtm.get_parameter('BPND')
        Rsquared = srtm.Rsquared
        
        k2p_valid = []
        for k, R, bpnd in zip(k2p, Rsquared, BPND_srtm):
            if R > 0.90 and bpnd > 0:
                k2p_valid.append(k)
        
        if k2p_valid == []:
            raise ValueError('k2p_valid is empty')
        
        k2p_median = np.median(k2p_valid)
        cohort.assign_value(subj = subj, 
                            header = 'k2p', 
                            value = k2p_median)

        loganrtm = LoganRTM_Model(reftac = reftac, 
                                  tacs = tacs, 
                                  k2p = k2p_median)
        
        loganrtm.plot_tac(tissues = ['all'],
                          add_input = True,
                          op_dir = output_dir,
                          op_name_suffix = f'_{PVCorNOPVC}.png')
        
        loganrtm.fit(t0 = 30)
        
        # loganrtm.print_fitting_results()
        # loganrtm.plot_fitting(tissue_names = ['all'],
        #                     op_dir = output_dir)
        loganrtm.export_fitting_results(op_dir = output_dir,
                                        op_name_suffix = f'_{PVCorNOPVC}.csv')
        
                
        # Update header value
        for i in range(tacs.num_elements):
            header = loganrtm.tacs.rois[i].name
            value = loganrtm.get_parameter('BPND')[i]
            
            cohort.assign_value(subj = subj, 
                                header = header, 
                                value = value)

    cohort.write_to_excel(filepath = os.path.join(root_dir, f'km/group_analysis_results/{model_name}_{PVCorNOPVC}.xlsx'))
        
            

