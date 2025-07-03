import os




def setup_env(env, root_dir: str, subj: str):
    """
    env: an Environment object
    """
    
    _ , ID, name_initials, AorB, scan_date = subj.split('_')
    
    env.path_dict['root_dir'] = root_dir
    
    env.path_dict['subj_dir'] = os.path.join(env.path_dict['root_dir'], subj)
    
    env.path_dict['MRI_dir'] = os.path.join(env.path_dict['subj_dir'], 'Freesurfer/bert1/mri')
    env.path_dict['PET_dir'] = os.path.join(env.path_dict['subj_dir'], 'PET')
    env.path_dict['blood_dir'] = os.path.join(env.path_dict['subj_dir'], 'blood')
    env.path_dict['kmresults_dir'] = os.path.join(env.path_dict['subj_dir'], 'km_results')

    env.path_dict['MR_masks_dir'] = os.path.join(env.path_dict['subj_dir'], 'Freesurfer/bert1/masks')
    env.path_dict['lta_dir'] = os.path.join(env.path_dict['subj_dir'], 'Freesurfer/bert1/lta')
    
    env.path_dict['PET_masks_dir'] = os.path.join(env.path_dict['PET_dir'], 'masks')
    env.path_dict['tacs_dir'] = os.path.join(env.path_dict['PET_dir'], 'tacs')
    
    env.path_dict['seg_path'] = os.path.join(env.path_dict['MRI_dir'], 'apas+head.mgz')
    env.path_dict['mr2pet_lta_path'] = os.path.join(env.path_dict['lta_dir'], 'pet2mr.reg.lta')
    env.path_dict['framemidpointfile_path'] = os.path.join(env.path_dict['PET_dir'], f'{name_initials}_{AorB}.actual.midtime.csv')
    
    env.path_dict['blood_ptac_path'] = os.path.join(env.path_dict['blood_dir'], 'plasma_tac.csv')
    env.path_dict['blood_pif_path'] = os.path.join(env.path_dict['blood_dir'], 'plasma_intact_fraction.csv')
    env.path_dict['blood_p2wb_ratio_path'] = os.path.join(env.path_dict['blood_dir'], 'plasma2wb_conc_ratio.csv')
    
    #env.path_dict['PETimg_path'] = os.path.join(env.path_dict['PET_dir'], 'petpvc.nii.gz')
    env.path_dict['PETimg_unit'] = 'Bq/mL'
    
    env.path_dict['mask_domain'] = 'MR'
    
    return None



