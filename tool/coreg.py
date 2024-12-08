"""
Co-registration and transform

"""

import os
import subprocess


def freesurfer_transform(
        infile_path: str, 
        lta_path: str, 
        outfile_path: str) -> None:
        
    # Check if the lta_path exists
    assert os.path.isfile(lta_path) is True, f"""The lta_path {lta_path} does not exist."""
    
    command = ['mri_convert',
               '-at',
               lta_path,
               infile_path,
               outfile_path]
    
    #os.system(f'$FREESURFER_HOME/bin/mri_convert -at {lta_path} {infile_path} {outfile_path}')
    subprocess.run(command)
    
    return None




def freesurfer_coreg(mov_path: str,
                     reg_path: str,
                     ref_path: str | None = None,
                     subj_name: str | None = None,
                     Freesurfer_dir: str | None = None,
                     no_ref_mask: bool | None = None,
                     dof: int | None = None,
                     ) -> None:
    """
    If subj_name is set and ref_path is not set, the default ref_path is
    Freesurfer_dir/subj_name/mri/brainmask.mgz. 
    
    Default dof = 6
    """
    
    if subj_name is not None:
        if Freesurfer_dir is None:
            raise ValueError('When subj_name is specified. Freesurfer_dir cannot be None.')
        
        # Freesurfer_dir is not None
        subj_dir = os.path.join(Freesurfer_dir, subj_name)
        if not os.path.exists(subj_dir):
            raise ValueError(f'{subj_dir} does not exist.')
                
    command = ['mri_coreg', 
                '--mov', mov_path,
                '--reg', reg_path] 
    
    if ref_path is not None:
        command += ['--ref', ref_path]
        
    if subj_name is not None:
        command += ['--s', subj_name]
        
    if no_ref_mask is True:
        command += ['--no-ref-mask']
        
    if dof is not None:
        command += ['--dof', str(dof)]
        
    if Freesurfer_dir is None:
        subprocess.run(command)
    else:
        myenv = os.environ.copy()
        myenv['SUBJECTS_DIR'] = Freesurfer_dir
        subprocess.run(command, env=myenv)
    
    return None


def fsl_transform(
        infile_path: str,  
        outfile_path: str,
        reffile_path: str,
        transmat_path: str) -> None:
    """
    reffile is used to determine the size of the outfile volume, but the contents of reffile are NOT used.
    """
        
    command = ['flirt',
               '-in', infile_path,
               '-ref', reffile_path,
               '-out', outfile_path,
               '-init', transmat_path,
               '-applyxfm']
    
    subprocess.run(command)
    
    return None



def fsl_coreg(move_path: str,
              ref_path: str,
              out_path: str | None = None,
              omat_path: str | None = None,
              dof: int | None = None):
    
    """
    Default dof = 12
    """
    
    if out_path is None and omat_path is None:
        raise ValueError('ref_path and out_path cannot both be None')
        
    command = ['flirt',
               '-in', move_path,
               '-ref', ref_path]
    
    if out_path is not None:
        command += ['-out', out_path]
        
    if omat_path is not None:
        command += ['-omat', omat_path]
        
    if dof is not None:
        command += ['-dof', str(dof)]
        
    subprocess.run(command)    
     
    return None



def fsl_decompose_mat_params(in_mat_path: str,
                             out_params_path: str):
    """
    Decompose the 4x4 transformation matrix in .mat file into the 12 DOF
    transformation parameters. 
    """
    
    command = ['avscale', '--allparams', in_mat_path]
    
    result = subprocess.run(command,
                   capture_output = True, 
                   text = True)
    
    with open(out_params_path, "w") as file:
        file.write(result.stdout)
    
    # print(type(result.stdout))
    # print(result.stdout)
    
    return None
    


