"""
registration.py

Functions for image co-registration and spatial transformations using FreeSurfer and FSL tools.
Includes wrapper utilities for `mri_convert`, `mri_coreg`, `flirt`, and `avscale`.

Author: Zeyu Zhou
Date: 2025-05-15
"""

import os
import subprocess


# ==========================
# FreeSurfer Registration
# ==========================

def freesurfer_transform(infile_path: str, lta_path: str, outfile_path: str) -> None:
    """
    Apply a FreeSurfer LTA (linear transform array) to an image using mri_convert.

    Parameters
    ----------
    infile_path : str
        Path to the input image file.
    lta_path : str
        Path to the .lta transform file.
    outfile_path : str
        Path to save the transformed output image.
    """
    assert os.path.isfile(lta_path), f"The LTA file does not exist: {lta_path}"
    
    command = ['mri_convert', '-at', lta_path, infile_path, outfile_path]
    subprocess.run(command)
    
    return None


# To delete: freesurfer_transform
# def freesurfer_transform(
#         infile_path: str, 
#         lta_path: str, 
#         outfile_path: str) -> None:
        
#     # Check if the lta_path exists
#     assert os.path.isfile(lta_path) is True, f"""The lta_path {lta_path} does not exist."""
    
#     command = ['mri_convert',
#                '-at',
#                lta_path,
#                infile_path,
#                outfile_path]
    
#     #os.system(f'$FREESURFER_HOME/bin/mri_convert -at {lta_path} {infile_path} {outfile_path}')
#     subprocess.run(command)
    
#     return None


def freesurfer_coreg(mov_path: str,
                     reg_path: str,
                     ref_path: str | None = None,
                     subj_name: str | None = None,
                     Freesurfer_dir: str | None = None,
                     no_ref_mask: bool | None = None,
                     dof: int | None = None) -> None:
    """
    Run FreeSurfer mri_coreg to compute transformation from moving image to reference.

    Parameters
    ----------
    mov_path : str
        Path to the moving image.
    reg_path : str
        Path to save the output registration .dat file.
    ref_path : str, optional
        Reference image path (default: derived from subj_name if not provided).
    subj_name : str, optional
        FreeSurfer subject name. Required if ref_path is not provided.
    Freesurfer_dir : str, optional
        Path to SUBJECTS_DIR for FreeSurfer.
    no_ref_mask : bool, optional
        If True, disables masking of reference image.
    dof : int, optional
        Degrees of freedom for registration (default: 6).
    """
    if subj_name and not Freesurfer_dir:
        raise ValueError("Freesurfer_dir must be provided if subj_name is used.")

    if subj_name:
        subj_dir = os.path.join(Freesurfer_dir, subj_name)
        if not os.path.exists(subj_dir):
            raise ValueError(f"Subject directory not found: {subj_dir}")

    command = ['mri_coreg', '--mov', mov_path, '--reg', reg_path]

    if ref_path:
        command += ['--ref', ref_path]
    if subj_name:
        command += ['--s', subj_name]
    if no_ref_mask:
        command += ['--no-ref-mask']
    if dof is not None:
        command += ['--dof', str(dof)]

    if Freesurfer_dir:
        env = os.environ.copy()
        env['SUBJECTS_DIR'] = Freesurfer_dir
        subprocess.run(command, env=env)
    else:
        subprocess.run(command)
        
    return None


# To delete: freesurfer_coreg
# def freesurfer_coreg(mov_path: str,
#                      reg_path: str,
#                      ref_path: str | None = None,
#                      subj_name: str | None = None,
#                      Freesurfer_dir: str | None = None,
#                      no_ref_mask: bool | None = None,
#                      dof: int | None = None,
#                      ) -> None:
#     """
#     If subj_name is set and ref_path is not set, the default ref_path is
#     Freesurfer_dir/subj_name/mri/brainmask.mgz. 
    
#     Default dof = 6
#     """
    
#     if subj_name is not None:
#         if Freesurfer_dir is None:
#             raise ValueError('When subj_name is specified. Freesurfer_dir cannot be None.')
        
#         # Freesurfer_dir is not None
#         subj_dir = os.path.join(Freesurfer_dir, subj_name)
#         if not os.path.exists(subj_dir):
#             raise ValueError(f'{subj_dir} does not exist.')
                
#     command = ['mri_coreg', 
#                 '--mov', mov_path,
#                 '--reg', reg_path] 
    
#     if ref_path is not None:
#         command += ['--ref', ref_path]
        
#     if subj_name is not None:
#         command += ['--s', subj_name]
        
#     if no_ref_mask is True:
#         command += ['--no-ref-mask']
        
#     if dof is not None:
#         command += ['--dof', str(dof)]
        
#     if Freesurfer_dir is None:
#         subprocess.run(command)
#     else:
#         myenv = os.environ.copy()
#         myenv['SUBJECTS_DIR'] = Freesurfer_dir
#         subprocess.run(command, env=myenv)
    
#     return None


# ==========================
# FSL Registration
# ==========================

def fsl_transform(infile_path: str,  
                  outfile_path: str,
                  reffile_path: str,
                  transmat_path: str) -> None:
    """
    Apply a precomputed FSL transformation matrix to an image using `flirt -applyxfm`.

    Parameters
    ----------
    infile_path : str
        Input image to be transformed.
    outfile_path : str
        Path to save the output image.
    reffile_path : str
        Reference image for defining output dimensions.
    transmat_path : str
        Transformation matrix (.mat) file from FSL.
    """
    command = [
        'flirt', '-in', infile_path,
        '-ref', reffile_path,
        '-out', outfile_path,
        '-init', transmat_path,
        '-applyxfm'
    ]
    subprocess.run(command)


# To delete: fsl_transform
# def fsl_transform(
#         infile_path: str,  
#         outfile_path: str,
#         reffile_path: str,
#         transmat_path: str) -> None:
#     """
#     reffile is used to determine the size of the outfile volume, but the contents of reffile are NOT used.
#     """
        
#     command = ['flirt',
#                '-in', infile_path,
#                '-ref', reffile_path,
#                '-out', outfile_path,
#                '-init', transmat_path,
#                '-applyxfm']
    
#     subprocess.run(command)
    
#     return None


def fsl_coreg(move_path: str,
              ref_path: str,
              out_path: str | None = None,
              omat_path: str | None = None,
              dof: int | None = None) -> None:
    """
    Perform linear registration using FSL FLIRT.

    Parameters
    ----------
    move_path : str
        Moving image path.
    ref_path : str
        Reference image path.
    out_path : str, optional
        Output image path after transformation.
    omat_path : str, optional
        Output .mat transformation file path.
    dof : int, optional
        Degrees of freedom (default: 12).
    """
    if not out_path and not omat_path:
        raise ValueError("At least one of out_path or omat_path must be provided.")

    command = ['flirt', '-in', move_path, '-ref', ref_path]

    if out_path:
        command += ['-out', out_path]
    if omat_path:
        command += ['-omat', omat_path]
    if dof:
        command += ['-dof', str(dof)]

    subprocess.run(command)


# To delete: fsl_coreg
# def fsl_coreg(move_path: str,
#               ref_path: str,
#               out_path: str | None = None,
#               omat_path: str | None = None,
#               dof: int | None = None):
    
#     """
#     Default dof = 12
#     """
    
#     if out_path is None and omat_path is None:
#         raise ValueError('ref_path and out_path cannot both be None')
        
#     command = ['flirt',
#                '-in', move_path,
#                '-ref', ref_path]
    
#     if out_path is not None:
#         command += ['-out', out_path]
        
#     if omat_path is not None:
#         command += ['-omat', omat_path]
        
#     if dof is not None:
#         command += ['-dof', str(dof)]
        
#     subprocess.run(command)    
     
#     return None


def fsl_decompose_mat_params(in_mat_path: str, out_params_path: str) -> None:
    """
    Decompose a 4x4 FSL transformation matrix (.mat) into its 12 DOF parameters.

    Parameters
    ----------
    in_mat_path : str
        Path to .mat file with affine transformation.
    out_params_path : str
        Path to write decomposed transformation parameters.
    """
    command = ['avscale', '--allparams', in_mat_path]
    result = subprocess.run(command, capture_output=True, text=True)

    with open(out_params_path, 'w') as f:
        f.write(result.stdout)
        
    return None


# To delete: fsl_decompose_mat_params
# def fsl_decompose_mat_params(in_mat_path: str,
#                              out_params_path: str):
#     """
#     Decompose the 4x4 transformation matrix in .mat file into the 12 DOF
#     transformation parameters. 
#     """
    
#     command = ['avscale', '--allparams', in_mat_path]
    
#     result = subprocess.run(command,
#                    capture_output = True, 
#                    text = True)
    
#     with open(out_params_path, "w") as file:
#         file.write(result.stdout)
    
#     # print(type(result.stdout))
#     # print(result.stdout)
    
#     return None
    


