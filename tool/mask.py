"""
Functions related to masking. 

"""

import numpy as np
import nibabel as nib
import copy
import os
import shutil
from pathlib import Path
from skimage.morphology import ball, binary_dilation, binary_erosion
from . import aux  # NOTE: simply "import aux" won't work

from .math import gaussian_fwhm2sigma, threshold_and_classify
from .smooth import gaussian_filter_3D



def extract_MR_domain_mask_from_segmentation(
        IDs: list[int], 
        ID_type: str,
        seg_path: str, 
        out_path: str) -> str:

    # load segmentation as an nifti image
    seg = nib.load(seg_path)
    
    # make a copy of the image (_data is np array)
    mask_opROI_data = copy.deepcopy(seg.get_fdata())
    
    if ID_type == "including":
        # Find the binary mask for the output ROI, including the in_IDs
        mask_opROI_data = np.isin(mask_opROI_data, IDs).astype(int)
    elif ID_type == "excluding":
        # Find the binary mask for the output ROI, excluding the ex_IDs
        mask_opROI_data = (~np.isin(mask_opROI_data, IDs)).astype(int)
    
    empty_mask = not np.any(mask_opROI_data)
    if empty_mask:
        raise ValueError(f"IDs {IDs} not found in {seg_path}")
    
    # Make the nifti mask image
    mask_opROI = nib.Nifti1Image(mask_opROI_data, seg.affine, seg.header)
    
    # Save the nifti mask image
    nib.save(mask_opROI, out_path)
    
    return mask_opROI



def create_MR_domain_mask(
        IDs: list[int], 
        ID_type: str,
        seg_path: str, 
        opROI_name: str, 
        op_dir: str) -> str:
    """
    Create an MR mask (binary .nii.gz image).

    Parameters
    ----------
    IDs : The integer IDs of ROIs that the mask should include/exclude. 
    ID_type: "including" or "excluding"
    seg_path : file path. The path of the MR segmentation file from which the 
               mask is created. The file should be .nii or .nii.gz.
    opROI_name : The name of the output combined ROI. 
    op_dir : directory path. The path of the output directory where the output mask is stored. 

    Returns
    -------
    opmask_path: file path. The path of the output mask file, ending in .nii.gz.

    """
    
    # load segmentation as an nifti image
    seg = nib.load(seg_path)
    
    # make a copy of the image (_data is np array)
    mask_opROI_data = copy.deepcopy(seg.get_fdata())
    
    if ID_type == "including":
        # Find the binary mask for the output ROI, including the in_IDs
        mask_opROI_data = np.isin(mask_opROI_data, IDs).astype(int)
    elif ID_type == "excluding":
        # Find the binary mask for the output ROI, excluding the ex_IDs
        mask_opROI_data = (~np.isin(mask_opROI_data, IDs)).astype(int)
    
    empty_mask = not np.any(mask_opROI_data)
    if empty_mask:
        raise ValueError(f"The mask for {opROI_name} is empty: not found in {seg_path}")
    
    # Make the nifti mask image
    mask_opROI = nib.Nifti1Image(mask_opROI_data, seg.affine, seg.header)
    
    opmask_fullname = f'mask_mr_{opROI_name}.nii.gz'
    opmask_path = os.path.join(op_dir, opmask_fullname)
    
    # Save the nifti mask image
    nib.save(mask_opROI, opmask_path)
    
    return opmask_path



def transform(
        ipmask_path: str, 
        opmask_path: str, 
        lta_path: str, 
        thr: float, 
        save_bfthr_mask: bool) -> str:
    
    op_dir = str(Path(opmask_path).parent)
    
    opmask_basename, extension = aux.extract_file_name(opmask_path)

    # names and path of the bfthr output mask
    opmask_bfthr_basename = opmask_basename + '_bfthr'
    opmask_bfthr_fullname = opmask_bfthr_basename + extension
    opmask_bfthr_path = os.path.join(op_dir, opmask_bfthr_fullname)
    
    # Check if the lta_path exists
    assert os.path.isfile(lta_path) is True, f"""The lta_path {lta_path} does not exist."""
    
    # map the input mask from inDomain to outDomain
    # mapped mask has decimal values
    os.system(f'$FREESURFER_HOME/bin/mri_convert -at {lta_path} {ipmask_path} {opmask_bfthr_path}')
    
    # load the bfthr output mask
    opmask_bfthr = nib.load(opmask_bfthr_path)
    opmask_bfthr_data = opmask_bfthr.get_fdata()
    
    # thresholding the bfthr output mask (decimal) to make it binary
    opmask_data = (opmask_bfthr_data >= thr).astype(int)
    opmask = nib.Nifti1Image(opmask_data, opmask_bfthr.affine, opmask_bfthr.header)
    
    # name and path of the binary output mask
    opmask_fullname = opmask_basename + extension
    opmask_path = os.path.join(op_dir, opmask_fullname)
    nib.save(opmask, opmask_path)
        
    if save_bfthr_mask == False:
        # delete the intermediate bfthr mask
        os.remove(opmask_bfthr_path)

    return opmask_path



def linear_transform(
        ipmask_path: str, 
        inDomain: str, 
        outDomain: str, 
        lta_path: str, 
        thr: float, 
        save_bfthr_mask: bool, 
        op_dir: str) -> str:
    """
    Performs linear transform of the input mask from inDomain to outDomain. 

    Parameters
    ----------
    ipmask_path : directory path. The path of the input mask .nii.gz file. 
    inDomain : The input domain, 'mr' or 'pet'.
    outDomain : The output domain, 'mr' or 'pet'.
    lta_path : file path. The path of the .reg.lta file, containing information of the linear transform. 
    thr : float in [0, 1]. The threshold for mapping decimal values to binary 
          values for the PET mask transformed from MR domain.
    save_bfthr_mask : True - save the intermediate decimal-valued mask before thresholding; 
                      False - do not save.  
    op_dir : directory path. The path of the output directory where the output mask is stored. 

    Returns
    -------
    opmask_path: file path. The path of the output mask file, ending in .nii.gz

    """
    
    # dashed versions of inDomain and outDomain
    inD = f'_{inDomain}_'
    outD = f'_{outDomain}_'
    
    ipmask_basename, extension = aux.extract_file_name(ipmask_path)

    assert inD in ipmask_basename, f"The mask name {ipmask_basename} should contain {inD}"
    
    # create the output mask'a base name by replacing the first occurence of inD in ipmask_name with outD
    opmask_basename = ipmask_basename.replace(inD, outD, 1)
    
    # names and path of the bfthr output mask
    opmask_bfthr_basename = opmask_basename + '_bfthr'
    opmask_bfthr_fullname = opmask_bfthr_basename + extension
    opmask_bfthr_path = os.path.join(op_dir, opmask_bfthr_fullname)
    
    # Check if the lta_path exists
    assert os.path.isfile(lta_path) is True, f"""The lta_path {lta_path} does not exist."""
    
    # map the input mask from inDomain to outDomain
    # mapped mask has decimal values
    os.system(f'$FREESURFER_HOME/bin/mri_convert -at {lta_path} {ipmask_path} {opmask_bfthr_path}')
    
    # load the bfthr output mask
    opmask_bfthr = nib.load(opmask_bfthr_path)
    opmask_bfthr_data = opmask_bfthr.get_fdata()
    
    # thresholding the bfthr output mask (decimal) to make it binary
    opmask_data = (opmask_bfthr_data >= thr).astype(int)
    opmask = nib.Nifti1Image(opmask_data, opmask_bfthr.affine, opmask_bfthr.header)
    
    # name and path of the binary output mask
    opmask_fullname = opmask_basename + extension
    opmask_path = os.path.join(op_dir, opmask_fullname)
    nib.save(opmask, opmask_path)
        
    if save_bfthr_mask == False:
        # delete the intermediate bfthr mask
        os.remove(opmask_bfthr_path)

    return opmask_path
            



def create_PET_domain_mask(
        IDs: list[int], 
        ID_type: str,
        thr: float, 
        save_PET_bfthr_mask: bool, 
        save_MR_mask: bool, 
        opROI_name: str, 
        seg_path: str,
        mr_masks_dir: str,
        mr2pet_lta_path: str,
        op_dir: str) -> str:
    """
    Create a PET mask (binary .nii.gz image) that includes the given ROIs. 
    
    Parameters
    ----------
    IDs : The integer IDs of ROIs that the mask should include/exclude. 
    ID_type: "including" or "excluding"
    thr : float in [0, 1]. The threshold for mapping decimal values to binary 
          values for the PET mask transformed from MR domain.
    save_PET_bfthr_mask : True - save the intermediate decimal-valued PET mask before thresholding; 
                          False - do not save. 
    save_MR_mask : True - save the intermediate MR mask; 
                   False - do not save.
    opROI_name : The name of the output combined ROI. 
    seg_path: segmentation file path
    mr_masks_dir: directory path of the MR masks folder
    mr2pet_lta_path: file path of the mr2pet lta transformation
    op_dir : directory path. The path of the output directory where the output mask is stored.  

    Returns
    -------
    PETmask_path : file path of the PET mask.
    """

    MRmask_path = create_MR_domain_mask(
        IDs = IDs,
        ID_type = ID_type,
        seg_path = seg_path,
        opROI_name = opROI_name,
        op_dir = mr_masks_dir)

    PETmask_path = linear_transform(
        ipmask_path = MRmask_path,
        inDomain = 'mr',
        outDomain = 'pet',
        lta_path = mr2pet_lta_path,
        thr = thr,
        save_bfthr_mask = save_PET_bfthr_mask,
        op_dir = op_dir)

    if save_MR_mask == False:
        # delete the MR mask
        os.remove(MRmask_path)
        
    return PETmask_path



def transform_MR_segmentation(seg_path: str,
                              seg_transformed_path: str,
                              lta_path: str,
                              middle_dir: str,
                              del_middle_dir: bool) -> None:

    seg = nib.load(seg_path)
    seg_data = seg.get_fdata().astype(int)
    all_IDs = list(set(seg_data.flatten()))
    all_IDs.sort()
    # remove 0: background
    try:
        all_IDs.remove(0)
    except:
        pass
    #print(all_IDs)
    
    
    mask_paths = []
    print("Extracting MR domain mask from segmentation")
    for ID in all_IDs:
        
        print(f'ID: {ID}')
        
        mask_path = os.path.join(middle_dir, f'mask_mr_ID{ID}.nii.gz')
        mask_paths.append(mask_path)
        
        extract_MR_domain_mask_from_segmentation(
                IDs = [ID],
                ID_type = 'including',
                seg_path = seg_path,  
                out_path = mask_path)
    
    
    mask_transformed_paths = []
    print("Transforming masks")
    for (ID, mask_path) in zip(all_IDs, mask_paths):
        
        print(f'ID: {ID}')
        
        mask_transformed_path = os.path.join(middle_dir, f'mask_mr_ID{ID}_tran.nii.gz')
        mask_transformed_paths.append(mask_transformed_path)
        
        transform(ipmask_path = mask_path, 
                  opmask_path = mask_transformed_path, 
                  lta_path = lta_path, 
                  thr = 0.5, 
                  save_bfthr_mask = False)

    mask_tran_sample_path = mask_transformed_paths[0]
    mask_tran_sample = nib.load(mask_tran_sample_path)
    mask_tran_sample_affine = mask_tran_sample.affine
    mask_tran_sample_header = mask_tran_sample.header
        
    merge_masks_to_segmentation(IDs = all_IDs,
                                mask_paths = mask_transformed_paths,
                                seg_out_path = seg_transformed_path,
                                affine = mask_tran_sample_affine,
                                header = mask_tran_sample_header)
    
    if del_middle_dir:
        shutil.rmtree(middle_dir)
        
        
    # # test of the masks overlap
    # all_ones = [1] * len(all_IDs)
    # merge_masks_to_segmentation_overlap_test(IDs = all_ones,
    #                                          mask_paths = mask_transformed_paths,
    #                                          seg_out_path = seg_transformed_path,
    #                                          affine = mask_tran_sample_affine,
    #                                          header = mask_tran_sample_header)    
    
    return None    




def merge_masks_to_segmentation(IDs: list[int],
                                mask_paths: list[str],
                                seg_out_path: str,
                                affine,
                                header) -> None:
    
    
    seg_out_data = None
    
    for (ID, mask_path) in zip(IDs, mask_paths):
        
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata().astype(int)
        
        if seg_out_data is None:
            
            seg_out_data = ID * mask_data
        
        else:
            
            seg_out_data += ID * mask_data
            
    seg_out = nib.Nifti1Image(seg_out_data, affine, header)
    
    nib.save(seg_out, seg_out_path)
            
    return None


def merge_masks_to_segmentation_overlap_test(IDs: list[int],
                                             mask_paths: list[str],
                                             seg_out_path: str,
                                             affine,
                                             header) -> None:
    
    seg_out_data = None
    
    for (ID, mask_path) in zip(IDs, mask_paths):
        
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata().astype(int)
        
        if seg_out_data is None:
            
            seg_out_data = ID * mask_data
        
        else:
            
            seg_out_data += ID * mask_data
            
    uniq = set(seg_out_data.flatten())
    print(f'uniq = {uniq}')
    
    num_twos = np.count_nonzero(seg_out_data == 2)
    print(f'number of 2: {num_twos}')
            
    return None




def generate_masked_img(ipimg_path, mask_path, maskedROI_name, op_dir):
    """
    For a given input image, apply a binary mask and generate a masked image. 

    Parameters
    ----------
    ipimg_path : string, file path
        The file path of the input image, ending in .nii or .nii.gz. 
        The input image can be either 3D (one frame) or 4D (multi-frame).
    mask_path : string, file path
        The file path of the binary mask, ending in .niior .nii.gz.
        The mask should be 3D. 
    maskedROI_name : string
        The name of the masked ROI.
    op_dir : string, directory path
        The path of the output directory where the masked image is stored. 

    Returns
    -------
    opimage_path : string, file path
        The path of the output masked image file, ending in .nii or .nii.gz.
        The output image is of the same dimension as the input image. 
    """
    
    # Load the input image
    ipimg = nib.load(ipimg_path)
    ipimg_data = ipimg.get_fdata()
    
    # Load the mask
    mask = nib.load(mask_path)
    mask_data = mask.get_fdata()
    

    
    if len(ipimg.shape) == 3:
        # one frame (i.e. single 3D image)
        
        assert ipimg.shape == mask.shape, f"""The input image {ipimg_path} (shape = {ipimg_data.shape}) and 
        the mask {mask_path} (shape = mask_data.shape) should have the same dimension."""
    
        # Generate the output nifti image
        opimg_data = ipimg_data * mask_data
        
    elif len(ipimg.shape) == 4:
        # multi-frame 
        
        num_frames = ipimg.shape[-1]
        
        assert ipimg.shape[0:3] == mask.shape, f"""Each frame of the input image {ipimg_path} (shape = {ipimg.shape[0:3]}) and 
        the mask {mask_path} (shape = mask_data.shape) should have the same dimension."""
    
        opimg_data = copy.deepcopy(ipimg_data)
        for i in range(num_frames):
            opimg_data[..., i] = ipimg_data[..., i] * mask_data


    opimg = nib.Nifti1Image(opimg_data, ipimg.affine, ipimg.header)


    ipimg_basename, extension = aux.extract_file_name(ipimg_path)
        
    # Generate the output image's name and path
    # E.g. if input image = frame5.nii.gz and maskedROI_name is "cerebellum"
    # Then output image = frame5_cerebellum.nii.gz
    opimg_basename = ipimg_basename + '_' + maskedROI_name
    opimg_fullname = opimg_basename + extension
    opimg_path = os.path.join(op_dir, opimg_fullname)


    nib.save(opimg, opimg_path)
    
    return opimg_path
            
        
        
def num_voxels(mask_path):
    
    # Load the mask
    mask = nib.load(mask_path)
    mask_data = mask.get_fdata()

    v = int(np.count_nonzero(mask_data))
    
    return v
        


def dilation(
        ball_radius: float,
        ippath: str,
        oppath: str) -> None:
    """
    Dilate the mask by a ball with a given radius.

    """

    # load input image
    ip = nib.load(ippath)
    
    # make a copy of the image 
    op_data = copy.deepcopy(ip.get_fdata())
    
    # dilation by a 
    op_data = binary_dilation(op_data, ball(ball_radius)).astype(op_data.dtype)
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip.affine, ip.header)
            
    # Save the output image
    nib.save(op, oppath)
    
    return None 


def erosion(
        depth: int,
        ippath: str,
        oppath: str) -> None:
    """
    Erode the mask.
    
    depth: number of layers/pixels to erode

    """
    
    # load input image
    ip = nib.load(ippath)
    
    # make a copy of the image 
    op_data = copy.deepcopy(ip.get_fdata())
    
    # erosion 
    for i in range(depth):
        op_data = binary_erosion(op_data).astype(op_data.dtype)
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip.affine, ip.header)
            
    # Save the output image
    nib.save(op, oppath)
    
    return None


def erosion_ODonell2024(
        fwhm: float,
        threshold: float,
        ippath: str,
        oppath: str) -> None:
    """
    Erode the mask by first applying a Gaussian filter, then thresholding. 
    
    ODnell2024 paper: https://jnm.snmjournals.org/content/65/6/956.abstract

    """
    
    # ippath: jack/home/Document/myfile.txt
    # gaussian_oppath: jack/home/Document/gaussiantemp_myfile.txt
    ipp = Path(ippath)
    parent_dir = ipp.parent
    ipname = ipp.name
    gaussian_opp = os.path.join(parent_dir, 'gaussiantemp_' + ipname)
    gaussian_oppath = str(gaussian_opp)
    
    sigma = gaussian_fwhm2sigma(fwhm)
    
    gaussian_filter_3D(sigma = sigma,
                       ippath = ippath,
                       oppath = gaussian_oppath)
    
    threshold_and_classify(threshold = threshold, 
                           up_class = 1, 
                           down_class = 0, 
                           ippath = gaussian_oppath, 
                           oppath = oppath)
    
    # remove the intermediate file
    os.remove(gaussian_oppath)
    return None


def union(
        ippath1: str,
        ippath2: str,
        oppath: str) -> None:
    """
    Find the union of two masks.
    """

    ip1 = nib.load(ippath1)
    ip1_data = copy.deepcopy(ip1.get_fdata()).astype(int)
    
    ip2 = nib.load(ippath2)
    ip2_data = copy.deepcopy(ip2.get_fdata()).astype(int)
    
    op_data = ip1_data | ip2_data
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip1.affine, ip1.header)
            
    # Save the output image
    nib.save(op, oppath)
    
    return None


def intersect(
        ippath1: str,
        ippath2: str,
        oppath: str) -> None:
    """
    Find the intersection of two masks.
    """

    ip1 = nib.load(ippath1)
    ip1_data = copy.deepcopy(ip1.get_fdata()).astype(int)
    
    ip2 = nib.load(ippath2)
    ip2_data = copy.deepcopy(ip2.get_fdata()).astype(int)
    
    op_data = ip1_data & ip2_data
    
    # Make the output image
    op = nib.Nifti1Image(op_data, ip1.affine, ip1.header)
            
    # Save the output image
    nib.save(op, oppath)
    
    return None


def minus(
        ippath1: str,
        ippath2: str,
        oppath: str) -> None:
    """
    Find all regions in image1 that is not in image2. 
    """

    ip1 = nib.load(ippath1)
    ip1_data = copy.deepcopy(ip1.get_fdata()).astype(int)
    
    ip2 = nib.load(ippath2)
    ip2_data = copy.deepcopy(ip2.get_fdata()).astype(int)
    
    intersect = ip1_data & ip2_data 
    
    op_data = np.logical_xor(ip1_data, intersect)

    # Make the output image
    op = nib.Nifti1Image(op_data, ip1.affine, ip1.header)
            
    # Save the output image
    nib.save(op, oppath)
    
    return None


def complement(
        ippath: str,
        oppath: str) -> None:
    """
    Find the complement mask. 
    """

    ip = nib.load(ippath)
    ip_data = copy.deepcopy(ip.get_fdata()).astype(int)
    
    op_data = 1 - ip_data

    # Make the output image
    op = nib.Nifti1Image(op_data, ip.affine, ip.header)
            
    # Save the output image
    nib.save(op, oppath)
    
    return None



def where_in_seg(seg, vlist: list[int]):
    """
    Return the indices of values in vlist in seg.
    
    seg: numpy array
    
    Output: a numpy array of lists/tuples of indices
    """
    
    indices = None
    
    for v in vlist:
        
        if indices is None:
            
            indices = np.argwhere(seg == v)
            
        else:
            
            newindices = np.argwhere(seg == v)
            indices = np.vstack((indices, newindices))
    
    return indices


def relocate_labels_in_seg(segin_path: str,
                           old_tissue_labels: list[int],
                           new_tissue_label: int,
                           new_tissue_mask_path: str,
                           segout_path: str,
                           relabel_exclude: list[int] | None = None) -> None:
    """
    To relocate some tissue labels and relabel them. This is useful when one has 
    a better segmentation of a particular tissue and want to update the original
    segmentation. 
    
    Essense:
        A. The old tissue labels produces an old mask. The new tissue has a new mask. 
        B. These two masks form three regions: (1) intersection; (2) voxels in new but not in old;
            (3) voxels in old but not in new.
        C. For regions (1) & (2), simply label the voxels to new_tissue_label
        D. For region (3), for each voxel, we search for the nearest tissue and give the voxel the label
            of that tissue. 
    
    segin_path: path of the input segmentation
    old_tissue_labels: list of tissue labels to be relocated and relabeled
    new_tissue_label: the new label given to the relocated tissues
    new_tissue_mask_path: path of the new tissue mask
    segout_path: path of the output segmentation
    relabel_exclude: in step D above, exclude these list of tissues in the search
    
    """
    
    
    
    segin_full = nib.load(segin_path)
    segin = segin_full.get_fdata().astype(int)
    
    old_mask = np.isin(segin, old_tissue_labels).astype(int)
    #print('volume of old mask: ', np.count_nonzero(old_mask))
    
    new_mask_full = nib.load(new_tissue_mask_path)
    new_mask = new_mask_full.get_fdata().astype(int)
    #print('volume of new mask: ', np.count_nonzero(new_mask))
    
    
    # set all voxels in new_mask to the new label
    segout_tmp = (1-new_mask) * segin + new_tissue_label * new_mask
    
    # find the old-excluding-new mask 
    intersect_mask = (old_mask & new_mask).astype(int)
    oldExclNew_mask = np.logical_xor(old_mask, intersect_mask).astype(int)
    #print('volume of oldExclNew mask: ', np.count_nonzero(oldExclNew_mask))
    

    # find the surrounding tissues of old-excluding-new region
    oldExclNewDilate_mask = binary_dilation(oldExclNew_mask, ball(1)).astype(oldExclNew_mask.dtype)
    oldExclNew_surround_mask = oldExclNewDilate_mask - oldExclNew_mask
    oldExclNew_surround_indices = np.argwhere(oldExclNew_surround_mask == 1)
    #print('number of oldExclNew-surround indices: ', len(oldExclNew_surround_indices))
    
    oldExclNew_surround_tissues = []
    for ind in oldExclNew_surround_indices:
        oldExclNew_surround_tissues.append(segout_tmp[tuple(ind)])
    oldExclNew_surround_tissues = set(oldExclNew_surround_tissues)
    oldExclNew_surround_tissues.remove(new_tissue_label)
    
    if relabel_exclude is not None:
        for label in relabel_exclude:
            if label in oldExclNew_surround_tissues:
                oldExclNew_surround_tissues.remove(label)
    #print('number of oldExclNew-surround tissues: ', len(oldExclNew_surround_tissues))
    #print('oldExclNew-surround tissues: ', oldExclNew_surround_tissues)
    
    # For each voxel in oldExclNew_mask, we need to assign a new label to it.
    # To find the nearest tissue to the target voxel, we need to find the distance from
    # each surrounding tissue to the target voxel, which equals to the minimum distance
    # from all voxels in that tissue to the target voxel. 
    # This distance can be calculated by considering only the tissue's voxels in
    # oldExclNew_surround_mask, which saves much computation. 
    
    seg_in_oldExclNew_surround = oldExclNew_surround_mask * segout_tmp 
    critical_indices_of_tissue = {}   # int -> list[indices]
    for label in oldExclNew_surround_tissues:
        critical_tissue_mask = np.isin(seg_in_oldExclNew_surround, [label]).astype(int)
        critical_tissue_indices = np.argwhere(critical_tissue_mask == 1)
        critical_indices_of_tissue[label] = critical_tissue_indices
    
    # print('Surrouding tissue - number of critical indices in tissue:')
    # for (tissue, indices) in critical_indices_of_tissue.items():
    #     print(tissue, len(indices))
    
    
    oldExclNew_indices = np.argwhere(oldExclNew_mask == 1)
    for target_ind in oldExclNew_indices:
        nearest_tissue = None
        min_dist_alltissues = np.Inf
        for (tissue, indices) in critical_indices_of_tissue.items():
            
            min_dist_thistissue = np.Inf
            for tissue_ind in indices:
                dist = np.linalg.norm(tissue_ind - target_ind)
                if dist < min_dist_thistissue:
                    min_dist_thistissue = dist
                # print(f'Tissue {tissue}, dist: {dist}, min_dist: {min_dist_thistissue}')
            
            if min_dist_thistissue < min_dist_alltissues:
                min_dist_alltissues = min_dist_thistissue
                nearest_tissue = tissue
            
            # print('###################################')
            # print(f'target_ind: {target_ind}, nearest_tissue: {nearest_tissue}, min_dist: {min_dist_alltissues}')
            # print('###################################')
        
        
        # print('************************************')
        # print(f'target_ind: {target_ind}, nearest_tissue: {nearest_tissue}, min_dist: {min_dist_alltissues}')
        # print('************************************')
        
        # update the label at target_ind
        segout_tmp[tuple(target_ind)] = nearest_tissue
        
    
    segout = segout_tmp
    
    # Make the output image
    segout_full = nib.Nifti1Image(segout, segin_full.affine, segin_full.header)
            
    # Save the output image
    nib.save(segout_full, segout_path)
    
    return None

