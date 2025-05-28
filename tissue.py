"""
tissue.py

This module defines classes and utilities for managing brain tissues in PET/MR analysis.
It includes tools for creating and transforming masks, extracting time-activity curves (TACs),
performing morphological operations, and managing collections of tissue data.

Classes:
- Tissue: Represents a tissue ROI and its associated TAC and mask.
- Ref_Tissue: Subclass of Tissue for reference regions.
- Tissue_Collections: Manages multiple Tissue instances as a collection.
- Input_Collections: A container for multiple input sources (BloodInput, Tissue, etc.)

Author: Zeyu Zhou
Date: 2025-05-26
"""

import os
import copy
import shutil
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import copy

from .environment import Environment
from .frameschedule import FrameSchedule
from .tac import TAC
from .arterial import BloodInput
from .utils import filesystem_utils as fu
from .utils import mask


class Tissue:
    """
    Represents a tissue region of interest (ROI) defined by a mask and associated with a TAC.

    Attributes:
    - name (str): Tissue name.
    - IDs (list[int]): Region labels from segmentation maps.
    - ID_type (str): 'including' or 'excluding'.
    - MR_domain_mask_path (str): Path to the MR-space binary mask.
    - PET_domain_mask_path (str): Path to the PET-space binary mask.
    - tac (TAC): Time-activity curve for the region.
    - num_voxels (int): Number of voxels in the mask.
    - fitted_tac: (optional) Fitted version of the TAC.
    """
    
    def __init__(self, 
                 name, 
                 IDs: list[int] | None = None, 
                 ID_type: str | None = None, 
                 MR_domain_mask_path: str | None = None, 
                 PET_domain_mask_path: str | None = None, 
                 tac: TAC | None = None):
        
        self.name = name     # str
        self.IDs = IDs 
        self.ID_type = ID_type  # "including" or "excluding"
        
        self.MR_domain_mask_path = MR_domain_mask_path
        self.PET_domain_mask_path = PET_domain_mask_path

        self.tac = tac
        
        self.num_voxels = calculate_num_voxels_in_mask(MR_domain_mask_path)
        
        self.fitted_tac = None
         
    
    def copy(self):
        return copy.deepcopy(self)
    
    
    @classmethod
    def by_name_and_ID(cls,
                name: str,
                IDs: list[int],
                ID_type: str,
                fs: FrameSchedule,
                env: Environment):
        """
        Initialize a Tissue from segmentation IDs and ID type.
    
        Automatically creates a mask in MR domain and extracts TAC.
    
        Parameters:
        - name (str): ROI name.
        - IDs (list[int]): Region IDs to include or exclude.
        - ID_type (str): 'including' or 'excluding'.
        - fs (FrameSchedule): Frame schedule.
        - env (Environment): Environment with paths and config.
        """
        
        # generate mask
        MR_domain_mask_path = \
            generate_Tissue_MR_domain_mask(name = name, 
                                           IDs = IDs, 
                                           ID_type = ID_type, 
                                           env = env)
                
        # generate tac
        tac = extract_Tissue_tac(name = name, 
                                 fs = fs, 
                                 mask_path = MR_domain_mask_path,
                                 env = env)
        
        return cls(name = name,
                   IDs = IDs,
                   ID_type = ID_type,
                   MR_domain_mask_path = MR_domain_mask_path,
                   PET_domain_mask_path = None,
                   tac = tac)
        

    @classmethod
    def by_MR_mask(cls,
                   name: str,
                   MR_domain_mask_path: str,
                   fs: FrameSchedule,
                   env: Environment):
        """
        Create a Tissue from an existing MR-domain mask file.
        """
        
        # generate tac
        tac = extract_Tissue_tac(name = name, 
                                 fs = fs, 
                                 mask_path = MR_domain_mask_path,
                                 env = env)
        
        return cls(name = name,
                   IDs = None,
                   ID_type = None,
                   MR_domain_mask_path = MR_domain_mask_path,
                   PET_domain_mask_path = None,
                   tac = tac)


    @classmethod
    def by_erosion(cls, 
                   tissue_to_erode, 
                   new_name: str,
                   depth: int,
                   fs: FrameSchedule,
                   env: Environment):
    
        """
        Create a new Tissue by applying binary erosion on an existing Tissue mask.
        """
        
        new_MR_domain_mask_path = tissue_to_erode.MR_domain_mask_path.replace(tissue_to_erode.name, new_name)
            
        mask.erosion(depth = depth, 
                     ippath = tissue_to_erode.MR_domain_mask_path, 
                     oppath = new_MR_domain_mask_path)         
        
        return cls.by_MR_mask(name = new_name, 
                              MR_domain_mask_path = new_MR_domain_mask_path, 
                              fs = fs, 
                              env = env)
    
    
    @classmethod
    def by_erosion_ODonell2024(cls, 
                               tissue_to_erode, 
                               new_name: str,
                               fwhm: float,
                               threshold: float,
                               fs: FrameSchedule,
                               env: Environment):
    
        """
        Create a new Tissue using a smoothing-based erosion approach inspired by O'Donell et al. (2024).
        """
        
        new_MR_domain_mask_path = tissue_to_erode.MR_domain_mask_path.replace(tissue_to_erode.name, new_name)
            
        mask.erosion_ODonell2024(fwhm = fwhm, 
                                 threshold = threshold, 
                                 ippath = tissue_to_erode.MR_domain_mask_path, 
                                 oppath = new_MR_domain_mask_path)         
        
        return cls.by_MR_mask(name = new_name, 
                              MR_domain_mask_path = new_MR_domain_mask_path, 
                              fs = fs, 
                              env = env)


    @classmethod
    def by_dilation(cls, 
                    tissue_to_dilate, 
                    new_name: str,
                    radius: int,
                    fs: FrameSchedule,
                    env: Environment):
    
        """
        Create a new Tissue by applying binary dilation on an existing Tissue mask.
        """
        
        new_MR_domain_mask_path = tissue_to_dilate.MR_domain_mask_path.replace(tissue_to_dilate.name, new_name)
            
        mask.dilation(ball_radius = radius, 
                      ippath = tissue_to_dilate.MR_domain_mask_path, 
                      oppath = new_MR_domain_mask_path)         
        
        return cls.by_MR_mask(name = new_name, 
                              MR_domain_mask_path = new_MR_domain_mask_path, 
                              fs = fs, 
                              env = env)


    def generate_mask(self, 
                      delete_existing_mask_first: bool,
                      env: Environment):
        """
        Generate MR or PET domain mask for the tissue based on environment setting.
        """
        
        if env.variables['mask_domain'] == 'MR':
            self.generate_MR_domain_mask(env = env)
            self.num_voxels = calculate_num_voxels_in_mask(self.MR_domain_mask_path)    
        
        elif env.variables['mask_domain'] == 'PET':
            self.generate_PET_domain_mask(env = env)
            self.num_voxels = calculate_num_voxels_in_mask(self.PET_domain_mask_path)    
        
        return None

        

    def generate_MR_domain_mask(self, env: Environment):
        
        """Generate or regenerate the MR-domain mask."""
                
        self.MR_domain_mask_path = \
            generate_Tissue_MR_domain_mask(name = self.name,
                                           IDs = self.IDs,
                                           ID_type = self.ID_type,
                                           env = env)
        
        #print(f'{self.name} MR mask generated.')
                
        return None
   
 
    def generate_PET_domain_mask(self, 
                                 env: Environment,
                                 THR: float | None = None):
        
        """Generate or regenerate the PET-domain mask using LTA transformation."""
        
        self.PET_domain_mask_path = \
            generate_Tissue_PET_domain_mask(name = self.name,
                                            IDs = self.IDs,
                                            ID_type = self.ID_type,
                                            MR_domain_mask_path = self.MR_domain_mask_path,
                                            env = env,
                                            THR = THR)
        
        return None
            

    def extract_tac(self, 
                    fs: FrameSchedule,
                    env: Environment):
        
        """Extract the tissue TAC, using existing CSV if available, or from PET image."""
        
        
        if env.variables['mask_domain'] == 'MR':
            mask_path = self.MR_domain_mask_path
        elif env.variables['mask_domain'] == 'PET':
            mask_path = self.PET_domain_mask_path
        
        
        self.tac = extract_Tissue_tac(name = self.name,
                                      fs = fs,
                                      mask_path = mask_path,
                                      env = env)
            
        return None


    def extract_tac_full(self, 
                         fs: FrameSchedule,
                         env: Environment):
        
        """Extract the full TAC information including statistics across voxels."""

        
        if env.variables['mask_domain'] == 'MR':
            mask_path = self.MR_domain_mask_path
        elif env.variables['mask_domain'] == 'PET':
            mask_path = self.PET_domain_mask_path
        
        self.tac = TAC.from_PETimg(fs = fs, 
                                   PETimg_path = env.variables['PETimg_path'], 
                                   mask_path = mask_path,
                                   opfile_path = None,
                                   PETimg_unit = env.variables['PETimg_unit'])
            
        print(f'{self.name} tac full extracted.')
        
        return None

    
    def erode_mask(self,
                    env: Environment,
                    depth: int | None = None):
        
        """Apply in-place binary erosion to the tissue's MR-domain mask."""
        
        if depth is None:
            depth = 1
        
        mask.erosion(depth = depth, 
                      ippath = self.MR_domain_mask_path, 
                      oppath = self.MR_domain_mask_path) 
            
        return None
    
        

    def dilate_mask(self,
                    env: Environment,
                    radius: int | None = None):
        
        """Apply in-place binary dilation to the tissue's MR-domain mask."""
        
        if radius is None:
            radius = 1
        
        mask.dilation(ball_radius = radius, 
                      ippath = self.MR_domain_mask_path, 
                      oppath = self.MR_domain_mask_path)        
            
        return None
    




class Ref_Tissue(Tissue):
    
    """
    Specialized subclass of Tissue representing a reference region in PET quantification.
    """    
    
    def __init__(self,
                 name, 
                 IDs: list[int] | None = None, 
                 ID_type: str | None = None, 
                 MR_domain_mask_path: str | None = None, 
                 PET_domain_mask_path: str | None = None, 
                 tac: TAC | None = None,
                 num_voxels = None):
    
        super().__init__(name = name, 
                         IDs = IDs, 
                         ID_type = ID_type, 
                         MR_domain_mask_path = MR_domain_mask_path, 
                         PET_domain_mask_path = PET_domain_mask_path, 
                         tac = tac,
                         num_voxels = num_voxels)
        
    


class Tissue_Collections:
    
    """
    Container for multiple Tissue objects. Provides batch operations and plotting utilities.
    """
    
    def __init__(self, 
                 tissues: list[Tissue]):
        
        self.tissues = tissues
        
        self.num_tissues = len(tissues)
        self.names = self.tissue_names()
        self.tac_y_unit, self.tac_t_unit = self.read_tac_units()
        
        
    def copy(self):
        return copy.deepcopy(self)    
        
        
    @classmethod
    def from_txt(cls, 
                 file_path: str, 
                 fs: FrameSchedule,
                 env: Environment):
        """
        Load a set of tissues from a text file containing name-ID pairs.
        """

        tissues = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():
                    parts = line.strip().split(' ')
                    name = parts[0]
                    ID_type = parts[1]
                    IDs_str = parts[2:]
                    IDs = [int(x) for x in IDs_str]
                    
                    tissue = Tissue.by_name_and_ID(name = name, 
                                                   IDs = IDs, 
                                                   ID_type = ID_type, 
                                                   fs = fs, 
                                                   env = env)

                    tissues.append(tissue)
                    
        return cls(tissues)
        
    
    def tissue_names(self):
        
        """Return a list of all tissue names."""
        
        names = [tissue.name for tissue in self.tissues]
        
        return names


    def tissue_by_name(self, name: str) -> Tissue:
        
        """Retrieve a Tissue object by name."""
        
        if name not in self.names:
            raise ValueError(f'{name} not found')
        else:
            ind = self.names.index(name)
            tissue = self.tissues[ind]
            
        return tissue
    
    
    def index_by_name(self, name: str) -> int:
        
        """Return index of a Tissue by its name."""
        
        if name not in self.names:
            raise ValueError(f'{name} not found')
        else:
            ind = self.names.index(name)
            
        return ind
    

    def remove_tissue(self, name: str):
        
        """Remove a Tissue by its name."""

        ind = self.index_by_name(name)
        self.tissues.pop(ind)
        self.num_tissues = len(self.tissues)
        self.names = self.tissue_names()
            
        return None
    
    
    def add_tissue(self, tissue: Tissue):
        
        """Add a Tissue object to the collection."""
        
        self.tissues.append(tissue)
        self.num_tissues = len(self.tissues)
        self.names = self.tissue_names()
        self.tac_y_unit, self.tac_t_unit = self.read_tac_units()
        
        return None
    
       
    def add_tissue_by_name_and_ID(self, 
                                  name: str,
                                  IDs: list[str],
                                  ID_type: str,
                                  fs: FrameSchedule,
                                  env: Environment):
        
        """Create and add a Tissue from ID specification."""
        
        tissue = Tissue.by_name_and_ID(name = name, 
                                       IDs = IDs, 
                                       ID_type = ID_type, 
                                       fs = fs, 
                                       env = env)
        
        self.add_tissue(tissue)
        
        return None

    
    def add_tissue_by_MR_mask(self,
                              name: str,
                              MR_domain_mask_path: str,
                              fs: FrameSchedule,
                              env: Environment):
        
        """Create and add a Tissue using an MR mask path."""
        
        tissue = Tissue.by_MR_mask(name = name, 
                                   MR_domain_mask_path = MR_domain_mask_path, 
                                   fs = fs, 
                                   env = env)
        
        self.add_tissue(tissue)
        
        return None

    
    def read_tac_units(self):
        
        """Infer and validate consistency of TAC units across all tissues."""
        
        if self.tissues is None:
            y_unit, t_unit = None, None
        else:
            all_y_units = np.array([tissue.tac.unit for tissue in self.tissues])
            if (all_y_units == all_y_units[0]).all():
                y_unit = all_y_units[0]
            else:
                raise ValueError('units of tissue tacs do not match')

            all_t_units = np.array([tissue.tac.t_unit for tissue in self.tissues])
            if (all_t_units == all_t_units[0]).all():
                t_unit = all_t_units[0]
            else:
                raise ValueError('t_units of tissue tacs do not match')
            
        return y_unit, t_unit


    def scale_tacs(self, 
                   multiply_factor: float,
                   new_tac_unit: str):
        
        """Scale all tissue TAC values by a specified factor and update units."""
        
        for tissue in self.tissues:
            tissue.tac.scale_y(multiply_factor = multiply_factor,
                               new_y_unit = new_tac_unit)
        
        self.tac_y_unit, self.tac_t_unit = self.read_tac_units()
            
        return None



    def plot_tacs(self, 
                  tissue_names: list[str] | None = None,
                  op_dir: str | None = None,
                  op_filename: str | None = None,
                  title: str | None = None,
                  xlim: list[float] | tuple[float] | None = None,
                  ylim: list[float] | tuple[float] | None = None) -> None:
        
        """Plot TACs for selected tissues with options for saving figures."""
        
        if tissue_names is None or (len(tissue_names) == 1 and tissue_names[0] == 'all'):
            tissues_to_plot = self.tissues
        else:
            tissues_to_plot = []
            for name in tissue_names:
                tissues_to_plot.append(self.tissue_by_name(name))            
            
        colors = cm.rainbow(np.linspace(0, 1, len(tissues_to_plot)))
        
        
        plt.figure()
        for (k, tissue) in enumerate(tissues_to_plot):
            color = colors[k].reshape(1,-1)
            
            ts = tissue.tac.t
            ys = tissue.tac.y

            #plt.scatter(ts, ys, c=color, label= f'{tissue.name} ({tissue.num_voxels})')        
            plt.plot(ts, ys, '-o', c=color, label= f'{tissue.name} ({tissue.num_voxels})')
            
        plt.xlabel(f't ({self.tac_t_unit})')
        plt.ylabel(f'{self.tac_y_unit}')
        
        if xlim is not None:
            if xlim[0] is not None:
                plt.xlim(xmin = xlim[0])
            if xlim[1] is not None:
                plt.xlim(xmax = xlim[1])
        if ylim is not None:
            if ylim[0] is not None:
                plt.ylim(ymin = ylim[0])
            if ylim[1] is not None:
                plt.ylim(ymax = ylim[1])
        
        if title is None:
            plt.title('TACs')
        else:
            plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        if op_dir is not None:
            if op_filename is None:
                opfile_path = os.path.join(op_dir, 'tac.png')
            else:
                opfile_path = os.path.join(op_dir, op_filename)
            plt.savefig(opfile_path, bbox_inches="tight", dpi=300)
        #plt.show()
        plt.close()
        
        return None


    def plot_tacs_std(self, 
              tissue_names: list[str] | None = None,
              op_dir: str | None = None,
              op_filename: str | None = None,
              title: str | None = None,
              xlim: list[float] | tuple[float] | None = None,
              ylim: list[float] | tuple[float] | None = None) -> None:
        
        """Plot standard deviation of TACs across tissues."""
        
        # TO DO
        # similar to plot_tacs
        return None 


    def return_tissue_copy(self, name):
        """Return a deep copy of a tissue identified by name."""
        
        tissue = self.tissue_by_name(name)
        tissue_copy = tissue.copy()
        
        return tissue_copy

    
    def tissue_tac(self, name):
        """Return the TAC object for a specific tissue."""
        
        tissue = self.tissue_by_name(name)
        
        return tissue.tac



class Input_Collections:
    """
    Simple container for a mixed list of BloodInput, Tissue, or Ref_Tissue instances.
    """
    
    def __init__(self, 
                 inputs: list[BloodInput, Tissue, Ref_Tissue]):
        
        self.inputs = inputs





def generate_Tissue_MR_domain_mask(name: str,
                                   IDs: list[int],
                                   ID_type: str,
                                   env: Environment):

    """
    Generate the MR-domain binary mask for a tissue using segmentation labels.

    This function constructs a binary mask in MR space by including or excluding
    specified label IDs from the segmentation image. If a mask with the same name
    already exists and `delete_existing_mask_first` is set in `env`, the old mask is deleted.

    Parameters:
    ----------
    name : str
        Name of the tissue or region of interest (used for file naming).
    IDs : list[int]
        List of label IDs to include or exclude from the segmentation image.
    ID_type : str
        Mode of operation: either "including" (keep listed IDs) or "excluding" (remove listed IDs).
    env : Environment
        Environment object containing paths and processing settings (e.g., segmentation path, output dir).

    Returns:
    -------
    MR_domain_mask_path : str
        File path to the generated binary mask in MR space.
    """
    
    # MR mask
    MR_domain_mask_path = os.path.join(env.variables['MR_masks_dir'], f'mask_mr_{name}.nii.gz')
    delete_existing_mask_first = env.variables['delete_existing_mask_first']
    
    if os.path.exists(MR_domain_mask_path) and delete_existing_mask_first:
        fu.delete_file(MR_domain_mask_path)
    
    if os.path.exists(MR_domain_mask_path):
        pass
    
    else:
        MR_domain_mask_path = mask.create_MR_domain_mask(
                                IDs = IDs, 
                                ID_type = ID_type,
                                seg_path = env.variables['seg_path'], 
                                opROI_name = name, 
                                op_dir = env.variables['MR_masks_dir'])
    
    #print(f'{self.name} MR mask generated.')
            
    return MR_domain_mask_path



def generate_Tissue_PET_domain_mask(name: str, 
                                    IDs: list[int],
                                    ID_type: str,
                                    MR_domain_mask_path: str,
                                    env: Environment,
                                    THR: float | None = None):

    """
    Generate a PET-domain binary mask for a tissue by transforming its MR-domain mask.
    
    If the MR-domain mask does not exist, it will be generated using segmentation labels.
    The transformation uses a linear transform `.lta` file and thresholding.
    
    Parameters:
    ----------
    name : str
        Name of the tissue (used for naming the output file).
    IDs : list[int]
        List of segmentation label IDs (used only if MR mask needs to be generated).
    ID_type : str
        Either "including" or "excluding", to control mask generation logic.
    MR_domain_mask_path : str
        Path to the MR-space mask to be transformed.
    env : Environment
        Environment object containing paths and configuration variables.
    THR : float, optional
        Threshold to apply on the resampled MR-domain mask after transformation to PET space. 
        Default is 0.8.
    
    Returns:
    -------
    PET_domain_mask_path : str
        Path to the transformed binary mask in PET space.
    """

    
    if THR is None:
        THR = 0.8
    
    # PET mask
    PET_domain_mask_path = os.path.join(env.variables['PET_masks_dir'], f'mask_pet_{name}.nii.gz')
    delete_existing_mask_first = env.variables['delete_existing_mask_first']
    
    if os.path.exists(PET_domain_mask_path) and delete_existing_mask_first:
        fu.delete_file(PET_domain_mask_path)
    
    if os.path.exists(PET_domain_mask_path):
        pass
    
    else:
        if not os.path.exists(MR_domain_mask_path):
            
            MR_domain_mask_path = generate_Tissue_MR_domain_mask(name = name,
                                                          IDs = IDs,
                                                          ID_type = ID_type,
                                                          env = env)
                    
        PET_domain_mask_path = mask.linear_transform(
                                ipmask_path = MR_domain_mask_path,
                                inDomain = 'mr',
                                outDomain = 'pet',
                                lta_path = env.variables['mr2pet_lta_path'],
                                thr = THR,
                                save_bfthr_mask = False,
                                op_dir = env.variables['PET_masks_dir'])
    
    #print(f'{self.name} PET mask generated.')
            
    return PET_domain_mask_path


def calculate_num_voxels_in_mask(mask_path: str | None):
    """
    Calculate the number of non-zero voxels in a binary mask.

    Parameters:
    ----------
    mask_path : str | None
        Path to the binary mask file (in NIfTI format). If None, returns None.

    Returns:
    -------
    num_voxels : int | None
        Number of non-zero voxels (i.e., included region). Returns None if input is None.
    """
    
    if mask_path is None:
        num_voxels = None
    else:
        num_voxels = mask.num_voxels(mask_path)
    return num_voxels


def extract_Tissue_tac(name: str,
                       fs: FrameSchedule,
                       mask_path: str,  
                       env: Environment):
    
    """
    Extract or compute the time-activity curve (TAC) for a tissue.

    If a CSV file containing the TAC already exists, it is read from disk unless 
    the environment setting `delete_existing_tac_first` is True. If no CSV exists 
    or deletion is requested, the TAC is extracted from the PET image using the 
    binary mask provided.

    Parameters:
    ----------
    name : str
        Name of the tissue (used for naming the TAC CSV file).
    fs : FrameSchedule
        Frame schedule providing the time midpoints for each frame.
    mask_path : str
        Path to the binary mask (either in MR or PET space).
    env : Environment
        Environment object providing file paths, units, and options.

    Returns:
    -------
    tac : TAC
        TAC object representing the mean activity in the tissue across frames.
    """
    
    tacfilename_extension = '_tac.csv'
    tacfile_name = name + tacfilename_extension
    tacfile_path = os.path.join(env.variables['tacs_dir'], tacfile_name)
    
    delete_existing_tac_first = env.variables['delete_existing_tac_first']
    
    if os.path.exists(tacfile_path) and delete_existing_tac_first:
        fu.delete_file(tacfile_path)
        
    if os.path.exists(tacfile_path):
        
        tac = TAC.from_file(fs = fs, 
                            tacfile_path = tacfile_path)

    else:
                
        tac = TAC.from_PETimg(fs = fs, 
                              PETimg_path = env.variables['PETimg_path'], 
                              mask_path = mask_path,
                              opfile_path = tacfile_path,
                              PETimg_unit = env.variables['PETimg_unit'])
        
    print(f'{name} tac extracted.')
    
    return tac

   