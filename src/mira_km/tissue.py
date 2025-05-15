"""
Tissue
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
from .tool import file_handling as fh
from .tool import mask


class Tissue:
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
        Assume MR domain path for now. 
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
        Return a new Tissue by erosion of mask from Tissue T0. Assume mask in 
        MR domain.
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
        Erosion by the ODonell2024 method. Assume mask in MR domain.
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
        Return a new Tissue by dilation of mask from Tissue T0. Assume mask in 
        MR domain.
        """
        
        new_MR_domain_mask_path = tissue_to_dilate.MR_domain_mask_path.replace(tissue_to_dilate.name, new_name)
            
        mask.dilation(ball_radius = radius, 
                      ippath = tissue_to_dilate.MR_domain_mask_path, 
                      oppath = new_MR_domain_mask_path)         
        
        return cls.by_MR_mask(name = new_name, 
                              MR_domain_mask_path = new_MR_domain_mask_path, 
                              fs = fs, 
                              env = env)


        
    # def rename(self, new_name: str):
        
    #     old_name = self.name
    #     self.name = new_name
        
    #     if self.MR_domain_mask_path is not None:
            
    #         old_MR_domain_mask_path = self.MR_domain_mask_path
    #         new_MR_domain_mask_path = old_MR_domain_mask_path.replace(old_name, new_name)
            
    #         # copy the local mask path
    #         fh.copy_file(old_MR_domain_mask_path, new_MR_domain_mask_path)
            
    #         # update
    #         self.MR_domain_mask_path = new_MR_domain_mask_path
        
    #     if self.PET_domain_mask_path is not None:
            
    #         old_PET_domain_mask_path = self.PET_domain_mask_path
    #         new_PET_domain_mask_path = old_PET_domain_mask_path.replace(old_name, new_name)
            
    #         # copy the local mask path
    #         fh.copy_file(old_PET_domain_mask_path, new_PET_domain_mask_path)
        
    #         # update
    #         self.PET_domain_mask_path = new_PET_domain_mask_path
        
    #     return None


    def generate_mask(self, 
                      delete_existing_mask_first: bool,
                      env: Environment):
        
        if env.variables['mask_domain'] == 'MR':
            self.generate_MR_domain_mask(env = env)
            self.num_voxels = calculate_num_voxels_in_mask(self.MR_domain_mask_path)    
        
        elif env.variables['mask_domain'] == 'PET':
            self.generate_PET_domain_mask(env = env)
            self.num_voxels = calculate_num_voxels_in_mask(self.PET_domain_mask_path)    
        
        return None

        

    def generate_MR_domain_mask(self, env: Environment):
        
                
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
        
        """
        Extract tac from the tac file (csv) if it exists; otherwise, extract it 
        from the PET image. 
        """
        
        
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
        
        """
        Extract tac full information.
        """

        
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
        """
        Assume MR domain mask for now.

        """
        
        if depth is None:
            depth = 1
        
        mask.erosion(depth = depth, 
                      ippath = self.MR_domain_mask_path, 
                      oppath = self.MR_domain_mask_path) 
            
        return None
    
        

    def dilate_mask(self,
                    env: Environment,
                    radius: int | None = None):
        
        if radius is None:
            radius = 1
        
        mask.dilation(ball_radius = radius, 
                      ippath = self.MR_domain_mask_path, 
                      oppath = self.MR_domain_mask_path)        
            
        return None
    
    


    # def complement_mask(self, env: Environment):
        
    #     if env.variables['mask_domain'] == 'MR':
    #         mask.complement(ippath = self.MR_domain_mask_path, 
    #                         oppath = self.MR_domain_mask_path)
            
    #     elif env.variables['mask_domain'] == 'PET':
    #         mask.complement(ippath = self.PET_domain_mask_path, 
    #                         oppath = self.PET_domain_mask_path)
            
    #     return None




class Ref_Tissue(Tissue):
    
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
        
        names = [tissue.name for tissue in self.tissues]
        
        return names


    def tissue_by_name(self, name: str) -> Tissue:
        
        if name not in self.names:
            raise ValueError(f'{name} not found')
        else:
            ind = self.names.index(name)
            tissue = self.tissues[ind]
            
        return tissue
    
    
    def index_by_name(self, name: str) -> int:
        
        if name not in self.names:
            raise ValueError(f'{name} not found')
        else:
            ind = self.names.index(name)
            
        return ind
    

    def remove_tissue(self, name: str):

        ind = self.index_by_name(name)
        self.tissues.pop(ind)
        self.num_tissues = len(self.tissues)
        self.names = self.tissue_names()
            
        return None
    
    
    def add_tissue(self, tissue: Tissue):
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
        
        tissue = Tissue.by_MR_mask(name = name, 
                                   MR_domain_mask_path = MR_domain_mask_path, 
                                   fs = fs, 
                                   env = env)
        
        self.add_tissue(tissue)
        
        return None

    
    def read_tac_units(self):
        
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
        
        # TO DO
        # similar to plot_tacs
        return None 


    def return_tissue_copy(self, name):
        """
        Return a tissue copy by the given name.
        """
        
        tissue = self.tissue_by_name(name)
        tissue_copy = tissue.copy()
        
        return tissue_copy

    
    def tissue_tac(self, name):
        
        tissue = self.tissue_by_name(name)
        
        return tissue.tac



class Input_Collections:
    
    def __init__(self, 
                 inputs: list[BloodInput, Tissue, Ref_Tissue]):
        
        self.inputs = inputs





def generate_Tissue_MR_domain_mask(name: str,
                                   IDs: list[int],
                                   ID_type: str,
                                   env: Environment):
    
    # MR mask
    MR_domain_mask_path = os.path.join(env.variables['MR_masks_dir'], f'mask_mr_{name}.nii.gz')
    delete_existing_mask_first = env.variables['delete_existing_mask_first']
    
    if os.path.exists(MR_domain_mask_path) and delete_existing_mask_first:
        fh.delete_file(MR_domain_mask_path)
    
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
    
    if THR is None:
        THR = 0.8
    
    # PET mask
    PET_domain_mask_path = os.path.join(env.variables['PET_masks_dir'], f'mask_pet_{name}.nii.gz')
    delete_existing_mask_first = env.variables['delete_existing_mask_first']
    
    if os.path.exists(PET_domain_mask_path) and delete_existing_mask_first:
        fh.delete_file(PET_domain_mask_path)
    
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
    Number of voxels in the mask. 
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
    Extract tac from the tac file (csv) if it exists; otherwise, extract it 
    from the PET image. 
    """
    
    tacfilename_extension = '_tac.csv'
    tacfile_name = name + tacfilename_extension
    tacfile_path = os.path.join(env.variables['tacs_dir'], tacfile_name)
    
    delete_existing_tac_first = env.variables['delete_existing_tac_first']
    
    if os.path.exists(tacfile_path) and delete_existing_tac_first:
        fh.delete_file(tacfile_path)
        
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

   