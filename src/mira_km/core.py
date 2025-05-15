"""
Core class definitions. 
"""

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from matplotlib.pyplot import cm

from .typing_utils import NumpyRealNumberArray
from collections.abc import Callable

from .tool import aux, mask, tac



class Environment:
    def __init__(self, path_names_file):
        
        self.path_dict = {}
        
        self.read_path_names_from_file(path_names_file)
        
        
    def print_all(self):
        
        for (name, path) in self.path_dict.items():
            print(f'{name} = {path}')
        
        return None
    
    
    def reset_all(self):
        
        for name in self.path_dict.keys():
            self.path_dict[name] = ''
            
        
    def read_path_names_from_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Strip whitespace and add non-empty lines to the array
                    path_name = line.strip()
                    if path_name:
                        self.path_dict[path_name] = ''
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except IOError:
            print(f"Error: Could not read file at {file_path}")
        
        return None



class FrameSchedule:
    def __init__(self, mid_points: NDArray, durations: NDArray, start_points: NDArray, unit: str):
        self.mid_points = mid_points
        self.durations = durations         
        self.start_points = start_points
        self.unit = unit
        self.num_frames = len(mid_points)
        
    @classmethod
    def from_midpoints(cls, 
                       array: list[float] = None, 
                       filepath: str = None):
    
        if array is not None and filepath is not None:
            raise ValueError("Only one of array or filepath should be provided, not both.")
        
        elif array is not None:
            mid_points = array
            unit = ''
            
        elif filepath is not None:
            mid_points, _, unit = aux.read_from_csv_onecol(filepath)
        
        else:
            raise ValueError("Either array or filepath must be provided.")
        
        durations = []
        start_points = []
        
        cur_start = 0.0
        for mid in mid_points:
            
            start_points.append(cur_start)
            
            duration = 2 * (mid - cur_start)
            durations.append(duration)
            
            nxt_start = cur_start + duration
            
            cur_start = nxt_start
        
        mid_points = np.array(mid_points)
        durations = np.array(durations)
        start_points = np.array(start_points)
        
        return cls(mid_points, durations, start_points, unit)

    
    @classmethod
    def from_durations(cls, 
                       array: list[float] = None,
                       filepath: str = None):
        
        if array is not None and filepath is not None:
            raise ValueError("Only one of array or filepath should be provided, not both.")
        
        elif array is not None:
            durations = array
            unit = ''
            
        elif filepath is not None:
            durations, _, unit = aux.read_from_csv_onecol(filepath)
        
        else:
            raise ValueError("Either array or filepath must be provided.")
        
        
        mid_points = []
        start_points = []
        
        cur_start = 0.0
        for duration in durations:
            
            start_points.append(cur_start)
            
            nxt_start = cur_start + duration
            
            mid = (cur_start + nxt_start) / 2.0
            mid_points.append(mid)
            
            cur_start = nxt_start
        
        mid_points = np.array(mid_points)
        durations = np.array(durations)
        start_points = np.array(start_points)
        
        return cls(mid_points, durations, start_points, unit)




class ROI:
    def __init__(self, name, IDs=None, ID_type=None, MR_domain_mask_path=None, PET_domain_mask_path=None, num_voxels=None):
        self.name = name     # str
        self.IDs = IDs if IDs is not None else None     # list[int]
        self.ID_type = ID_type if ID_type is not None else None  # "including" or "excluding"
        
        self.MR_domain_mask_path = MR_domain_mask_path
        self.PET_domain_mask_path = PET_domain_mask_path

        self.num_voxels = num_voxels  # int
        # self.vol_ml = None   # volume in [mL]
                 
    
    @classmethod
    def from_copy(cls, roi0):
        """
        Create an ROI by copying roi0
        """
        
        return cls(name = roi0.name,
                   IDs = roi0.IDs,
                   ID_type = roi0.ID_type,
                   MR_domain_mask_path = roi0.MR_domain_mask_path,
                   PET_domain_mask_path = roi0.PET_domain_mask_path,
                   num_voxels = roi0.num_voxels)
        
    
    def rename(self, new_name: str):
        
        old_name = self.name
        self.name = new_name
        
        if self.MR_domain_mask_path is not None:
            
            old_MR_domain_mask_path = self.MR_domain_mask_path
            new_MR_domain_mask_path = old_MR_domain_mask_path.replace(old_name, new_name)
            
            # copy the local mask path
            shutil.copy2(old_MR_domain_mask_path, new_MR_domain_mask_path)
            
            # update
            self.MR_domain_mask_path = new_MR_domain_mask_path
        
        if self.PET_domain_mask_path is not None:
            
            old_PET_domain_mask_path = self.PET_domain_mask_path
            new_PET_domain_mask_path = old_PET_domain_mask_path.replace(old_name, new_name)
            
            # copy the local mask path
            shutil.copy2(old_PET_domain_mask_path, new_PET_domain_mask_path)
        
            # update
            self.PET_domain_mask_path = new_PET_domain_mask_path
        
        return None


    def generate_mask(self, 
                       delete_existing_mask_first: bool,
                       env: Environment):
        
        if env.path_dict['mask_domain'] == 'MR':
            self.generate_MR_domain_mask(delete_existing_mask_first = delete_existing_mask_first,
                                    env = env)
        
        elif env.path_dict['mask_domain'] == 'PET':
            self.generate_PET_domain_mask(delete_existing_mask_first = delete_existing_mask_first,
                                     env = env)
                
        return None

        

    def generate_MR_domain_mask(self, 
                                delete_existing_mask_first: bool,
                                env: Environment):
        
        # MR mask
        MR_domain_mask_path = os.path.join(env.path_dict['MR_masks_dir'], f'mask_mr_{self.name}.nii.gz')
        
        if os.path.exists(MR_domain_mask_path) and delete_existing_mask_first:
            os.remove(MR_domain_mask_path)
        
        if os.path.exists(MR_domain_mask_path):
            pass
        
        else:
            MR_domain_mask_path = mask.create_MR_domain_mask(
                                    IDs = self.IDs, 
                                    ID_type = self.ID_type,
                                    seg_path = env.path_dict['seg_path'], 
                                    opROI_name = self.name, 
                                    op_dir = env.path_dict['MR_masks_dir'])
        self.MR_domain_mask_path = MR_domain_mask_path
                
        return None

   
 
    def generate_PET_domain_mask(self, 
                                 delete_existing_mask_first: bool,
                                 env: Environment):
        
        THR = 0.8
        
        # PET mask
        PET_domain_mask_path = os.path.join(env.path_dict['PET_masks_dir'], f'mask_pet_{self.name}.nii.gz')
        
        if os.path.exists(PET_domain_mask_path) and delete_existing_mask_first:
            os.remove(PET_domain_mask_path)
        
        if os.path.exists(PET_domain_mask_path):
            pass
        
        else:
            if not os.path.exists(self.MR_domain_mask_path):
                
                self.generate_MR_domain_mask(delete_existing_mask_first = delete_existing_mask_first,
                                             env = env)
                    
            PET_domain_mask_path = mask.linear_transform(
                                    ipmask_path = self.MR_domain_mask_path,
                                    inDomain = 'mr',
                                    outDomain = 'pet',
                                    lta_path = env.path_dict['mr2pet_lta_path'],
                                    thr = THR,
                                    save_bfthr_mask = False,
                                    op_dir = env.path_dict['PET_masks_dir'])

        self.PET_domain_mask_path = PET_domain_mask_path
                
        return None


    # def generate_masks(self, 
    #                    delete_existing_masks_first: bool,
    #                    env: Environment):
        
    #     # MR mask
    #     mr_mask_path = os.path.join(env.path_dict['mr_masks_dir'], f'mask_mr_{self.name}.nii.gz')
        
    #     if os.path.exists(mr_mask_path) and delete_existing_masks_first:
    #         os.remove(mr_mask_path)
        
    #     if not os.path.exists(mr_mask_path):
    #         mr_mask_path = mask.create_MR_mask(
    #                                 IDs = self.IDs, 
    #                                 ID_type = self.ID_type,
    #                                 seg_path = env.path_dict['seg_path'], 
    #                                 opROI_name = self.name, 
    #                                 op_dir = env.path_dict['mr_masks_dir'])
    #     self.mr_mask_path = mr_mask_path
        
    #     THR = 0.8
    #     # PET mask
    #     pet_mask_path = os.path.join(env.path_dict['pet_masks_dir'], f'mask_pet_{self.name}.nii.gz')
        
    #     if os.path.exists(pet_mask_path) and delete_existing_masks_first:
    #         os.remove(pet_mask_path)
        
    #     if not os.path.exists(pet_mask_path):
    #         pet_mask_path = mask.linear_transform(
    #                                 ipmask_path = self.mr_mask_path,
    #                                 inDomain = 'mr',
    #                                 outDomain = 'pet',
    #                                 lta_path = env.path_dict['mr2pet_lta_path'],
    #                                 thr = THR,
    #                                 save_bfthr_mask = False,
    #                                 op_dir = env.path_dict['pet_masks_dir'])
    #     self.pet_mask_path = pet_mask_path
        
    #     return None
    
    
    def extract_tac(self, 
                    delete_existing_tac_first: bool, 
                    env: Environment | None = None,
                    tacfilename_extension: str | None = None) -> (NumpyRealNumberArray, str):
        
        """
        Extract tac from the tac file (csv) if it exists; otherwise, extract it 
        from the PET image. 
        
        PETimg_path: dynamic (4D) image. 
        """
        
        if tacfilename_extension is None:
            tacfilename_extension = '_tac.csv'
        
        tacfile_name = self.name + tacfilename_extension
        tacfile_path = os.path.join(env.path_dict['tacs_dir'], tacfile_name)
        
        if os.path.exists(tacfile_path) and delete_existing_tac_first:
            os.remove(tacfile_path)
            
        if os.path.exists(tacfile_path):
                
            mytac, std, num_voxels, unit = tac.extract_tac_from_csv(tacfile_path)
                
        else:
            
            if env.path_dict['mask_domain'] == 'MR':
                ROImask_path = self.MR_domain_mask_path
            elif env.path_dict['mask_domain'] == 'PET':
                ROImask_path = self.PET_domain_mask_path
            
            mytac_data = tac.extract_tac_from_PETimg(
                            PETimg_path = env.path_dict['PETimg_path'],
                            ROImask_path = ROImask_path,
                            opfile_path = tacfile_path, 
                            PETimg_unit = env.path_dict['PETimg_unit'])
            
            mytac = mytac_data.avg
            std = mytac_data.std
            unit = mytac_data.unit
            num_voxels = mytac_data.num_voxels
            
        self.num_voxels = num_voxels
        
        print(f'{self.name} OK')
        
        return mytac, std, unit   
    
    

    def return_tac_full_results(self, 
                                env: Environment | None = None):
        
        """
        Return full tac results. 
        """

        if env.path_dict['mask_domain'] == 'MR':
            ROImask_path = self.MR_domain_mask_path
        elif env.path_dict['mask_domain'] == 'PET':
            ROImask_path = self.PET_domain_mask_path
            
        mytac_data = tac.extract_tac_from_PETimg(
            PETimg_path = env.path_dict['PETimg_path'],
            ROImask_path = ROImask_path,
            opfile_path = None, 
            PETimg_unit = env.path_dict['PETimg_unit'])
                    
        return mytac_data



    def erode_mask_ODonell2024(self,
                               env: Environment,
                               fwhm: float,
                               threshold: float):
        
        if env.path_dict['mask_domain'] == 'MR':
            mask_path = self.MR_domain_mask_path
            
        elif env.path_dict['mask_domain'] == 'PET':
            mask_path = self.PET_domain_mask_path
        
        mask.erosion_ODonell2024(fwhm = fwhm, 
                                 threshold = threshold, 
                                 ippath = mask_path, 
                                 oppath = mask_path)    
        return None


    
    def erode_mask(self,
                   env: Environment,
                   depth: int | None = None):
        
        if env.path_dict['mask_domain'] == 'MR':
            self.erode_MR_domain_mask(depth)
            
        elif env.path_dict['mask_domain'] == 'PET':
            self.erode_PET_domain_mask(depth)
            
        return None
    
    
    
    def erode_MR_domain_mask(self, depth: int | None = None):
        """
        Depth: number of pixels to erode
        """
        
        if depth is None:
            depth = 1
        
        mask.erosion(depth = depth, 
                     ippath = self.MR_domain_mask_path, 
                     oppath = self.MR_domain_mask_path)        
        return None


    
    def erode_PET_domain_mask(self, depth: int | None = None):
        """
        Depth: number of pixels to erode
        """
        
        if depth is None:
            depth = 1
        
        mask.erosion(depth = depth, 
                     ippath = self.PET_domain_mask_path, 
                     oppath = self.PET_domain_mask_path)
        
        return None
    

    def dilate_mask(self,
                   env: Environment,
                   radius: int | None = None):
        
        if env.path_dict['mask_domain'] == 'MR':
            self.dilate_MR_domain_mask(radius)
            
        elif env.path_dict['mask_domain'] == 'PET':
            self.dilate_PET_domain_mask(radius)
            
        return None
    
    
    
    def dilate_MR_domain_mask(self, radius: int | None = None):
        """
        radius: number of pixels to dilate
        """
        
        if radius is None:
            radius = 1
        
        mask.dilation(ball_radius = radius, 
                      ippath = self.MR_domain_mask_path, 
                      oppath = self.MR_domain_mask_path)        
        return None

    
    def dilate_PET_domain_mask(self, radius: int | None = None):
        """
        radius: number of pixels to dilate
        """
        
        if radius is None:
            radius = 1
        
        mask.dilation(ball_radius = radius, 
                      ippath = self.PET_domain_mask_path, 
                      oppath = self.PET_domain_mask_path)
        
        return None


    def complement_mask(self, env: Environment):
        
        if env.path_dict['mask_domain'] == 'MR':
            mask.complement(ippath = self.MR_domain_mask_path, 
                            oppath = self.MR_domain_mask_path)
            
        elif env.path_dict['mask_domain'] == 'PET':
            mask.complement(ippath = self.PET_domain_mask_path, 
                            oppath = self.PET_domain_mask_path)
            
        return None
    
                      
    

class TAC:
    def __init__(self, 
                 is_ref: bool | None = None,
                 t: NDArray | None = None, 
                 data: NDArray | None = None,
                 std: NDArray | None = None,
                 rois: list[ROI] | None = None,
                 unit: str | None = None,
                 t_unit: str | None = None):

        self.is_ref = is_ref        
        self.t = t
        self.data = data
        self.std = std
        self.rois = rois
        self.unit = unit
        self.t_unit = t_unit
        
        self.num_elements = None
        self.num_frames = None
        
        if t is not None:
            self.num_frames = len(t)        

        if data is not None:
            self.num_elements = data.shape[0]
            self.num_frames = data.shape[1]
                
        if rois is not None:
            self.num_elements = len(rois)
        else:
            self.num_elements = 0
            
        if data is not None and t is not None:
            if data.shape[1] != len(t):
                raise ValueError("data.shape[1] must be the same as len(t)")
        
        if data is not None and rois is not None:
            if data.shape[0] != len(rois):
                raise ValueError("data.shape[0] must equal to len(rois)")
            
    
    @classmethod
    def from_copy(cls, tac0):
        """
        Create a TAC by copying tac0
        """
        
        return cls(is_ref = tac0.is_ref,
                   t = tac0.t,
                   data = tac0.data,
                   std = tac0.std,
                   rois = tac0.rois,
                   unit = tac0.unit,
                   t_unit = tac0.t_unit)
    
    
    def construct_from_rois_file(self,
                                 file_path: str,
                                 delete_existing_mask_first: bool,
                                 delete_existing_tac_first: bool,
                                 env: Environment):
        
        self.read_rois_from_file(file_path = file_path)
        self.generate_roi_masks(delete_existing_mask_first = delete_existing_mask_first,
                                env = env)
        self.extract_tac(delete_existing_tac_first = delete_existing_tac_first,
                         env = env)
        return None
        
    
    
    def read_rois_and_masks_from_file(self, 
                                      file_path: str, 
                                      delete_existing_mask_first: bool,
                                      env: Environment):
        self.rois = []
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.strip():
                        parts = line.strip().split(' ')
                        name = parts[0]
                        ID_type = parts[1]
                        IDs_str = parts[2:]
                        IDs = [int(x) for x in IDs_str]
                        roi = ROI(name = name,
                                  IDs = IDs,
                                  ID_type = ID_type)
                        roi.generate_mask(delete_existing_mask_first = delete_existing_mask_first,
                                          env = env)
                        self.rois.append(roi)
                        
            self.num_elements = len(self.rois)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except IOError:
            print(f"Error: Could not read file at {file_path}")
            
        return None
    
    
    def read_rois_from_file(self, file_path: str):
        self.rois = []
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.strip():
                        parts = line.strip().split(' ')
                        name = parts[0]
                        ID_type = parts[1]
                        IDs_str = parts[2:]
                        IDs = [int(x) for x in IDs_str]
                        roi = ROI(name = name,
                                  IDs = IDs,
                                  ID_type = ID_type)
                        self.rois.append(roi)
                        
            self.num_elements = len(self.rois)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except IOError:
            print(f"Error: Could not read file at {file_path}")
            
        return None


    def generate_roi_masks(self, 
                           delete_existing_mask_first: bool,
                           env: Environment):
        
        for roi in self.rois:
            roi.generate_mask(delete_existing_mask_first = delete_existing_mask_first,
                              env = env)
        
        return None


    def extract_tac(self, 
                    delete_existing_tac_first: bool, 
                    env: Environment,
                    tacfilename_extension: str | None = None) -> None:
        
        """
        Extract tac from the tac file (csv) if it exists; otherwise, extract it 
        from the PET image. 
        """
        
        if self.rois is None:
            raise ValueError("self.rois not defined")
        
        self.data = np.zeros((self.num_elements, self.num_frames))
        self.std = np.zeros((self.num_elements, self.num_frames))
        
        
        for (i, roi) in enumerate(self.rois):
            
            self.data[i,:], self.std[i,:],  self.unit = roi.extract_tac(delete_existing_tac_first = delete_existing_tac_first,
                                                                        env = env,
                                                                        tacfilename_extension = tacfilename_extension)
            
        if self.data.shape[1] != self.num_frames:
            
            raise ValueError("self.data.shape[1] must be the same as self.num_frames")
        
        return None


    def remove_ROI_with_tac(self, name: str):
        """
        Remove an ROI with its tac by the name. 
        """
        
        i0 = None
        
        for i in range(self.num_elements):
            
            if self.rois[i].name == name:
                i0 = i
                break
            
        if i0 is not None:  # name is found
            
            # update rois
            del self.rois[i0] 
        
            # update data
            # 0 here means the row axis (1 is the col axis)
            self.data = np.delete(self.data, i0, 0)
            self.std = np.delete(self.std, i0, 0)
            
            # update num_elements
            self.num_elements -= 1
        
        else:
            raise ValueError(f"{name} not found in self.rois")
            
        return None
        
    
    def add_ROI_with_tac(self, 
                         new_roi: ROI, 
                         delete_existing_tac_first: bool, 
                         env: Environment,
                         tacfilename_extension: str | None = None):
        """
        Add a new ROI and add its tac.
        """
        
        # update rois
        if self.rois is None:
            self.rois = [new_roi]
        else:
            self.rois.append(new_roi)
        
        newtac, newstd, unit = new_roi.extract_tac(delete_existing_tac_first = delete_existing_tac_first,
                                                   env = env,
                                                   tacfilename_extension = tacfilename_extension)
        
        # update data
        # reshape newtac so that it has ndim=2
        if self.data is None:
            self.data = newtac.reshape(1, -1)
        else:
            self.data = np.append(self.data, newtac.reshape(1, -1), axis=0)
        if self.std is None:
            self.std = newstd.reshape(1, -1)
        else:
            self.std = np.append(self.std, newstd.reshape(1, -1), axis=0)
            
            
        if self.unit is None:
            self.unit = unit
        
        # update num_elements
        self.num_elements += 1
        
        if self.num_elements != self.data.shape[0] or self.num_elements != len(self.rois):
            raise ValueError("num_elements does not match self.data.shape[0] or len(self.rois)")
        
        return None
    
    
    def plot(self, 
             tissues: list[str] | None = None,
             op_dir: str | None = None,
             op_filename: str | None = None,
             title: str | None = None,
             xlim: list[float] | tuple[float] | None = None,
             ylim: list[float] | tuple[float] | None = None) -> None:
        
        tissues_all = [roi.name for roi in self.rois]
        
        if tissues is None or (len(tissues) == 1 and tissues[0] == 'all'):
            ind_to_plot = list(range(self.num_elements))
        else:
            
            ind_to_plot = []
            for tissue in tissues:
                ind_to_plot += [tissues_all.index(tissue)]
        
        colors = cm.rainbow(np.linspace(0, 1, len(ind_to_plot)))
        
        ts = self.t
        
        plt.figure()
        for (k, ind) in enumerate(ind_to_plot):
            color = colors[k].reshape(1,-1)
            
            ys = self.data[ind,:]
        
            if self.rois is not None:
                ROIname = self.rois[ind].name
                ROI_num_voxels = self.rois[ind].num_voxels
            else:
                ROIname = None
        
            #plt.scatter(ts, ys, c=color, label=ROIname)
            if ROI_num_voxels is None:
                plt.plot(ts, ys, '-o', c=color, label= f'{ROIname}')
            else:
                plt.plot(ts, ys, '-o', c=color, label= f'{ROIname} ({ROI_num_voxels})')
        
         
        plt.xlabel(f't ({self.t_unit})')
        plt.ylabel(f'{self.unit}')
        
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
    

    def plot_std(self, 
             tissues: list[str] | None = None,
             op_dir: str | None = None,
             op_filename: str | None = None,
             title: str | None = None,
             xlim: list[float] | tuple[float] | None = None,
             ylim: list[float] | tuple[float] | None = None) -> None:
        
        tissues_all = [roi.name for roi in self.rois]
        
        if tissues is None or (len(tissues) == 1 and tissues[0] == 'all'):
            ind_to_plot = list(range(self.num_elements))
        else:
            
            ind_to_plot = []
            for tissue in tissues:
                ind_to_plot += [tissues_all.index(tissue)]
        
        colors = cm.rainbow(np.linspace(0, 1, len(ind_to_plot)))
        
        ts = self.t
        
        plt.figure()
        for (k, ind) in enumerate(ind_to_plot):
            color = colors[k].reshape(1,-1)
            
            ys = self.std[ind,:]
        
            if self.rois is not None:
                ROIname = self.rois[ind].name
            else:
                ROIname = None
        
            plt.scatter(ts, ys, c=color, label=ROIname)
        
         
        plt.xlabel(f't ({self.t_unit})')
        plt.ylabel(f'{self.unit}')
        
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        
        if title is None:
            plt.title('TAC Standard Deviation')
        else:
            plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        if op_dir is not None:
            if op_filename is None:
                opfile_path = os.path.join(op_dir, 'tac_std.png')
            else:
                opfile_path = os.path.join(op_dir, op_filename)
            plt.savefig(opfile_path, bbox_inches="tight", dpi=300)
        #plt.show()
        plt.close()
        
        return None 


    def roi_by_name(self, name: str) -> ROI:
        
        tissues_all = self.roi_names()
        
        if name not in tissues_all:
            raise ValueError(f'{name} not found')
        
        else:
            
            ind = tissues_all.index(name)
            
            roi = self.rois[ind]
            
        return roi



    def roi_and_index_by_name(self, name: str) -> (ROI, int):
        
        tissues_all = self.roi_names()
        
        if name not in tissues_all:
            raise ValueError(f'{name} not found')
        
        else:
            
            ind = tissues_all.index(name)
            
            roi = self.rois[ind]
            
        return roi, ind


    def return_roi_copy(self, roi_name):
        """
        Return an ROI copy by the given name.
        """
        
        roi0 = self.roi_by_name(roi_name) 
        
        return ROI.from_copy(roi0)


    def tac_of_roi(self, roi_name):
        
        roi, ind = self.roi_and_index_by_name(roi_name)
        
        roi_tac = self.data[ind, :]
        
        return roi_tac

    
    def roi_names(self):
        
        names = [roi.name for roi in self.rois]
        return names
    
    
    
    
    
    # def plot(self) -> None:

    #     # check if self.frameschedule and self.ys exist before plotting

    #     ts = self.frameschedule.mid_points
    #     ys = self.ys
        
    #     plt.figure()
    #     plt.scatter(ts, ys)
    #     plt.xlabel(f't ({self.frameschedule.unit})')
    #     plt.ylabel(f'{self.unit}')
    #     if self.ROI is not None:
    #         plt.title(f'{self.ROI.name}')
    #     plt.show()

    #     return None
    
    





    


    

        
            
if __name__ == "__main__":
        
    
    array = [1, 1, 1, 2, 2, 3, 3, 4, 6, 10]
    myfs = FrameSchedule.from_durations(array)
    myfs.unit = 'min'
    
    mytac = TAC(myfs)
    
    print(mytac.frameschedule.durations)

    
    # myf = lambda t: t**2
    
    # ctac = ContinuousTAC(frameschedule = myfs, 
    #                      f = myf, 
    #                      unit = 'Bq/mL', 
    #                      name = 'ctac')
    
    # ctac.plot(trange = [0, 2])
    
    
    
    # ys = [2, 3, 5, 6, 8, 9, 10, 13, 15, 19]
    # roi = ROI(name = 'brain', IDs = [1, 2], ID_type = "including")
    
    # dtac = DiscreteTAC(frameschedule = myfs,
    #                    ys = ys, 
    #                    unit = 'Bq/mL', 
    #                    roi = roi)
    
    # dtac.plot()
    
    
    
    
    
    