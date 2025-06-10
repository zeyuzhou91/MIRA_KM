"""
tac.py 

TAC (Time-Activity Curve) Module

This module provides classes and functions for handling and analyzing TAC data
in PET imaging workflows. Supports extracting TACs from PET images and CSV files,
fitting models to TACs, and visualizing results.

Author: Zeyu Zhou
Date: 2025-05-26
"""

import os
import numpy as np
import nibabel as nib
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from collections.abc import Callable

from .frameschedule import FrameSchedule
from .typing_utils import NumpyRealNumberArray
from .utils import auxiliary as aux 



class TAC_FULL:
    """
    A container for full TAC statistics over multiple frames.

    Attributes:
        avg (np.ndarray): Average signal per frame.
        tot (np.ndarray): Total signal per frame.
        std (np.ndarray): Standard deviation per frame.
        num_voxels (int): Number of voxels used for extraction.
        unit (str): Unit of signal (e.g., kBq/mL).
        histogram (tuple): Histogram of voxel values per frame (optional).
    """
    def __init__(self):
        self.avg = None
        self.tot = None
        self.std = None
        self.num_voxels = None
        self.unit = None
        self.histogram = None



class TAC:
    """
    Represents a Time-Activity Curve (TAC) of a PET region of interest (ROI).

    Attributes:
        t (np.ndarray): Time midpoints for each frame.
        y (np.ndarray): Activity values.
        std (np.ndarray): Standard deviations.
        unit (str): Unit of activity.
        t_unit (str): Unit of time.
        full_info (TAC_FULL): Full statistics from TAC extraction.
        fitted_func (Callable): Fitted function from model fitting.
    """
    
    def __init__(self,
                 t: NumpyRealNumberArray | None = None,
                 y: NumpyRealNumberArray | None = None,
                 std: NumpyRealNumberArray | None = None,
                 num_voxels: int | None = None, 
                 unit: str | None = None,
                 t_unit: str | None = None,
                 full_info: TAC_FULL | None = None):
        
        self.t = t
        self.y = y
        self.std = std
        self.unit = unit
        self.t_unit = t_unit
        
        self.num_voxels = num_voxels
        
        self.num_frames = None
        
        self.full_info = full_info
        
        self.fitted_func = None
        
        if t is not None:
            self.num_frames = len(t)
            
        if t is not None and y is not None:
            if len(t) != len(y):
                raise ValueError("length of t and y do not match")
        
        
    @classmethod
    def from_file(cls,
                  fs: FrameSchedule,
                  tacfile_path: str):
        """
        Load TAC from CSV file.

        Args:
            fs (FrameSchedule): Time frame schedule.
            tacfile_path (str): Path to the CSV file containing TAC data.

        Returns:
            TAC: Instance of TAC with loaded data.
        """
        
        y, std, num_voxels, unit = extract_tac_from_csv(tacfile_path)
    
        return cls(t = fs.mid_points,
                   y = y,
                   std = std,
                   num_voxels = num_voxels, 
                   unit = unit,
                   t_unit = fs.unit,
                   full_info = None)
    
    
    @classmethod
    def from_PETimg(cls,
                    fs: FrameSchedule,
                    PETimg_path: str,
                    mask_path: str,
                    opfile_path: str | None = None, 
                    PETimg_unit: str | None = None):
        """
        Extract TAC from a 3D or 4D PET image using a binary mask.

        Args:
            fs (FrameSchedule): Time frame schedule.
            PETimg_path (str): Path to the PET image.
            mask_path (str): Path to binary mask.
            opfile_path (str, optional): Where to save the extracted TAC.
            PETimg_unit (str, optional): Unit of PET signal.

        Returns:
            TAC: TAC instance with extracted data.
        """        
        
        
        tac_full = extract_tac_from_PETimg(PETimg_path, mask_path, opfile_path, PETimg_unit)
        
        return cls(t = fs.mid_points,
                   y = tac_full.avg,
                   std = tac_full.std,
                   num_voxels = tac_full.num_voxels,
                   unit = tac_full.unit,
                   t_unit = fs.unit,
                   full_info = tac_full)

    
    def cut(self, indices: list[int]):
        """
        Remove selected frame indices from the TAC.

        Args:
            indices (list[int]): List of indices to remove.
        """
        
        self.t = np.delete(self.t, indices)
        self.y = np.delete(self.y, indices)
        self.std = np.delete(self.std, indices)
        
        self.num_frames = len(self.t)
        
        return None


    def scale_y(self, 
                multiply_factor: float,
                new_y_unit: str):
        """
        Scale the y-values of TAC by a factor and update the unit.

        Args:
            multiply_factor (float): Scale factor.
            new_y_unit (str): Updated unit after scaling.
        """
        
        self.y *= multiply_factor
        self.unit = new_y_unit
        if self.std is not None:
            self.std *= multiply_factor
            
        return None
    
    
    def fit(self,
            model: Callable[..., float],
            p0: tuple | None = None,
            bounds: tuple | None = None, 
            ) -> None:
        """
        Fit the TAC using a given model function.

        Args:
            model (Callable): Model function (time, *params).
            p0 (tuple, optional): Initial parameter guess.
            bounds (tuple, optional): Bounds for parameters.

        Returns:
            np.ndarray: Optimized parameter values.
        """
        
        if bounds is None:
            popt, _ = curve_fit(f = model, 
                                xdata = self.t,
                                ydata = self.y,
                                p0 = p0)
        else:
            popt, _ = curve_fit(f = model, 
                                xdata = self.t,
                                ydata = self.y,
                                p0 = p0,
                                bounds = bounds)
        
        def f(t):
            return model(t, *popt)
        
        self.fitted_func = f
        
        return popt
    
    
    def plot_with_fitting(self,
                          xlim: list[float] | tuple[float] | None = None,
                          ylim: list[float] | tuple[float] | None = None,
                          title: str | None = None,
                          op_dir: str | None = None,
                          op_filename: str | None = None):
        
        plt.figure()
        
        plt.scatter(self.t, self.y, c='blue')
        
        if self.fitted_func is not None:
            tmax = np.max(self.t)
            tfit = np.linspace(0, tmax*1.1, 1000)
            yfit = self.fitted_func(tfit)
            plt.plot(tfit, yfit, c='red')
            
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
        
        if title is not None:
            plt.title(title)
        
        plt.grid()
        if op_dir is not None:
            if op_filename is None:
                opfile_path = os.path.join(op_dir, 'tac_with_fitting.png')
            else:
                opfile_path = os.path.join(op_dir, op_filename)
            plt.savefig(opfile_path, bbox_inches="tight", dpi=300)
        
        plt.show()
        plt.close()
        
        return None        
        
    
    
    
    
    

def extract_tac_from_PETimg(
        PETimg_path: str, 
        mask_path: str,
        opfile_path: str | None = None, 
        PETimg_unit: str | None = None):

    """
    Extracts TAC data from a PET image and binary mask.

    Args:
        PETimg_path (str): Path to PET image (3D or 4D).
        mask_path (str): Path to binary mask.
        opfile_path (str, optional): Where to write results.
        PETimg_unit (str, optional): Unit of PET signal.

    Returns:
        TAC_FULL: Object containing extracted TAC statistics.
    """

    
    # For a given PET image, apply a binary mask and generate the ROI information. 
    
    # Load the input PET image
    
    PETimg = nib.load(PETimg_path)
    PETimg_data = PETimg.get_fdata()
    
    # Load the mask
    mask = nib.load(mask_path)
    mask_data = mask.get_fdata()
    
    result = TAC_FULL()
    result.num_voxels = int(np.count_nonzero(mask_data))
    
    if len(PETimg.shape) == 3:
        # one frame (i.e. single 3D image)
        
        num_frames = 1
        
        assert PETimg.shape == mask.shape, f"""The input image {PETimg_path} (shape = {PETimg_data.shape}) and 
        the mask {mask_path} (shape = mask_data.shape) should have the same dimension."""
        
        # outputs an array containing only the masked voxel values in PETimg
        ROI_reduced = np.extract(mask_data, PETimg_data)
        
        if PETimg_unit == 'Bq/mL':
            ROI_reduced /= 1000.0 # from Bq/mL to kBq/mL
                
        result.tot = np.array([np.sum(ROI_reduced)])
        result.avg = np.array([np.mean(ROI_reduced)])
        result.std = np.array([np.std(ROI_reduced)])
        
        
        
    elif len(PETimg.shape) == 4:
        # multi-frame 
        
        num_frames = PETimg.shape[-1]
        
        assert PETimg.shape[0:3] == mask.shape, f"""Each frame of the input image {PETimg_path} (shape = {PETimg.shape[0:3]}) and 
        the mask {mask_path} (shape = mask_data.shape) should have the same dimension."""
    
        result.tot = np.zeros(num_frames)
        result.avg = np.zeros(num_frames)
        result.std = np.zeros(num_frames)     
        result.histogram = ()
        for i in range(num_frames):
            
            # the ith frame
            frame = PETimg_data[..., i]
            
            # outputs an array containing only the masked voxel values in this frame
            frame_ROI_reduced = np.extract(mask_data, frame)
            
            if PETimg_unit == 'Bq/mL':
                frame_ROI_reduced /= 1000.0 # from Bq/mL to kBq/mL
            
            result.tot[i] = np.sum(frame_ROI_reduced)
            result.avg[i] = np.mean(frame_ROI_reduced)
            result.std[i] = np.std(frame_ROI_reduced)
            result.histogram += (np.histogram(frame_ROI_reduced, bins=range(26)),)
           
    
    if PETimg_unit == 'kBq/mL':
        #aux.write_to_csv_onecol(avg, 'average', 'kBq/mL', opfile_path)
        if opfile_path is not None:
            aux.write_to_csv_threecols(result.avg, 'average', 'kBq/mL', 
                                       result.std, 'std', 'kBq/mL', 
                                       result.num_voxels * np.ones(num_frames, dtype=int), 'num_voxels', '#',
                                       opfile_path)
        result.unit = PETimg_unit
        
    elif PETimg_unit == 'Bq/mL':
        if opfile_path is not None:
            aux.write_to_csv_threecols(result.avg, 'average', 'kBq/mL', 
                                     result.std, 'std', 'kBq/mL', 
                                     result.num_voxels * np.ones(num_frames, dtype=int), 'num_voxels', '#',
                                     opfile_path)
        result.unit = 'kBq/mL'
        
    elif PETimg_unit == 'unitless':
        if opfile_path is not None:
            aux.write_to_csv_threecols(result.avg, 'average', 'unitless', 
                                     result.std, 'std', 'unitless', 
                                     result.num_voxels * np.ones(num_frames, dtype=int), 'num_voxels', '#',
                                     opfile_path)
        result.unit = 'unitless'

    elif PETimg_unit == 'uncertain':
        if opfile_path is not None:
            aux.write_to_csv_threecols(result.avg, 'average', 'uncertain', 
                                     result.std, 'std', 'uncertain', 
                                     result.num_voxels * np.ones(num_frames, dtype=int), 'num_voxels', '#',
                                     opfile_path)
        result.unit = 'uncertain'

    return result





def extract_tac_from_csv(filepath: str):

    """
    Loads TAC data from a CSV file with average, std, and voxel count.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        tuple: (avg, std, num_voxels, unit)
    """
        
    #ys, header, unit = aux.read_from_csv_onecol(filepath)
        
    avg, header1, unit1, std, header2, unit2, num_voxels_list, header3, unit3 =  aux.read_from_csv_threecols(filepath)
        
    avg = np.array(avg)
    std = np.array(std)
        
    if unit1 == 'kBq/mL':
        pass
    
    elif unit1 == 'Bq/mL':
        avg = avg / 1000.0
        std = std / 1000.0
        unit1 = 'kBq/mL'
    
    elif unit1 == 'unitless':
        pass
    
    elif unit1 == 'uncertain':
        pass
        
    num_voxels = int(num_voxels_list[0])
    
    return avg, std, num_voxels, unit1

    


    

        
            
# if __name__ == "__main__":
        
    
#     array = [1, 1, 1, 2, 2, 3, 3, 4, 6, 10]
#     myfs = FrameSchedule.from_durations(array)
#     myfs.unit = 'min'
    
#     mytac = TAC(myfs)
    
#     print(mytac.frameschedule.durations)

    
#     # myf = lambda t: t**2
    
#     # ctac = ContinuousTAC(frameschedule = myfs, 
#     #                      f = myf, 
#     #                      unit = 'Bq/mL', 
#     #                      name = 'ctac')
    
#     # ctac.plot(trange = [0, 2])
    
    
    
#     # ys = [2, 3, 5, 6, 8, 9, 10, 13, 15, 19]
#     # roi = ROI(name = 'brain', IDs = [1, 2], ID_type = "including")
    
#     # dtac = DiscreteTAC(frameschedule = myfs,
#     #                    ys = ys, 
#     #                    unit = 'Bq/mL', 
#     #                    roi = roi)
    
#     # dtac.plot()
    
    
    
    
    
    