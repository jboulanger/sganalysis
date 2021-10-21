"""Analysis of stress granule images

conda create -n sganalysis
conda activate sganalysis
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
conda install tifffile scikit-image pandas
python -m pip install edt
python -m pip install cellpose

"""
import argparse
import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
from pathlib import Path
from cellpose import models
from skimage.filters import difference_of_gaussians
from skimage.measure import label, regionprops
import edt
import math
from scipy.stats import pearsonr, spearmanr 


class SGA:
    """Stress granule analysis
    Manipulate a list of files and process one file from it
    """
    def __init__(self, filename, use_gpu=False):
        """Create a SGA instance setting the filename"""
        self.filename = filename
        self.filelist = pd.read_csv(filename)
        self.root = Path(os.path.dirname(filename))        
        self.cell_channels = [0,2]
        self.nuclei_channels = [0,0]
        self.granule_channel = 3
        self.other_channel = 2
        self.use_gpu = use_gpu

    def get_image(self, index):
        fn =  self.root / self.filelist['Filename'][index]
        print(fn) 
        return tifffile.imread(fn)
    
    def get_condition(self,index):
        return self.filelist['Condition'][index]

    def segment_cells(self, img):
        model = models.Cellpose(gpu=self.use_gpu, model_type='cyto2')
        mask, flows, styles, diams = model.eval(img, diameter=150, flow_threshold=None, channels=self.cell_channels)
        return mask.astype(np.uint)

    def segment_nuclei(self, img):
        model = models.Cellpose(gpu=self.use_gpu, model_type='nuclei')
        mask, flows, styles, diams = model.eval(img, diameter=80, flow_threshold=None, channels=self.nuclei_channels)
        return mask.astype(np.uint)

    def segment_granules(self, img):
        flt = difference_of_gaussians(img[self.granule_channel,:,:], 1, 4)
        t = flt.mean() + 3*flt.std()
        return label(flt > t).astype(np.uint)
    
    def get_distance_to_nuclei(self, cells, nuclei, id):
        mask = (cells == id) * (nuclei == 0)
        return edt.edt(mask)   

    def measure_objects_in_cell(self, roi, img, cells, nuclei, granules):        
        id = roi.label        
        # measure the distance of granules in this cell
        d = self.get_distance_to_nuclei(cells,nuclei,id)
        if np.isinf(d).any():
            return None
        granules_in_cell = granules * (cells==id) * (nuclei==0)
        distance = regionprops(granules_in_cell, d)
        # intensity in the granules in all channels
        regions = regionprops(granules_in_cell, np.moveaxis(img,0,2))
        # remove very small regions                
        keep = [k for k,x in enumerate(regions) if x.perimeter_crofton > 1 and x.area > 1 and x.major_axis_length > 1]
        regions = [regions[k] for k in keep]
        distance = [distance[k] for k in keep]
        # mask of the cytosol without nuclei and granules
        cytosol_only = (cells==id) * (1- (granules > 0)) * (1- (nuclei > 0))
        # intensity in granule channel in cytosol
        granule_cytosol = np.sum(img[self.granule_channel,:,:] *  cytosol_only) / np.sum(cytosol_only)
        # intensity in other channel in cytosol
        other_cytosol = np.sum(img[self.other_channel,:,:] *  cytosol_only) / np.sum(cytosol_only)
        D1 = {
            "Cell ID": id,
            "ID": [int(x.label) for x in regions],
            
            "Mean Intensity in Granule [Granule]": [x.mean_intensity[self.granule_channel] for x in regions],   
            "Total Intensity in Granule [Granule]" : [x.area * x.mean_intensity[self.granule_channel] for x in regions],
            "Mean Intensity in Cytosol [Granule]" : granule_cytosol,
            "Ratio of Mean Intensity:Cytosol [Granule]":  [x.mean_intensity[self.granule_channel]/granule_cytosol for x in regions], 

            "Mean Intensity in Granule [Other]":  [x.mean_intensity[self.other_channel] for x in regions],
            "Total Intensity in Granule [Other]" : [x.area * x.mean_intensity[self.other_channel] for x in regions],
            "Mean Intensity in Cytosol [Other]":  other_cytosol,
            "Ratio of Mean Intensity:Cytosol [Other]":  [x.mean_intensity[self.other_channel]/other_cytosol for x in regions], 
            
            "Area": [x.area for x in regions],
            "Perimeter" : [x.perimeter for x in regions],
            "Distance to Nuclei": [x.mean_intensity for x in distance],
            "Circularity": [4.0*math.pi*x.area/ x.perimeter_crofton**2 for x in regions],
            "Aspect Ratio" : [x.minor_axis_length / x.major_axis_length for x in regions],
            "Solidity" : [x.solidity for x in regions],
            "Roundness" : [4.0*x.area /(math.pi * x.major_axis_length**2) for x in regions],
        }

        # extract intensity in grnaule and other in cell cytosol
        x1 = img[self.granule_channel,:,:]
        x2 = img[self.other_channel,:,:]
        idx = (cells==id) * (nuclei==0)
        x1 = x1[idx]
        x2 = x2[idx]
        D2 = {            
            "Cell Area" : roi.area,
            "Number of Granules in Cell" : len(regions),
            "Colocalization Spearman Granule:Other" : spearmanr(x1,x2)[0],
            "Colocalization Pearson Granule:Other" : pearsonr(x1,x2)[0],
            "Mean Intensity in Cytosol [Granule]" : granule_cytosol,
            "Mean Intensity in Cytosol [Other]":  other_cytosol,
            "Total Intensity in Cytosol [Granule]" : granule_cytosol * roi.area,
            "Total Intensity in Cytosol [Other]":  other_cytosol * roi.area,
            "Cell ID": id
        }

        return pd.DataFrame.from_dict(D1), pd.DataFrame(D2, index=[roi.label]) 

    def measure(self, img, cells, nuclei, granules):
        rois = regionprops(cells, np.moveaxis(img,0,2))
        rois = [x for x in rois if x.area > 20]       
        obj_df = []
        roi_df = []
        for roi in rois:
            o, r = self.measure_objects_in_cell(roi, img, cells, nuclei, granules)
            if o is not None:
                obj_df.append(o)
                roi_df.append(r)
        return pd.concat(obj_df), pd.concat(roi_df)
        
    def process(self, index):
        img = self.get_image(index)
        cells = self.segment_cells(img)
        print(cells.dtype)
        nuclei = self.segment_nuclei(img)
        granules = self.segment_granules(img)
        O,R = self.measure(img,cells,nuclei,granules)
        for df in [O,R]:
            df['Condition'] = self.get_condition(index)
            df['Filename'] = self.filelist['Filename'][index]
            df['File ID'] = index

        return O,R

def main():
    parser = argparse.ArgumentParser(description='Stress granules analysis')
    parser.add_argument('--file-list',help='filelist')
    parser.add_argument('--index',help='file index',type=int)
    parser.add_argument('--output-by-granules',help='filename of the output table by granule')
    parser.add_argument('--output-by-cells',help='filename of the output table by cell')

    args = parser.parse_args()
    print(f'file list {args.file_list}')
    print(f'index {args.index}')
    sga = SGA(args.file_list)
    granules, cells = sga.process(args.index)
    
    if args.output_by_granules is not None:
        print(f'Saving results by granules to file {args.output_by_granules}')
        granules.to_csv(args.output_by_granules)

    if args.output_by_cells is not None:
        print(f'Saving results by cells to file {args.output_by_cells}')
        cells.to_csv(args.output_by_cells)
        
    
if __name__ == "__main__":
    main()

