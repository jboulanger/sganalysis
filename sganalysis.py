"""Analysis of stress granule images

conda create -n sganalysis
conda activate sganalysis
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install tifffile scikit-image pandas
conda install -c conda-forge jupyterlab
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
from skimage.measure import label, regionprops, find_contours
import edt
import math
from scipy.stats import pearsonr, spearmanr
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('AGG')

def manders_coefficients(im1,im2,display=False):
    ''' Compute Manders overlap coefficients'''
    intersect = np.logical_and(im1,im2)
    p12 = np.sum(intersect>0)
    p1 = np.sum(im1>0)
    p2 = np.sum(im2>0)
    return ( p12 / p1, p12 / p2)

class SGA:
    """Stress granule analysis
    Manipulate a list of files and process one file from it
    """
    def __init__(self, filename, data_path, use_gpu=False):
        """Create a SGA instance setting the filename"""
        self.filename = filename
        self.filelist = pd.read_csv(filename)
        self.root = data_path
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

    def segment_other(self, img):
        flt = difference_of_gaussians(img[self.other_channel,:,:], 1, 4)
        t = flt.mean() + 3*flt.std()
        return label(flt > t).astype(np.uint)

    def spatial_spread(self, mask, intensity):
        """Spread as the trace of the moment matrix"""
        x,y = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
        w = mask * intensity
        sw = np.sum(w)
        if sw < 1e-9:
            return 0.0
        sx = np.sum(w * x) / sw
        sy = np.sum(w * y) / sw
        sxx = np.sum(w * np.square(x-sx)) / sw
        syy = np.sum(w * np.square(y-sy)) / sw
        #sxy = np.sum(w * (x-sx) * (y-sy)) / sw
        return np.sqrt(sxx+syy)

    def measure_objects_in_cell(self, roi, img, cells, nuclei, granules, other):
        id = roi.label

        # define the 3 regions
        mask_not_nuclei = (nuclei==0)
        mask_cell       = (cells==id) * mask_not_nuclei
        mask_particle   = mask_cell  * mask_not_nuclei *  granules
        mask_cytosol    = mask_cell  * mask_not_nuclei * (granules==0)

        if np.sum((cells==id) * (nuclei > 0)) < 1.0:
            return None,None

        # compute distance map
        distance_to_nucleus = regionprops(mask_particle, edt.edt( 1 - (nuclei > 0)))
        distance_to_edge = regionprops(mask_particle, edt.edt(cells == id))
        # intensity in the granules in all channels
        regions = regionprops(mask_particle, np.moveaxis(img,0,2))

        # remove very small regions
        keep = [k for k,x in enumerate(regions) if x.perimeter_crofton > 1 and x.area > 1 and x.major_axis_length > 1]
        regions = [regions[k] for k in keep]
        distance_to_nucleus = [distance_to_nucleus[k] for k in keep]
        distance_to_edge = [distance_to_edge[k] for k in keep]

        # mask of the cytosol without nuclei and granules
        area_cell = np.sum(mask_cell)
        area_cytosol = np.sum(mask_cytosol)
        area_particle = np.sum(mask_particle)
        sum_cell = [np.sum(img[k,:,:] * mask_cell)  for k in range(img.shape[0])]
        sum_cytosol = [np.sum(img[k,:,:] * mask_cytosol)  for k in range(img.shape[0])]
        sum_particle = [np.sum(img[k,:,:] * mask_particle) for k in range(img.shape[0])]
        mean_cell = [x / area_cell for x in sum_cell]
        mean_cytosol = [x / area_cytosol for x in sum_cytosol]
        mean_particle = [x / area_particle for x in sum_particle]
        if len(regions) > 0:
            D1 = {
                "Particle ID": [x.label for x in regions],
                "Cell ID": id,
                "Mean intensity of granule": [x.mean_intensity[self.granule_channel] for x in regions],
                "Total intensity of granule" : [x.area * x.mean_intensity[self.granule_channel] for x in regions],
                "Mean Intensity of granule ratio of particle:cytosol": [x.mean_intensity[self.granule_channel]/mean_cytosol[self.granule_channel] for x in regions],
                "Mean Intensity of other":  [x.mean_intensity[self.other_channel] for x in regions],
                "Total Intensity of other" : [x.area * x.mean_intensity[self.other_channel] for x in regions],
                "Mean Intensity of other ratio of particle:cytosol": [x.mean_intensity[self.other_channel]/mean_cytosol[self.other_channel] for x in regions],
                "Area": [x.area for x in regions],
                "Perimeter" : [x.perimeter_crofton for x in regions],
                "Distance to nuclei": [x.mean_intensity for x in distance_to_nucleus],
                "Distance to edge": [x.mean_intensity for x in distance_to_edge],
                "Circularity": [4.0*math.pi*x.area/ x.perimeter_crofton**2 for x in regions],
                "Aspect ratio" : [x.minor_axis_length / x.major_axis_length for x in regions],
                "Solidity" : [x.solidity for x in regions],
                "Roundness" : [4.0*x.area /(math.pi * x.major_axis_length**2) for x in regions],
            }
        else:
            D1 = None

        # extract intensity in granules and other in cell cytosol
        I_granule = img[self.granule_channel,:,:]
        I_other = img[self.other_channel,:,:]
        I_granule = I_granule[mask_cell]
        I_other = I_other[mask_cell]
        m1,m2 = manders_coefficients((granules>0) * mask_cell, (other>0) * mask_cell)
        D2 = {
            "Cell ID": id,
            "Area of whole cell" : roi.area,
            "Area of cell without nuclei": area_cell,
            "Area of all particles" : area_particle,
            "Area of cytosol": area_cytosol,
            "Area ratio particle:cell": np.sum(mask_particle) / np.sum(mask_cell),
            "Area ratio cytosol:cell":  np.sum(mask_cytosol) / np.sum(mask_cell),
            "Average particle area": np.mean(np.array([x.area for x in regions])),
            "Mean intensity in cell of granule channel" : mean_cell[self.granule_channel],
            "Mean intensity in cell of other channel"   : mean_cell[self.other_channel],
            "Total Intensity in cell of granule channel": sum_cell[self.granule_channel],
            "Total Intensity in celll of other channel"  : sum_cell[self.other_channel],
            "Mean intensity in cytosol of granule channel" : mean_cytosol[self.granule_channel],
            "Mean intensity in cytosol of other channel"   : mean_cytosol[self.other_channel],
            "Total Intensity in cytosol of granule channel": sum_cytosol[self.granule_channel],
            "Total Intensity in cytosol of other channel"  : sum_cytosol[self.other_channel],
            "Mean intensity in particle of granule channel" : mean_particle[self.granule_channel],
            "Mean intensity in particle of other channel "  : mean_particle[self.other_channel],
            "Total Intensity in particle of granule channel": sum_particle[self.granule_channel],
            "Total Intensity in particle of other channel"  : sum_particle[self.other_channel],
            "Intensity ratio of other:granule in particle": mean_particle[self.other_channel] / mean_particle[self.granule_channel],
            "Intensity ratio of other:granule in cell" :  mean_cell[self.other_channel] / mean_cell[self.granule_channel],
            "Intensity ratio of other:granule in cytosol" :  mean_cytosol[self.other_channel] / mean_cytosol[self.granule_channel],
            "Mean intensity ratio of other particle:cytosol": mean_particle[self.other_channel] /  mean_cytosol[self.other_channel],
            "Mean intensity ratio of granule particle:cytosol": mean_particle[self.granule_channel] /  mean_cytosol[self.granule_channel],
            "Colocalization spearman granule:other" : spearmanr(I_granule,I_other)[0],
            "Colocalization pearson granule:other"  : pearsonr(I_granule,I_other)[0],
            "Colocalization manders 1": m1,
            "Colocalization manders 2": m2,
            "Number of particles" : len(regions),
            "Spread of particle" : self.spatial_spread(mask_particle, img[self.granule_channel,:,:]),
            "Mean particle area" : np.mean(D1["Area"]) if D1 is not None else 0.,
            "Mean particle perimeter" : np.mean(D1["Perimeter"]) if D1 is not None else 0.,
            "Mean particle distance to nuclei" : np.mean(D1["Distance to nuclei"]) if D1 is not None else 0.,
            "Mean particle distance to edge" : np.mean(D1["Distance to edge"]) if D1 is not None else 0.,
            "Mean particle circularity" : np.mean(D1["Circularity"]) if D1 is not None else 0.,
            "Mean particle aspect ratio" : np.mean(D1["Aspect ratio"]) if D1 is not None else 0.,
            "Mean particle solidity" : np.mean(D1["Solidity"]) if D1 is not None else 0.,
            "Mean particle roundness" : np.mean(D1["Roundness"]) if D1 is not None else 0.
        }
        if D1 is None :
            return None,  pd.DataFrame(D2, index=[roi.label])
        else :
            return pd.DataFrame.from_dict(D1), pd.DataFrame(D2, index=[roi.label])

    def render_image(self,img):
        stack = np.sqrt(img)
        for i in range(stack.shape[0]):
            plane = stack[i,:,:]
            stack[i,:,:] = 255 * (plane - plane.min()) / (plane.max() - plane.min())
        M = np.array([[0,0,1,1],[0,0.7,0,1],[0.7,0,0,1]])
        rgb = np.apply_along_axis(lambda x:np.matmul(M,x),0,stack)
        rgb = (255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())).astype(int)
        return np.moveaxis(rgb,0,2)

    def measure(self, img, cells, nuclei, granules, other):
        rois = regionprops(cells, np.moveaxis(img,0,2))
        roi_areas = [x.area for x in rois]
        area_min = np.mean(roi_areas) - np.std(roi_areas)
        rois = [x for x in rois if x.area > area_min]
        obj_df = []
        roi_df = []
        cell_contours = []
        for roi in rois:
            # roi not touching the boundaries
            if np.all(roi.coords > 0) and (np.all(roi.coords[:,0] < img.shape[1]-1)) and (np.all(roi.coords[:,1] < img.shape[2]-1)):
                o, r = self.measure_objects_in_cell(roi, img, cells, nuclei, granules, other)
                if o is not None:
                    obj_df.append(o)
                    roi_df.append(r)
                    cell_contours.append(find_contours(cells==roi.label, 0.5))

        return pd.concat(obj_df), pd.concat(roi_df), cell_contours

    def process(self, index):
        img = self.get_image(index)
        cells = self.segment_cells(img)
        nuclei = self.segment_nuclei(img)
        granules = self.segment_granules(img)
        other = self.segment_other(img)
        O,R,C = self.measure(img,cells,nuclei,granules,other)
        for df in [O,R]:
            df['Condition'] = self.get_condition(index)
            df['Filename'] = self.filelist['Filename'][index]
            df['File ID'] = index

        return O,R,C

def main():
    parser = argparse.ArgumentParser(description='Stress granules analysis')
    parser.add_argument('--file-list',help='filelist')
    parser.add_argument('--data-path',help='path to data')
    parser.add_argument('--index',help='file index',type=int)
    parser.add_argument('--output-by-granules',help='filename of the output table by granule')
    parser.add_argument('--output-by-cells',help='filename of the output table by cell')
    parser.add_argument('--output-cell-contours',help='filename of the output contours file')
    parser.add_argument('--output-vignette',help='filename of the output vignette file')

    args = parser.parse_args()
    print(f'file list {args.file_list}')
    print(f'index {args.index}')
    sga = SGA(args.file_list,args.data_path)
    granules, cells, contours = sga.process(args.index)

    if args.output_by_granules is not None:
        print(f'Saving results by granules to file {args.output_by_granules}')
        granules.to_csv(args.output_by_granules)

    if args.output_by_cells is not None:
        print(f'Saving results by cells to file {args.output_by_cells}')
        cells.to_csv(args.output_by_cells)

    if args.output_cell_contours is not None:
        print(f'Saving cell contours to file {args.output_cell_contours}')
        np.savez(args.output_cell_contours, contours)

    if args.output_vignette is not None:
        print(f'Saving vignette to file {args.output_vignette}')
        img = sga.get_image(args.index)
        visu = sga.render_image(img)
        plt.imshow(visu)
        for c in contours:
            plt.plot(c[0][:,1],c[0][:,0])
        plt.axis('off')
        plt.savefig(args.output_vignette)

if __name__ == "__main__":
    main()

