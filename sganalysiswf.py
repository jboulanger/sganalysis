"""
Stress Granule Analysis in Wide Field Imaging
=============================================

This module provides tools for the analysis of stress granules in wide field microscopy images.
It supports ND2 and LSM file formats, segmentation of cells, nuclei, and granules, and extraction
of quantitative features for downstream analysis.

Main Features
-------------
- Loading and parsing multi-channel, multi-position microscopy images.
- Deconvolution using Richardson-Lucy algorithms.
- Segmentation of cells, nuclei, and granules using Cellpose and watershed.
- Extraction of region properties and statistics.
- Visualization and plotting utilities.
- Batch processing and result aggregation.

Example
-------
>>> python sganalysiswf.py scan --data-path ./images --file-type nd2 --file-list files.csv --config config.json
>>> python sganalysiswf.py process --data-path ./images --file-list files.csv --index 0 --config config.json --output-by-cells results.csv --output-vignette vignette.png
>>> python sganalysiswf.py figure --data-path ./images --file-list files.csv

References
----------
- Cellpose: https://www.cellpose.org/
- NumPy: https://numpy.org/
- Scikit-image: https://scikit-image.org/
- SciPy: https://scipy.org/
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from nd2reader import ND2Reader
import math
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr, spearmanr
from skimage.filters import difference_of_gaussians, laplace
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import watershed
from skimage import morphology
from cellpose import models
from cellpose import core
import cellpose
import edt
import pandas as pd
from pathlib import Path
import os
import json
# import nd2 # did not work on the cluster
import seaborn as sns
import tifffile

__version__ = "2025.11.25"


def get_nd2_number_of_positions(filename):
    """
    Get the number of positions (fields of view) in an ND2 file.

    Parameters
    ----------
    filename : str
        Path to the ND2 file.

    Returns
    -------
    n : int
        Number of positions in the2 file.
    """
    with ND2Reader(filename) as images:
        n = images.sizes["v"]
    return n


def load_nd2(filename, fov=0):
    """
    Load all z planes and channels images from a multi-position ND2 file.

    Parameters
    ----------
    filename : str
        Path to the ND2 file.
    fov : int, optional
        Field of view index to load (default is 0).

    Returns
    -------
    data : ndarray
        Image data of shape (z, c, y, x).
    pixel_size : list of float
        Pixel size in [z, y, x] (nanometers).
    """
    planes = []
    with ND2Reader(filename) as images:
        md = images.metadata
        pixel_size = [
            (md["z_coordinates"][2] - md["z_coordinates"][1]) * 1000,
            md["pixel_microns"] * 1000,
            md["pixel_microns"] * 1000,
        ]
        for z in range(images.sizes["z"]):
            for c in range(images.sizes["c"]):
                planes.append(images.get_frame_2D(c=c, t=0, z=z, x=0, y=0, v=fov))
    shp = [images.sizes["z"], images.sizes["c"], images.sizes["y"], images.sizes["x"]]
    return np.reshape(np.stack(planes), shp), pixel_size


def load_lsm(filename, fov):
    """
    Load image data from an LSM file.

    Parameters
    ----------
    filename : str
        Path to the LSM file.
    fov : int
        Field of view index to load.

    Returns
    -------
    data : ndarray
        Image data.
    pixel_size : list of float
        Pixel size in [z, y, x] (nanometers).
    """
    with tifffile.TiffFile(filename) as tif:
        data = tif.asarray()
        md = tif.lsm_metadata
        pixel_size = [
            md["VoxelSizeZ"] * 1e9,
            md["VoxelSizeY"] * 1e9,
            md["VoxelSizeX"] * 1e9,
        ]
        if md["DimensionP"] == 1:
            return data, pixel_size
        else:
            return data[fov], pixel_size


def load_tiff(filename: str, fov: int):
    """
    Load image data from a TIFF file.

    Parameters
    ----------
    filename : str
        Path to the LSM file.
    fov : int
        Field of view index to load.

    Returns
    -------
    data : ndarray
        Image data.
    pixel_size : list of float
        Pixel size in [z, y, x] (nanometers).
    """
    with tifffile.TiffFile(filename) as tif:
        data = tif.asarray()

        if tif.is_imagej:
            md = tif.imagej_metadata
            print(md["spacing"], md["unit"])
            if md["unit"] == "nm":
                factor = 1
            elif md["unit"] == "micron" or md["unit"] == "\\u00B5m":
                factor = 1e3
            elif md["unit"] == "m":
                factor = 1e9
            else:
                raise ValueError(f"Unsupported unit ({md['unit']}) in TIFF file.")

            x = (
                tif.pages[0].tags["XResolution"].value[1]
                / tif.pages[0].tags["XResolution"].value[0]
                * factor
            )

            y = (
                tif.pages[0].tags["YResolution"].value[1]
                / tif.pages[0].tags["YResolution"].value[0]
                * factor
            )

            z = md["spacing"] * factor
        else:
            raise ValueError("Unsupported tiff format.")

        pixel_size = [z, y, x]
        print(data.shape, pixel_size)
        if data.ndim == 3:
            return data, pixel_size
        else:
            return data[fov], pixel_size


def load_image(filename, fov):
    """
    Load image data from ND2 or LSM file depending on file extension.

    Parameters
    ----------
    filename : str
        Path to the image file.
    fov : int
        Field of view index.

    Returns
    -------
    data : ndarray
        Image data.
    pixel_size : list of float
        Pixel size in [z, y, x] (nanometers).
    """
    path = Path(filename)
    if path.suffix == ".nd2":
        return load_nd2(filename, fov)
    elif path.suffix == ".lsm":
        return load_lsm(filename, fov)
    else:
        return load_tiff(filename, fov)


def generate_otf3d(
    shape, pixel_size, wavelength, numerical_aperture, medium_refractive_index
):
    """
    Generate a diffraction limited wide field optical transfer function and point spread function.

    Parameters
    ----------
    shape : list of int
        [nz, ny, nx] giving the shape of the final array.
    pixel_size : list of float
        Sampling in [z, y, x].
    wavelength : float
        Wavelength of the emitted light.
    numerical_aperture : float
        Numerical aperture.
    medium_refractive_index : float
        Refractive index of the immersion medium.

    Returns
    -------
    otf : ndarray
        The optical transfer function as an array of shape 'shape'.
    psf : ndarray
        The point spread function as an array of shape 'shape' centered in 0,0,0.
    """
    kx = np.reshape(np.fft.fftfreq(shape[2], pixel_size[2]), [1, 1, shape[2]])
    ky = np.reshape(np.fft.fftfreq(shape[1], pixel_size[1]), [1, shape[1], 1])
    z = np.reshape(
        np.concatenate((np.arange(0, shape[0] // 2), np.arange((-shape[0]) // 2, 0)))
        * pixel_size[0],
        [shape[0], 1, 1],
    )
    d2 = np.square(kx) + np.square(ky)
    rho = np.sqrt(d2) * (wavelength / numerical_aperture)
    # conservation of energy
    corr = np.power(
        np.maximum(1 - d2 / (medium_refractive_index / wavelength) ** 2, 1e-3), -0.25
    )
    P = np.where(rho <= 1.0, 1.0, 0.0)  # * corr
    defocus = z * np.sqrt(
        np.clip((medium_refractive_index / wavelength) ** 2 - d2, 0, None)
    )
    psf = np.square(
        np.abs(np.fft.fft2(P * np.exp(2j * math.pi * defocus), axes=[1, 2]))
    )
    psf = psf / psf.sum()
    otf = np.fft.fftn(psf)
    return otf, psf


def deconvolve_richardson_lucy(data, otf, background=0, iterations=100):
    """
    Deconvolve data according to the given OTF using a Richardson-Lucy algorithm.

    Parameters
    ----------
    data : ndarray
        Input image data.
    otf : ndarray
        Optical transfer function of the same size as data.
    background : float, optional
        Background level (default is 0).
    iterations : int, optional
        Number of iterations (default is 100).

    Returns
    -------
    estimate : ndarray
        Estimated image.
    """
    estimate = np.clip(
        np.real(np.fft.ifftn(otf * np.fft.fftn(data - background))), 1e-6, None
    )
    for _ in range(iterations):
        blurred = np.clip(
            np.real(np.fft.ifftn(otf * np.fft.fftn(estimate + background))), 1e-6, None
        )
        ratio = data / blurred
        estimate = estimate * np.real(np.fft.ifftn(otf * np.fft.fftn(ratio)))
    return estimate


def deconvolve_richardson_lucy_heavy_ball(data, otf, background, iterations):
    """
    Deconvolve data according to the given OTF using a scaled heavy ball Richardson-Lucy algorithm.

    Parameters
    ----------
    data : ndarray
        Input image data.
    otf : ndarray
        Optical transfer function of the same size as data.
    background : float
        Background level.
    iterations : int
        Number of iterations.

    Returns
    -------
    estimate : ndarray
        Estimated image.
    dkl : ndarray
        Kullback-Leibler divergence per iteration.

    Notes
    -----
    See: https://doi.org/10.1109/tip.2013.2291324
    """
    old_estimate = np.clip(
        np.real(np.fft.ifftn(otf * np.fft.fftn(data - background))), a_min=0, a_max=None
    )
    estimate = data
    dkl = np.zeros(iterations)
    for k in range(iterations):
        beta = (k - 1.0) / (k + 2.0)
        prediction = estimate + beta * (estimate - old_estimate)
        blurred = np.clip(
            np.real(np.fft.ifftn(otf * np.fft.fftn(prediction + background))),
            a_min=1e-6,
            a_max=None,
        )
        ratio = data / blurred
        gradient = 1.0 - np.real(np.fft.ifftn(otf * np.fft.fftn(ratio)))
        old_estimate = estimate
        estimate = np.clip(prediction - estimate * gradient, a_min=0.1, a_max=None)
        dkl[k] = np.mean(
            blurred - data + data * np.log(np.clip(ratio, a_min=1e-6, a_max=None))
        )
    return estimate, dkl


def correct_hotpixels_inplace(data):
    """
    Correct hot pixels in the image data in-place using a median filter.

    Parameters
    ----------
    data : ndarray
        Image data to be corrected.
    """
    baseline = ndimage.median_filter(data, size=3)
    delta = data - baseline
    thres = delta.mean() + delta.std()
    delta = np.abs(delta) > thres
    data[delta] = baseline[delta]


def deconvolve_all_channels(data, pixel_size, config):
    """
    Deconvolve all channels in the image data.

    Parameters
    ----------
    data : ndarray
        Image data of shape (z, c, y, x).
    pixel_size : list of float
        Pixel size in [z, y, x] (nanometers).
    config : dict
        Configuration dictionary with channel wavelengths, NA, and medium refractive index.

    Returns
    -------
    dec : ndarray
        Deconvolved image data.
    """
    print("Deconvolve all channels")
    wavelengths = [c["wavelength"] for c in config["channels"]]
    NA = config["NA"]
    medium_refractive_index = config["medium_refractive_index"]
    dec = []
    for k in range(data.shape[1]):
        img = np.squeeze(
            data[
                :,
                k,
                :,
                :,
            ]
        )
        correct_hotpixels_inplace(img)
        otf, psf = generate_otf3d(
            img.shape, pixel_size, wavelengths[k], NA, medium_refractive_index
        )
        dec.append(deconvolve_richardson_lucy_heavy_ball(img, otf, img.min(), 10))
    return np.stack(dec, axis=1)


def projection(data):
    """
    Compute maximum intensity projection for each channel.

    Parameters
    ----------
    data : ndarray
        Image data of shape (z, c, y, x) or (c, y, x).

    Returns
    -------
    mip : ndarray
        Projected image data of shape (c, y, x).
    """
    if len(data.shape) == 4:
        mip = []
        for k in range(data.shape[1]):
            w = np.exp(-0.1 * np.abs(laplace(data[:, k, :, :])))
            mip.append(
                np.mean(np.squeeze(data[:, k, :, :]) * w, axis=0) / np.mean(w, axis=0)
            )
        return np.stack(mip, axis=0)
    else:
        return data


def segment_cells(img, pixel_size, scale, mode):
    """
    Segment the cells in the image using Cellpose or Cellpose + watershed.

    Parameters
    ----------
    img : ndarray
        Image (C, H, W) with membrane [idx 0] and nuclei channels [idx 1].
    pixel_size : list of float
        Pixel size as [pz, py, px].
    scale : float
        Scale in microns.
    mode : int
        Type 0: Cellpose, 1: Cellpose + watershed.

    Returns
    -------
    clabels : ndarray
        Label array (H, W).
    """
    print(f" - Segmenting cells with mode {mode}")



    if mode == 1:
        d = 0.33 * 1000 * scale / pixel_size[-1]
        model = models.CellposeModel(gpu=core.use_gpu(), model_type="nuclei")
        nlabels = model.eval(
            img[1],
            diameter=d,
            flow_threshold=None,
            cellprob_threshold=0.1,
            # min_size=10000,
        )[0]

        dist = distance_transform_edt(nlabels > 0)
        mask = gaussian_filter(np.amax(img, 0), 15)
        mask = mask > mask[mask < np.quantile(mask, 0.1)].mean()
        clabels = watershed(-dist, nlabels, mask=mask)
    else:
        d = round(1000 * scale / pixel_size[-1])
        print(f"    Cell size {d}")
        print(f"    Image shape {img.shape}")
        model = models.CellposeModel(gpu=core.use_gpu(), model_type="cyto2")
        clabels = model.eval(
            img,
            diameter=d,
            # min_size=10000,
        )[0]

    return clabels


def segment_nuclei(img, pixel_size, scale):
    """
    Segment nuclei in the image using Cellpose.

    Parameters
    ----------
    img : ndarray
        Nuclei channel image.
    pixel_size : list of float
        Pixel size as [pz, py, px].
    scale : float
        Scale in microns.

    Returns
    -------
    mask : ndarray
        Nuclei mask.
    """
    print(" - Segmenting nuclei")
    d = 0.33 * 1000 * scale / pixel_size[-1]
    
    model = models.CellposeModel(gpu=core.use_gpu(), model_type="nuclei")
    mask = model.eval(img, diameter=d, flow_threshold=None)[0]
    return mask


def segment_granules(img):
    """
    Segment granules in the image using difference of Gaussians and morphology.

    Parameters
    ----------
    img : ndarray
        Granule channel image.

    Returns
    -------
    mask : ndarray
        Labeled granule mask.
    """
    print(" - Segmenting granule")
    flt = difference_of_gaussians(np.sqrt(img.astype(float)), 2, 4)
    t = flt.mean() + 2 * flt.std()
    mask = morphology.remove_small_holes(
        morphology.remove_small_objects(flt > t, min_size=5),
    )
    mask = morphology.opening(mask, morphology.disk(3))
    return label(mask).astype(np.uint)


def segment_image(img, pixel_size, config):
    """
    Segment images into cells, nuclei, granules, and other structures.

    Parameters
    ----------
    img : dict
        Dictionary of images by channel.
    pixel_size : list of float
        Pixel size as [pz, py, px].
    config : dict
        Configuration dictionary.

    Returns
    -------
    labels : dict
        Dictionary of label arrays.
    """
    print("Segmenting images ")
    scale = config["scale_um"]
    mode = config["mode"]
    if config["Analysis"] == "SG":
        tmp = img["membrane"]  # + img["granule"] + img["other"]
        # tmp = gaussian_filter(tmp, 20)
        print(f"   image shape {tmp.shape}")
        labels = {
            "cells": segment_cells(
                np.stack([tmp, img["nuclei"]], axis=0), pixel_size, scale, mode
            ),
            "nuclei": segment_nuclei(img["nuclei"], pixel_size, scale),
            "granule": segment_granules(img["granule"]),
            "other": segment_granules(img["other"]),
        }
    else:
        tmp = sum([img[c] for c in img if c != "nuclei"])
        tmp = ndimage.minimum_filter(ndimage.median_filter(tmp, 5), 11)
        labels = {
            "cells": segment_cells(
                np.stack([tmp, img["nuclei"]]), pixel_size, scale, mode
            ),
            "nuclei": segment_nuclei(img["nuclei"], pixel_size, scale),
        }
    return labels


def spatial_spread_roi(prop, image):
    """
    Compute spread as the trace of the moment matrix in a region for all channels.

    Parameters
    ----------
    prop : regionprops
        A single regionprop object.
    image : ndarray
        Image data (C, H, W).

    Returns
    -------
    S : list of tuple
        List of (centroid x, centroid y, sxx, sxy, syy) for each channel.
    """
    y, x = np.meshgrid(
        np.arange(prop.bbox[0], prop.bbox[2]),
        np.arange(prop.bbox[1], prop.bbox[3]),
        indexing="ij",
    )
    image = image[:, prop.bbox[0] : prop.bbox[2], prop.bbox[1] : prop.bbox[3]]
    S = []
    for weight in image:
        w = weight * (weight > (weight.mean() + weight.std())) * prop.image
        if (w.max() - w.min()) > 0:
            w = (w - w.min()) / (w.max() - w.min())
        else:
            w = prop.image
        sw = np.sum(w)
        if sw < 1e-9:
            return 0.0
        sx = np.sum(w * x) / sw
        sy = np.sum(w * y) / sw
        sxx = np.sum(w * np.square(x - sx)) / sw
        sxy = np.sum(w * (x - sx) * (y - sy)) / sw
        syy = np.sum(w * np.square(y - sy)) / sw
        S.append((sx, sy, sxx, sxy, syy))
    return S


def spatial_spread_mask(mask, intensity):
    """
    Compute spread as the trace of the moment matrix.

    Parameters
    ----------
    mask : ndarray
        Mask on which to compute the spread (cell).
    intensity : ndarray
        Intensity in the selected channel.

    Returns
    -------
    sx, sy, sxx, sxy, syy : float
        Moments of the region.
    """
    x, y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    w = ma.array(intensity - intensity.min(), mask=np.logical_not(mask))
    x = ma.array(x, mask=np.logical_not(mask))
    y = ma.array(y, mask=np.logical_not(mask))
    sw = np.sum(w)
    if sw < 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sx = np.sum(w * x) / sw
    sy = np.sum(w * y) / sw
    sxx = np.sum(w * np.square(x - sx)) / sw
    sxy = np.sum(w * (x - sx) * (y - sy)) / sw
    syy = np.sum(w * np.square(y - sy)) / sw
    return sx, sy, sxx, sxy, syy


def characteristic_radius(mask, intensity, fraction):
    """
    Compute radius corresponding to a fraction of the total intensity.

    Parameters
    ----------
    mask : ndarray
        Mask of the region.
    intensity : ndarray
        Intensity image.
    fraction : float
        Fraction of total intensity.

    Returns
    -------
    r0 : float
        Characteristic radius.
    """
    x0 = np.arange(mask.shape[1])
    y0 = np.arange(mask.shape[0])
    x, y = np.meshgrid(x0, y0)
    w = ma.array(intensity - intensity.min(), mask=np.logical_not(mask))
    x = ma.array(x, mask=np.logical_not(mask))
    y = ma.array(y, mask=np.logical_not(mask))
    sw = np.sum(w)
    ret = np.unravel_index(np.argmax(w), w.shape)
    sx1 = ret[1]
    sy1 = ret[0]
    d = np.sqrt(np.square(x - sx1) + np.square(y - sy1))
    r = np.linspace(0, d.max(), 100)
    F = np.array([(w * (d < v)).sum() / sw for v in r])
    r0 = np.interp(fraction, F, r)
    return r0


def fraction_in_spot(mask, intensity):
    """
    Compute fraction of the intensity in spot-like structures.

    Parameters
    ----------
    mask : ndarray
        Mask of the region.
    intensity : ndarray
        Intensity image.

    Returns
    -------
    fraction : float
        Fraction of intensity in spots.
    """
    score = gaussian_filter(intensity.astype(float), 2) - gaussian_filter(
        intensity.astype(float), 6
    )
    m = np.median(score)
    s = 1.48 * np.median(np.abs(score - m))
    score = score > (m + 6 * s)
    score = score * mask
    return (intensity * score).sum() / (intensity * mask).sum()


def does_not_touch_image_border(roi, img):
    """
    Return True if the ROI does not touch the image border.

    Parameters
    ----------
    roi : regionprops
        ROI object.
    img : dict
        Dictionary of images.

    Returns
    -------
    bool
        True if ROI does not touch border, False otherwise.
    """
    shape = img["nuclei"].shape
    return (
        np.all(roi.coords > 0)
        and np.all(roi.coords[:, 0] < shape[0] - 1)
        and np.all(roi.coords[:, 1] < shape[1] - 1)
    )


def strech_range(x):
    """
    Stretch the range of the array between 0 and 1 for display.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    y : ndarray
        Stretched array.
    """
    a = x.mean() + 3 * x.std()
    b = np.median(x)
    return np.clip((x - b) / (a - b), 0, 1)


def draw_spread(stats, channel, color, bbox=[0, 0, 1, 1]):
    """
    Draw spread circles on the plot for a given channel.

    Parameters
    ----------
    stats : DataFrame or dict
        Statistics for the ROIs.
    channel : str
        Channel name.
    color : str
        Color for the circles.
    bbox : list, optional
        Bounding box for offset (default is [0, 0, 1, 1]).
    """
    X = stats[f"Centroid X {channel}"] - bbox[1]
    Y = stats[f"Centroid Y {channel}"] - bbox[0]
    S = stats[f"Spread {channel}"]
    try:
        for x, y, s in zip(X, Y, S):
            c = plt.Circle((x, y), s, fill=False, color=color, alpha=0.75, lw=2)
            plt.gca().add_patch(c)
    except Exception as e:
        print(e)
        pass
    try:
        c = plt.Circle((X, Y), S, fill=False, color=color, alpha=0.75, lw=3)
        plt.gca().add_patch(c)
    except Exception as e:
        print(e)
        pass


def img2rgb(img):
    """
    Convert a dictionary of images to an RGB (H, W, 3) array.

    Parameters
    ----------
    img : dict
        Dictionary of images by channel.

    Returns
    -------
    visu : ndarray
        RGB image array.
    """
    visu = np.zeros([img["nuclei"].shape[0], img["nuclei"].shape[1], 3])
    if "granule" in img.keys():
        tmp = strech_range(img["granule"])
        visu[:, :, 0] = 0.5 * (strech_range(img["other"]) + tmp)
        visu[:, :, 1] = 0.5 * (strech_range(img["membrane"]) + tmp)
        visu[:, :, 2] = 0.5 * (strech_range(img["nuclei"]) + tmp)
    else:
        visu[:, :, 2] = strech_range(img["nuclei"])
        n = 0
        for c in img.keys():
            if c != "nuclei" and n <= 2:
                visu[:, :, n] = strech_range(img[c])
                n = n + 1
    return visu


def show_image(img, labels, rois, stats):
    """
    Show the image with labels and ROIs.

    Parameters
    ----------
    img : dict
        Dictionary of images by channel.
    labels : dict
        Dictionary of label arrays.
    rois : list
        List of regionprops objects.
    stats : DataFrame or dict
        ROI statistics.
    """
    visu = img2rgb(img)
    plt.imshow(visu)

    for r in rois:
        try:
            c = find_contours(ndimage.binary_erosion(labels["cells"] == r.label), 0.5)
            plt.plot(c[0][:, 1], c[0][:, 0])
            plt.text(r.centroid[1], r.centroid[0], f"{r.label}", color="white")
        except Exception as e:
            print(e)
            print("failed to show cell")

    for r in np.unique(labels["nuclei"]):
        try:
            if r > 0:
                c = find_contours(labels["nuclei"] == r, 0.5)
                plt.plot(c[0][:, 1], c[0][:, 0], "w", alpha=0.5)
        except Exception as e:
            print(e)
            print("Failed to show nuclei")
    try:
        if "granule" in img.keys():
            pass
        else:
            color = ["red", "green", "blue", "white"]
            n = 0
            for c in img.keys():
                if c != "nuclei":
                    draw_spread(stats, f"of {c}", color[n])
                    n = n + 1
    except Exception as e:
        print(e)
        print("Failed to draw spread")

    plt.axis("off")


def show_roi(roi, img, labels, stats):
    """
    Show a single ROI.

    Parameters
    ----------
    roi : regionprops
        ROI object.
    img : dict
        Dictionary of images by channel.
    labels : dict
        Dictionary of label arrays.
    stats : DataFrame or dict
        ROI statistics.
    """
    masks, img = compute_roi_masks(roi, labels, img)

    visu = img2rgb(img)
    plt.imshow(visu)

    c = find_contours(masks["cell"] == 1, 0.5)
    plt.plot(c[0][:, 1], c[0][:, 0], "w")

    if "granule" in img.keys():
        draw_spread(stats, "in cells of granule channel", "red")
    else:
        color = ["red", "green", "blue", "white"]
        n = 0
        for c in img.keys():
            if c != "nuclei":
                draw_spread(stats, f"of {c}", color[n], roi.bbox)
                n = n + 1

    plt.axis("off")
    plt.title(f"ROI {roi.label}")


def compute_roi_masks(roi, labels, img, border=20):
    """
    Compute a mask for each ROI.

    Parameters
    ----------
    roi : regionprops
        ROI object.
    labels : dict
        Dictionary of label arrays.
    img : dict
        Dictionary of images by channel.
    border : int, optional
        Border around the ROI (default is 20).

    Returns
    -------
    mask : dict
        Dictionary of cropped masks.
    img_crop : dict
        Dictionary of cropped images.
    """
    crop = {
        k: labels[k][
            roi.bbox[0] - border : roi.bbox[2] + border,
            roi.bbox[1] - border : roi.bbox[3] + border,
        ]
        for k in labels
    }
    img_crop = {
        k: img[k][
            roi.bbox[0] - border : roi.bbox[2] + border,
            roi.bbox[1] - border : roi.bbox[3] + border,
        ]
        for k in img
    }
    not_nuclei = crop["nuclei"] == 0
    cell = crop["cells"] == roi.label

    if "granule" in labels.keys():
        mask = {
            "nucleus": cell * crop["nuclei"],
            "cell": cell,
            "particle": cell * not_nuclei * crop["granule"],
            "cytosol": cell * not_nuclei * (crop["granule"] == 0),
            "other": cell * not_nuclei * crop["other"],
        }
    else:
        mask = {"nucleus": cell * crop["nuclei"], "cell": cell}

    return mask, img_crop


def compute_roi_distance(masks):
    """
    Compute distance maps for nuclei and membrane.

    Parameters
    ----------
    masks : dict
        Dictionary with nucleus and cell masks.

    Returns
    -------
    distances : dict
        Dictionary with distance to nucleus, membrane, and fraction.
    """
    d1 = edt.edt(1 - (masks["nucleus"] > 0).astype(int))
    d2 = edt.edt(masks["cell"])
    d1[np.logical_not(np.isfinite(d1))] = 0
    d2[np.logical_not(np.isfinite(d2))] = 0
    n = np.sqrt(d1 * d1 + d2 * d2)
    distances = {
        "nucleus": d1,
        "membrane": d2,
        "fraction": np.divide(d1, n, where=(np.abs(n) > 1)),
    }
    return distances


def manders_coefficients(mask1, mask2, im1, im2):
    """
    Compute Manders overlap coefficients.

    Parameters
    ----------
    mask1 : ndarray
        First mask.
    mask2 : ndarray
        Second mask.
    im1 : ndarray
        First image.
    im2 : ndarray
        Second image.

    Returns
    -------
    m1, m2 : float
        Manders coefficients.
    """
    intersect = np.logical_and(mask1, mask2)
    n1 = np.sum(mask1 * im1, dtype=float)
    n2 = np.sum(mask2 * im2, dtype=float)
    m1 = np.sum(intersect * im1, dtype=float) / n1 if n1 > 0 else 0
    m2 = np.sum(intersect * im2, dtype=float) / n2 if n2 > 0 else 0
    return (m1, m2)


def measure_roi_stats(roi, img, masks, distances):
    """
    Measure statistics for a given ROI.

    Parameters
    ----------
    roi : regionprops
        ROI object.
    img : dict
        Dictionary of images with keys cells, nuclei, granule, other.
    masks : dict
        Dictionary of masks with keys nucleus, cell, particle, cytosol, other.
    distances : dict
        Dictionary of distance maps with keys nucleus, membrane.

    Returns
    -------
    stats : dict
        Dictionary of measured statistics.
    """
    particles = regionprops(masks["particle"])
    particles = [
        r
        for r in particles
        if r.perimeter_crofton > 1 and r.area > 1 and r.major_axis_length > 1
    ]

    nuclei = regionprops(masks["nucleus"])

    stats = {
        "Cell ID": roi.label,
        "Area of the whole cell [px^2]": roi.area,
        "Area of the cell without nuclei [px^2]": np.sum(masks["cell"], dtype=float)
        - np.sum(masks["nucleus"] > 0, dtype=float),
        "Area of all particles [px^2]": np.sum(masks["particle"] > 0, dtype=float),
        "Area of cytosol [px^2]": np.sum(masks["cytosol"] > 0, dtype=float),
        "Area ratio particles:cell": np.sum(masks["particle"] > 0, dtype=float)
        / np.sum(masks["cell"], dtype=float),
        "Area ratio cytosol:cell": np.sum(masks["cytosol"], dtype=float)
        / np.sum(masks["cell"], dtype=float),
        "Area ratio particles:cytosol": np.sum(masks["particle"] > 0, dtype=float)
        / np.sum(masks["cytosol"], dtype=float),
    }

    for m in masks:
        sum_mask = np.sum(masks[m] > 0, dtype=float)
        for c in ["granule", "other"]:
            sum_mask_x_img = (
                (masks[m] > 0).astype(float) * (img[c].astype(float))
            ).sum()
            stats["Total intensity in " + m + " of " + c + " channel"] = sum_mask_x_img
            stats["Mean intensity in " + m + " of " + c + " channel"] = (
                sum_mask_x_img / sum_mask if sum_mask > 0 else 0
            )

    for c in ["granule", "other"]:
        bot = stats["Mean intensity in cytosol of " + c + " channel"]
        top = stats["Mean intensity in particle of " + c + " channel"]
        stats["Mean intensity ratio particle:cytosol of channel other"] = (
            top / bot if bot > 0 else 0
        )
        tmp = gaussian_filter(img[c], 10)
        sc = spatial_spread_mask(masks["cell"], tmp)
        stats["Centroid X in cells of " + c + " channel"] = roi.bbox[1] + sc[0]
        stats["Centroid Y in cells of " + c + " channel"] = roi.bbox[0] + sc[1]
        stats["Spread in cells of " + c + " channel"] = np.sqrt(sc[2] + sc[4])
        sp = spatial_spread_mask(masks["particle"], tmp)
        stats["Spread in particles of " + c + " channel"] = np.sqrt(sp[2] + sp[4])

    I1 = img["granule"][masks["cell"]].astype(float)
    I2 = img["other"][masks["cell"]].astype(float)
    stats["Colocalization spearman granule:other"] = spearmanr(I1, I2)[0]
    stats["Colocalization pearson granule:other"] = pearsonr(I1, I2)[0]

    m1, m2 = manders_coefficients(
        masks["particle"], masks["other"], img["granule"], img["other"]
    )
    stats["Colocalization manders m1 granule:other"] = m1
    stats["Colocalization manders m2 granule:other"] = m2
    stats["Number of particles"] = len(particles)
    stats["Particle area"] = (
        np.sum(np.array([x.area for x in particles])) if len(particles) > 0 else 0
    )
    stats["Particle area fraction"] = (
        np.sum(np.array([x.area for x in particles])) / roi.area
        if len(particles) > 0
        else 0
    )
    stats["Average particle area"] = (
        np.mean(np.array([x.area for x in particles])) if len(particles) > 0 else 0
    )
    stats["Average particle perimeter"] = (
        np.mean(np.array([x.perimeter_crofton for x in particles]))
        if len(particles) > 0
        else 0
    )
    stats["Average particles distance to nuclei"] = (
        np.sum(
            distances["nucleus"] * (masks["particle"] > 0).astype(float), dtype=float
        )
        / np.sum(masks["particle"] > 0, dtype=float)
        if len(particles) > 0
        else 0
    )
    stats["Average particles distance to membrane"] = (
        np.sum(distances["membrane"] * (masks["particle"] > 0), dtype=float)
        / np.sum(masks["particle"] > 0, dtype=float)
        if len(particles) > 0
        else 0
    )
    stats["Average particles distance ratio"] = (
        np.sum(distances["fraction"] * (masks["particle"] > 0), dtype=float)
        / np.sum(masks["particle"] > 0, dtype=float)
        if len(particles) > 0
        else 0
    )
    stats["Average particles circularity"] = (
        np.mean(
            np.array(
                [4.0 * math.pi * x.area / x.perimeter_crofton**2 for x in particles]
            )
        )
        if len(particles) > 0
        else 0
    )
    stats["Average particles aspect ratio"] = (
        np.mean(
            np.array([x.minor_axis_length / x.major_axis_length for x in particles])
        )
        if len(particles) > 0
        else 0
    )
    stats["Average particles solidity"] = (
        np.mean(np.array([x.solidity for x in particles])) if len(particles) > 0 else 0
    )
    stats["Average particles roundness"] = (
        np.mean(
            np.array(
                [4.0 * x.area / (math.pi * x.major_axis_length**2) for x in particles]
            )
        )
        if len(particles) > 0
        else 0
    )
    stats["Number of nuclei"] = len(nuclei)
    return stats


def measure_roi_spread(roi, img, masks, distances):
    """Measure spread statisics

    Parameters
    ----------
    roi : roi from regionprops
    img : dictionnary of images with keys cells,...
    masks :  dictionnary of images with keys nucleus, cell,...
    distances : dictionnary of distances map with keys nuclei,membrane,fraction
    """

    nuclei = regionprops(masks["nucleus"])

    sum_mask = np.sum(masks["cell"] > 0, dtype=float)
    sum_nuc = np.sum(masks["nucleus"] > 0, dtype=float)

    stats = {
        "Cell ID": roi.label,
        "Area of the whole cell [px^2]": sum_mask,
        "Area of the nuclei [px^2]": sum_nuc,
        "Number of nuclei": len(nuclei),
    }

    # measure intensity and spread in channels
    for c in img:
        sum_mask_x_img = (
            (masks["cell"] > 0).astype(float) * (img[c].astype(float))
        ).sum()
        stats[f"Total intensity in {c}"] = sum_mask_x_img
        stats[f"Mean intensity in {c}"] = (
            sum_mask_x_img / sum_mask if sum_mask > 0 else 0
        )
        tmp = (img[c] * masks["cell"]).astype(float)
        tmp = gaussian_filter(tmp, 5)
        # tmp = white_tophat(tmp, 10)
        tmp = np.maximum(tmp - np.median(tmp) - tmp.std(), 0)
        # plt.figure()
        # plt.imshow(tmp)
        sc = spatial_spread_mask(masks["cell"], tmp)
        stats["Centroid X of " + c] = roi.bbox[1] + sc[0]
        stats["Centroid Y of " + c] = roi.bbox[0] + sc[1]
        stats["Spread of " + c] = np.sqrt(sc[2] + sc[4])
        stats["Radius 0.1 of " + c] = characteristic_radius(masks["cell"], img[c], 0.1)
        stats["Spot fraction of " + c] = fraction_in_spot(masks["cell"], img[c])
    return stats


def load_config(path):
    return json.load(open(path))


def config2img(img, config):
    dst = dict()
    for c in config:
        dst[c["name"]] = img[c["index"]]
    return dst


def process_fov(filename: Path, position: int, config):
    """Process the field of view

    Parameter
    ---------
    filename: Path or str
        image filename
    position: int
        position in the multiposition file
    config: configuration with channel, scale and mode

    Returns
    -------
    pd.DataFrame
        dataframe with statistics for each ROI
    np.ndarray
        maximum intensity projection
    np.ndarray
        segmentation labels
    rois: list of RegionsProperties
        list of regionprops objects for each cell
    """
    print("Processing [SG]")

    data, pixel_size = load_image(filename, position)

    print(f"  pixel size {pixel_size}")

    # data = deconvolve_all_channels(data,pixel_size,config)

    mip = config2img(projection(data), config["channels"])

    labels = segment_image(mip, pixel_size, config)

    # define the list of ROI corresponding to the cells
    rois = regionprops(labels["cells"])

    # discard ROI touching border
    rois = [x for x in rois if does_not_touch_image_border(x, mip)]

    # discard too small cells ROI
    amin = 3.14 * (0.1 * config["scale_um"] * 1e3 / pixel_size[1]) ** 2
    rois = [x for x in rois if x.area > amin]

    print(" - Compute ROI statistics")
    stats = []
    for k, roi in enumerate(rois):
        try:
            masks, crop_img = compute_roi_masks(roi, labels, mip, border=0)
            distances = compute_roi_distance(masks)
            stats.append(measure_roi_stats(roi, crop_img, masks, distances))
        except Exception as e:
            print(e)
            print("Error encountered for ROI", k)

    stats = pd.DataFrame(stats)

    return stats, mip, labels, rois


def process_fov_spread(filename, position, config):
    """Process the field of view with spread analsysis

    Parameters
    ----------
    filename : name of the file
    position : index of the field of view
    config   : dictionnary with the field of view

    Result
    ------
    stats, mip, labels, rois
    """

    print("Processing [Spread]")

    data, pixel_size = load_image(filename, position)

    mip = config2img(projection(data), config["channels"])

    labels = segment_image(mip, pixel_size, config)

    # define the list of ROI corresponding to the cells
    rois = regionprops(labels["cells"])

    # discard ROI touching border
    rois = [x for x in rois if does_not_touch_image_border(x, mip)]

    # discard too small cells ROI
    amin = 3.14 * (0.1 * config["scale_um"] * 1e3 / pixel_size[1]) ** 2
    rois = [x for x in rois if x.area > amin]

    print(" - Compute ROI statistics")
    stats = []
    for k, roi in enumerate(rois):
        try:
            masks, crop_img = compute_roi_masks(roi, labels, mip, border=0)
            distances = compute_roi_distance(masks)
            stats.append(measure_roi_spread(roi, crop_img, masks, distances))
        except Exception as e:
            print(e)
            print("Error encountered for ROI", k)

    stats = pd.DataFrame(stats)

    return stats, mip, labels, rois


def scan_folder_nd2(folder: Path):
    """List the field of views of a ND2 file

    Parameters
    ----------
    folder: Path
        Folder where nd2 files are

    Returns
    -------
    pd.Dataframe
        Dataframe
    """
    L = []
    for file in folder.glob("*.nd2"):
        condition = file.split("_")[1].replace("Well", "")
        try:
            with ND2Reader(file) as images:
                for fov in range(images.sizes["v"]):
                    L.append(
                        {
                            "filename": file.name,
                            "fov": fov,
                            "condition": condition,
                            "channels": images.sizes["c"],
                        }
                    )
        except Exception as e:
            print(e)
            print("An error occured on this file " + str(file))
    return pd.DataFrame(L)


def scan_folder_lsm(folder: Path):
    """list the field of views of a LSM file

    Parameters
    ----------
    folder: Path
        Folder where lsm files are

    Returns
    -------
    pd.DataFrame
        Dataframe with filename, fov, condition and channels
    """
    L = []
    for file in folder.glob("*.lsm"):
        try:
            with tifffile.TiffFile(file) as tif:
                for fov in range(tif.lsm_metadata["DimensionP"]):
                    L.append(
                        {
                            "filename": file.name,
                            "fov": fov,
                            "condition": "unknown",
                            "channels": tif.lsm_metadata["DimensionChannels"],
                        }
                    )
        except Exception as e:
            print(e)
            print("An error occured on this file " + str(file))
    return pd.DataFrame(L)


def scan_folder_tiff(folder: Path):
    """list the field of views of a LSM file

    Parameters
    ----------
    folder: Path
        Folder where TIFF files are

    Returns
    -------
    pd.DataFrame
        Dataframe with filename, fov, condition and channels
    """
    # list of files
    L = []
    for file in folder.glob("*.tif"):
        if not file.name.startswith("."):
            try:
                with tifffile.TiffFile(file) as tif:
                    md = tif.imagej_metadata
                    # we use frame as positions
                    for fov in range(md["frames"]):
                        L.append(
                            {
                                "filename": file.name,
                                "fov": fov,
                                "condition": "unknown",
                                "channels": md["channels"],
                            }
                        )
            except Exception as e:
                print(e)
                print("An error occured on this file " + str(file))
    return pd.DataFrame(L)


def scan(args):
    """Scan a folder of nd2 files and list the field of views (fov)

    Parameters
    ----------
    args: argparse.Namespace or Path
        If Namespace, it should contain data_path, file_type and file_list.
        If Path, it should be the folder to scan.

    Returns
    -------
    pd.DataFrame
        DataFrame with filename, fov, condition and channels.
    """

    print("[ Scan ]")

    if isinstance(args, argparse.Namespace):
        folder = Path(args.data_path)
    else:
        folder = Path(args)

    print(f"Scanning folder  : {folder}")
    print(f"File type        : {args.file_type}")
    print(f"Output file list : {args.file_list}")

    if args.file_type == "nd2":
        df = scan_folder_nd2(folder)
    elif args.file_type == "lsm":
        df = scan_folder_lsm(folder)
    else:
        df = scan_folder_tiff(folder)

    # save the filelist
    if isinstance(args, argparse.Namespace):
        if args.file_list is not None:
            print(f"Saving filelist table to csv file {args.file_list}")
            df.to_csv(args.file_list, index_label="index")
        else:
            print(df)

    # create a default configuration file
    if isinstance(args, argparse.Namespace):
        if args.config is not None:
            if os.path.exists(args.config) is False:
                nchannels = df["channels"][0]
                config = {
                    "channels": [
                        {"index": k, "name": "undefined"} for k in range(nchannels)
                    ],
                    "scale_um": 50,
                }
                with open(args.config, "w") as fp:
                    json.dump(config, fp)

    return df


def process(args):
    """Sub command for processing item from a list of file / fov"""

    print(f"[ Process v{__version__}]")

    ### Parsing command line arguments ###

    idx = args.index  # from 0 to N-1

    filelist = pd.read_csv(args.file_list)

    if args.data_path is not None:
        filename = os.path.join(args.data_path, filelist["filename"][idx])
    else:
        filename = filelist["filename"][idx]

    fov = filelist["fov"][idx]

    if args.config is not None:
        with open(args.config, "r") as configfile:
            config = json.load(configfile)
    else:
        with open(os.path.join(args.data_path, "config.json"), "r") as configfile:
            print("Loading default configuration file")
            config = json.load(configfile)

    print(f"File: {filename}")
    print(f"Field of view: {fov}")
    print(f"Configuration: {config}")
    print(f"Analysis: {config['Analysis']}")

    ### Start processing ###
    if config["Analysis"] == "SG" or config["Analysis"] == "SG2":
        stats, mip, labels, rois = process_fov(filename, fov, config)
    else:
        stats, mip, labels, rois = process_fov_spread(filename, fov, config)

    ### Save the results ###
    if args.output_by_cells is not None:
        print(f"Saving csv table to file {args.output_by_cells}")
        stats["input"] = args.file_list
        stats["index"] = args.index
        stats.to_csv(args.output_by_cells, index=False)

    if args.output_vignette is not None:
        print(f"Saving vignette to file {args.output_vignette}")
        plt.style.use("default")
        plt.figure(figsize=(20, 20))
        show_image(mip, labels, rois, stats)
        name = filelist["filename"][idx]
        condition = filelist["condition"][idx]
        plt.title(f"#{idx} file:{name} fov:{fov} condition:{condition}")
        plt.savefig(args.output_vignette)


def facet_plot(data, cols, columns=4):
    """Create a boxplot for each column in a dataframe

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to plot.
    cols : list
        List of column names to plot.
    columns : int, optional
        Number of columns in the facet grid (default is 4).

    """
    import math

    rows = math.ceil(len(cols) / columns)
    _, ax = plt.subplots(rows, columns, figsize=(6 * columns, 6 * rows))
    ax = np.reshape(ax, (rows, columns))
    for r in range(rows):
        for c in range(columns):
            if columns * r + c < len(cols) - 1:
                try:
                    sns.boxplot(
                        data=data, y="condition", x=cols[columns * r + c], ax=ax[r, c]
                    )
                    if c > 0:
                        ax[r, c].set(ylabel=None)
                        ax[r, c].set(yticklabels=[])

                except Exception as e:
                    print("***")
                    print(f'* could not show column "{cols[columns * r + c]}"')
                    print(f"* {e}")
                    print("***")


def make_figure(args):
    """Make a figure

    Save a cells.csv and cells.pdf file in {data_path}/results

    Parameters
    ----------
    args: argparse.args
        file_list and data_path as attribute

    """
    print(f"[ Figure v{__version__}]")
    plt.style.use("default")
    cells = []
    folder = Path(args.data_path)
    filelist = pd.read_csv(args.file_list, index_col="index")

    for k in range(len(filelist)):
        try:
            cells.append(pd.read_csv(folder / "results" / f"cells{k:06d}.csv"))
        except Exception as e:
            print(e)
            print(f"missing {k}")

    cells = pd.concat(cells)
    cells = cells.join(filelist, on="index")

    csvname = folder / "results" / "cells.csv"
    print(f"Saving data to file {csvname}")
    cells.to_csv(csvname)

    print("Filtering out cell with more or less than 1 nuclei")
    cells = cells[cells["Number of nuclei"] == 1]

    sns.set_theme()
    sns.set_style("ticks")

    # filter our columns
    exclude = ["index", "filename", "fov", "channels", "Cell ID", "input"]
    columns = [x for x in cells.columns if x not in exclude]

    # make a boxplot for each selected column
    facet_plot(cells, columns, 7)
    figname = os.path.join(args.data_path, "results", "cells.pdf")
    print(f"Saving figure to file {figname}")
    plt.savefig(figname, dpi=150)


def version(args):
    print(__version__)


if __name__ == "__main__":
    # create the argument parser
    parser = argparse.ArgumentParser(
        description=f"Stress granules analysis {__version__}"
    )
    subparsers = parser.add_subparsers(help="sub-command help")

    # add the scan subparser
    parser_scan = subparsers.add_parser("scan", help="scan help")
    parser_scan.add_argument("--data-path", help="folder to scan", required=True)
    parser_scan.add_argument("--file-list", help="filelist")
    parser_scan.add_argument("--file-type", help="file type (n2 or lsm)")
    parser_scan.add_argument("--config", help="json configuration file")
    parser_scan.set_defaults(func=scan)

    # add the process subparser
    parser_process = subparsers.add_parser("process", help="process help")
    parser_process.add_argument("--data-path", help="path to data")
    parser_process.add_argument("--file-list", help="filelist", required=True)
    parser_process.add_argument("--index", help="file index", type=int, required=True)
    parser_process.add_argument("--config", help="json configuration file")
    parser_process.add_argument(
        "--output-by-cells", help="filename of the output csv table by cell"
    )
    parser_process.add_argument(
        "--output-vignette", help="filename of the output vignette file"
    )
    parser_process.set_defaults(func=process)

    # add the figure subparser
    parser_figure = subparsers.add_parser("figure", help="process help")
    parser_figure.add_argument("--data-path", help="path to data")
    parser_figure.add_argument("--file-list", help="filelist", required=True)
    parser_figure.set_defaults(func=make_figure)

    parser_scan = subparsers.add_parser("version", help="print verison")
    parser_figure.set_defaults(func=version())

    args = parser.parse_args()
    args.func(args)
