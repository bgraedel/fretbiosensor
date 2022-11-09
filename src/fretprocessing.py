"""This module contains functions for preprocessing flatfield and darkcurrent images
 and subsequently calculating the ratio."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, median_filter
from skimage import io


def load_correction_images(ch1_darkcurrent_folder):
    """
    Averages the dark current images in the folder.

    Parameters
    ----------
    ch1_darkcurrent_folder : str
        Path to the folder containing the dark current images.

    Returns
    -------
    dark_current : numpy array
        Averaged dark current image.
    """
    ch1_darkcurrent = []
    for i in os.listdir(ch1_darkcurrent_folder):
        if i.endswith(".tif") or i.endswith(".tiff"):
            ch1_darkcurrent.append(io.imread(os.path.join(ch1_darkcurrent_folder, i)))

    ch1_darkcurrent = np.array(ch1_darkcurrent)
    darkcurrent = np.median(ch1_darkcurrent, axis=0)
    return darkcurrent


def filter_darkfield_image(
        darkfield_image: np.ndarray,
        median_filter_size: tuple[int, int] = (3, 3)) -> np.ndarray:
    """
    Corrects the darkfield image for dark current.

    Parameters
    ----------
    darkfield_image : np.ndarray
        Darkfield image.
    median_filter : None | tuple[int, int], optional
        Size of the median filter to apply to the darkfield image.
        The default is (3,3).

    Returns
    -------
    darkfield_corrected : np.ndarray
        Corrected darkfield image.
    """
    darkfield_image_corrected = median_filter(darkfield_image, median_filter_size)
    return darkfield_image_corrected


def correct_flatfield_image(
    flatfield_image: np.ndarray,
    darkfield_image: np.ndarray,
    normalization: bool = True,
    median_filter_size: None | tuple[int, int] = (3, 3),
) -> np.ndarray:
    """
    Corrects the flatfield image for darkfield
    and applies normalization to mean of 1.

    Parameters
    ----------
    flatfield_image : np.ndarray
        Flatfield image.
    darkfield_image : np.ndarray
        Darkfield image.
    normalization : bool, optional
        Whether to normalize the flatfield image to mean of 1,
        strongly reccomended. The default is True.
    median_filter : None | tuple[int, int], optional.
        Size of the median filter to apply to the flatfield image.
        The default is (3,3).
    clip_values : bool, optional
        Whether to clip the values of the flatfield image below 1 to 1.
        The default is True.


    Returns
    -------
    flatfield_corrected : np.ndarray
        Corrected flatfield image.
    """
    flatfield_image = np.subtract(flatfield_image, darkfield_image)
    if normalization:
        flatfield_image = np.divide(flatfield_image, np.mean(flatfield_image))
    if median_filter_size:
        flatfield_image = median_filter(flatfield_image, median_filter_size)
    return flatfield_image


def flatfield_correction(
    img: np.ndarray,
    flatfield: np.ndarray,
    dark_current: np.ndarray,
    clip_image: bool = True,
):
    """
    Corrects the image for flatfielding and dark current.P

    Parameters
    ----------
    img : numpy array
        Image to correct.
    flatfield : numpy array
        Flatfield image.
    dark_current : numpy array
        Dark current image.
    clip_image : bool, optional
        Whether to clip the image to values above 0 after darkcurrent correction. The default is True.

    Returns
    -------
    img : numpy array
        Corrected image.
    """
    img = np.subtract(img, dark_current)
    if clip_image:
        img = np.where(img < 0, 0, img)
    img = np.true_divide(img, flatfield)
    return img


def bg_calculation(img, masked_frame: None | np.ndarray, iter_number: int = 20) -> np.ndarray:
    """
    Calculates the background of an image. Either by taking the median of the
    pixels in the image that are not part of the segmented object or by
    calculating the 75th percentile of the pixels in the image.

    Parameters
    ----------
    img : numpy array
        Image to calculate the background of.
    masked_frame : numpy array
        Binary masked image of the segmented object
    iterations : int, optional
        Number of iterations to expand the mask before
        calculating the backgroundmask.
        The default is 20.

    Returns
    -------
    bg : numpy array
        Background of the image.
    """
    if not isinstance(masked_frame, np.ndarray):
        hist, bins = np.histogram(img, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align="center", width=width)
        plt.ylim(0, 450000)
        plt.xlim(0, 750)
        bg = np.percentile(bins[0:30], 75)
    else:
        frame_segmented_expanded = binary_dilation(masked_frame, iterations=iter_number)
        bg = np.median(img[frame_segmented_expanded == 0])
    return bg


def subtract_bg(img, bg, clip_values: bool = True):
    """
    Subtracts the background from the image.

    Parameters
    ----------
    img : numpy array
        Image to subtract the background from.
    bg : array-like
        Background to subtract from the image.

    Returns
    -------
    img : numpy array
        Image with background subtracted.
    """
    imgcorrected = np.subtract(img, bg)
    if clip_values:
        imgcorrected[imgcorrected < 1] = 1
    return imgcorrected


def calculate_ratio(numerator, denominator, replace_nan_and_inf: bool = True):
    """
    Calculates the ratio of the numerator to the denominator.

    Parameters
    ----------
    numerator : numpy array
        Image to use as the numerator.
    denominator : numpy array
        Image to use as the denominator.
    replace_nan_and_inf : bool, optional
        Whether to replace nan and inf values with 0. The default is True.

    Returns
    -------
    ratio : numpy array
        Image of the ratio of the numerator to the denominator.
    """
    ratio_image = np.true_divide(numerator, denominator)
    if replace_nan_and_inf:
        ratio_image = np.nan_to_num(ratio_image)
    return ratio_image
