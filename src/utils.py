"""This module contains some functions to help with segmentation and analysis"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.morphology import erosion, remove_small_objects, square


def mask_frame(img, mask):
    """
    Masks the image with the mask.

    Parameters
    ----------
    img : numpy array
        Image to mask.
    mask : numpy array
        Mask to apply to the image.

    Returns
    -------
    masked_frame : numpy array
        Masked image.
    """
    masked_frame = np.where(mask == 0, 0, img)
    return masked_frame


def smooth_segmentation(binary_objects, expand_iterations=3, remove_small=True, remove_small_objects_size=100):
    """
    Smooths the segmentation by removing small objects and filling holes.

    Parameters
    ----------
    binary_objects : numpy array
        Binary image of the segmented objects.
    remove_small : bool, optional
        Whether to remove small objects. The default is True.
    remove_small_objects_size : int, optional
        Size of the objects to remove. The default is 100.

    Returns
    -------
    binary_objects : numpy array
        Smoothed binary image of the segmented objects.
    """
    if len(binary_objects.shape) == 3:
        for index, image in enumerate(binary_objects):
            image = binary_fill_holes(image)
            image = binary_dilation(image, square(5), iterations=expand_iterations)
            image = erosion(image, footprint=square(5))
            bool_img = image.astype(bool)
            if remove_small:
                image = remove_small_objects(bool_img, min_size=remove_small_objects_size**2)
            image = binary_fill_holes(image)
            binary = np.where(image, 1, 0)
            binary_objects[index] = binary
    else:
        binary_objects = binary_fill_holes(binary_objects)
        binary_objects = binary_dilation(binary_objects, square(5), iterations=expand_iterations)
        binary_objects = erosion(binary_objects, footprint=square(5))
        bool_img = binary_objects.astype(bool)
        if remove_small:
            binary_objects = remove_small_objects(bool_img, min_size=remove_small_objects_size**2)
        binary_objects = binary_fill_holes(binary_objects)
        binary_objects = np.where(binary_objects, 1, 0)
        return binary_objects