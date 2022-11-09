"""This module contains some utility functions to help with io operations"""

from re import search
import os

from typing import List


def create_folders(path, directory):
    """
    Creates the folders to save the images in.

    Parameters
    ----------
    path : str
        Path to the folder to save the images in.
    directory : str
        Name of the folder to create.

    Returns
    -------
    None.
    """
    for i in directory:
        iDir = os.path.join(path, i)
        try:
            if not os.path.exists(iDir):
                os.makedirs(iDir)
                print("Created folder: ", iDir)
            else:
                print("Folder: ", iDir, "already exists")
        except OSError:
            print("Error: Creating directory. " + iDir)


def sort_with_regex(img_list: List[str], pattern, match_group_id):
    """
    Sorts the images in the list using the regex pattern.

    Parameters
    ----------
    img_list : list
        List of images to sort.
    pattern : str
        Regex pattern to use to sort the images.
    match_group_id : int
        Group id of the regex pattern to use to sort the images.

    Returns
    -------
    img_list : list
        Sorted list of images.
    """

    img_list = [i for i in img_list if i.endswith(".tif") or i.endswith(".tiff")]
    match = [int(search(pattern, match).group(match_group_id)) for match in img_list]
    sorted_matches = sorted(match)
    tuple_list = [(i, j) for i, j in zip(img_list, match)]
    tuple_list.sort(key=lambda i: sorted_matches.index(i[1]))
    sorted_image_list = [i[0] for i in tuple_list]
    return sorted_image_list


def rebin(arr, new_shape):
    """
    Rebins an array to a new shape.

    Parameters
    ----------
    arr : numpy array
        Array to rebin.
    new_shape : tuple
        Shape to rebin the array to.

    Returns
    -------
    arr : numpy array
        Re-binned array.
    """
    shape = (
        new_shape[0],
        arr.shape[0] // new_shape[0],
        new_shape[1],
        arr.shape[1] // new_shape[1],
    )
    return arr.reshape(shape).mean(-1).mean(1)
