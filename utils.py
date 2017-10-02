"""Collection of utility functions."""

import matplotlib.pyplot as plt
import numpy as np

def imshow_gray(image,rescale=False):
    """ Displays an image in gray-scale.

    Args:
        image (ndarray): a 2D array representing the image

        rescale (boolean, optional): rescale the image so entries are in the
            range [0,1].
    """
    if not rescale:
        assert np.max(image) <= 1, 'max value must be <= 1'
        assert np.min(image) >= 0, 'max value must be >= 0'
        plt.imshow(image, cmap=plt.get_cmap('gray'),vmin=0,vmax=1,interpolation='None')
    else:
        plt.imshow(image, cmap=plt.get_cmap('gray'),interpolation='None')

    plt.draw()
    plt.pause(0.001)
