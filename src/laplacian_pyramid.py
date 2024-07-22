# The code for question 4 part a in maman 11, course: 22928
# Nadav Kahlon, November 2021.

import numpy as np
from matplotlib import pyplot as plt
import cv2
import math


# we create the kernel used for blurring images globally, so other modules can use it; It needs to be symetric and have equal-contribution
a = 0.4 # the a parameter for the kernel
kernel1D = np.array([[0.25-a/2], [0.25], [a], [0.25], [0.25-a/2]]) # the kernel in its 1-dimensional form
kernel2D = kernel1D @ np.transpose(kernel1D) # the kernel in its 2-dimensional form


# reduce: reduces an image to half its resolution; the kernel used for blurring (which should be symetric with equal contribution) is 'kernel'
def reduce(image, kernel=kernel2D):
    result = cv2.filter2D(src=image, ddepth=-1, kernel=kernel) # blur the image using the given kernel
    result = np.array([ [image[2*i][2*j]
                         for j in range(math.floor(image.shape[1]/2))]
                         for i in range(math.floor(image.shape[0]/2)) ]) # down-sampling: sample every second pixel in the image
    return result


# expand: expands an image to twice its resolution; the kernel used for blurring (which should be symetric with equal contribution) is 'kernel'
def expand(image, kernel=kernel2D):
    grid_im = np.array([ [image[math.floor(i/2)][math.floor(j/2)] if (i%2==0 and j%2==0) else np.zeros(image[0][0].shape)
                          for j in range(image.shape[1]*2)]
                          for i in range(image.shape[0]*2) ]) # up-sampline: exapnd the resolution and add black pixels in-between
    blur_grid = cv2.filter2D(src=grid_im, ddepth=-1, kernel=kernel) # blur the gird-image using the given equal-contribution blurring kernel
    result = blur_grid * 4 # we multiply the blurred grid by 4 because every pixel in the original image correspond to 4 pixels in the expanded image 
    return result


# imagePyramids: creates a gaussian pyramid and a laplacian pyramid of height 'height' for a given image; the kernel used for blurring (which should be
#   symetric with equal contribution) is 'kernel'. Note: we assume that the image dimensions are an equal power of 2.
#   If 'height' is None, we create pyramids of maximum height.
def imagePyramids(image, kernel=kernel2D, height=None):
    if height == None: height = math.ceil(math.log2(image.shape[0])) + 1 # if 'height' is None, we create pyramids of maximum height
    
    # first we create the gaussian pyramid:
    gpyramid = [] # a list to hold the different layers of the gaussian pyramid
    curr_image = np.array(image) # create a copy of the image
    for i in range(height):
        gpyramid.append(curr_image) # we add the current image to the next layer in the pyramid
        curr_image = reduce(curr_image, kernel) # and reduce the current image
    
    # next we create the laplacian pyramid
    lpyramid = [] # a list to hold the different layers of the laplace pyramid
    # in each layer of the laplacian pyramid, we insert the difference between the corresponding layer in the gaussian pyramid, and the expanded next layer:
    for i in range(height-1):
        lpyramid.append(gpyramid[i] - expand(gpyramid[i+1], kernel))
    lpyramid.append(gpyramid[-1]) # we also insert the last layer of the gaussian pyramid to the top of the laplacian pyramid
    
    return gpyramid, lpyramid # return the pyramids


# plotPyramids: plots the given gaussian and laplacian pyramids one on top of the another using matplotlib.pyplot
def plotPyramids(gpyramid, lpyramid):
    height = len(gpyramid) # the height of the pyramid (the same for both)
    
    # now, we create a number of subplots to plot the pyramids:
    ncols = math.ceil(height/2) # the number of subplots in each row
    fig, axes = plt.subplots(nrows=4, ncols=ncols)
    
    # plot the gaussian pyramid on the first 2 rows:
    for i in range(height):
        curr_subplot = axes[math.floor(i/ncols)][i%ncols] # the subplot for the current image
        curr_subplot.imshow(gpyramid[i], cmap='gray') # show the pyramid layer
        curr_subplot.axis('off') # discard the axis
        curr_subplot.set_title('Gauss: ' + str(i)) # tell what layer it is
    
    # plot the laplacian pyramid on the last 2 rows:
    for i in range(height):
        curr_subplot = axes[2 + math.floor(i/ncols)][i%ncols] # the subplot for the current image
        curr_subplot.imshow(lpyramid[i], cmap='gray') # show the pyramid layer
        curr_subplot.axis('off') # discard the axis
        curr_subplot.set_title('Laplace: ' + str(i)) # tell what layer it is
    
    # if there are left subplots, we make sure to remove them
    if height % 2 != 0:
        fig.delaxes(axes[1][math.floor(height/2)])
        fig.delaxes(axes[3][math.floor(height/2)])


# main program:
if __name__ == '__main__':
    # we load our image and turn it to gray-scale
    image_path = r'..\resources\cool_bear.png' # the path for the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    
    # next, we generate the pyramids:
    gpyramid, lpyramid = imagePyramids(image, kernel2D)
    
    # and finally - plotting them in full screen
    plotPyramids(gpyramid, lpyramid)
    figManager = plt.get_current_fig_manager()
    plt.show()