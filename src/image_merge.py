# The code for question 4 part b in maman 11, course: 22928
# Nadav Kahlon, November 2021.

import numpy as np
from matplotlib import pyplot as plt
import cv2
from laplacian_pyramid   import reduce, expand, imagePyramids, kernel2D


# collapse: collapses a laplace pyramid to a single image; the kernel used for blurring (which should be symetric with equal contribution) is 'kernel'
def collapse(lpyramid, kernel=kernel2D):
    image = lpyramid[-1] # we set the image to be the last entry in the pyramid 
    # and then collapse it onto each layer below it:
    for i in reversed(range(len(lpyramid)-1)):
        image = expand(image, kernel) # expand the last collapsed layer
        image += lpyramid[i] # and "drop" it onto the next one
    return image

# joinLPyramids: merges 2 Laplace pyramids into 1 Laplace pyramid using 'merge_filter': 'lpyramid1' is picked where 'merge_filter' is positive, and 'lpyramid2' is 
#   picked where 'merge_filter' is zero. Assumes the pyramids are of the same height, and that the dimensions of each base are an equal power of 2 (and
#   'merge_filter' is of the same base-dimensions);  the kernel used for reducing the filter (which should be symetric with equal contribution) is 'kernel'
def joinLPyramids(lpyramid1, lpyramid2, merge_filter, kernel=kernel2D):
    result = [] # a list to hold the new Laplacian pyramid
    curr_filter = np.array(merge_filter) # the filter for the current layer (a copy of the given filter)
    for i in range(len(lpyramid1)):
         # we merge the current layers of both pyramids together using the filter, and add it to the new pyramid
        result.append(np.where(curr_filter>0, lpyramid1[i], lpyramid2[i]))
        curr_filter = reduce(curr_filter, kernel) # and reduce the filter to the next layer
        curr_filter = np.where(curr_filter > 0, 1., 0.) # we make sure that the filter is still holding only binary values
    return result

# main program: merge a 512x512 image of an eye and a 512x512 image of a hand, making it look like the eye is in the middle of the hand.
if __name__ == '__main__': 
    pyrmid_height = 7 # the height of the pyramids
    
    # loading the eye image and creating pyramids for it
    image_path1 = r'..\resources\eye.jpg' # path for the image
    image1 = cv2.imread(image_path1) / 255 # loading and normalizing the image
    gpyramid1, lpyramid1 = imagePyramids(image1, kernel2D, pyrmid_height) # creating pyramids for it
    
    image_path2 = r'..\resources\hand.jpg' # path for the image
    image2 = cv2.imread(image_path2) / 255 # loading and normalizing the image
    gpyramid2, lpyramid2 = imagePyramids(image2, kernel2D, pyrmid_height) # creating pyramids for it
    
    # now, we create the merging 512x512 filter: I pre-measured the paramaters so that the filter will form an ellipse exactly around the eye; inside the ellipse
    #   the filter will be 1 (corresponding to image1 - the eye) and outside of it it will be 0 (corresponding to image2 - the hand)
    mergeFilter = np.array([ [(1.,1.,1.) if (j-270)**2 + 5*(i-282)**2 < 85**2 else (0.,0.,0.)
                              for j in range(512)]
                              for i in range(512) ])
    
    join_lpyramid = joinLPyramids(lpyramid1, lpyramid2, mergeFilter) # using that filter, we merge the 2 Laplacian pyramids:
    merged_im = collapse(join_lpyramid, kernel2D) # and collapse it, to form the merged image
    
    # we plot the process using matplotlib.pyplot:
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0][0].imshow(cv2.cvtColor(np.float32(image1), cv2.COLOR_BGR2RGB));     axes[0][0].axis('off')
    axes[0][1].imshow(cv2.cvtColor(np.float32(image2), cv2.COLOR_BGR2RGB));     axes[0][1].axis('off')
    axes[1][0].imshow(mergeFilter);                                             axes[1][0].axis('off')
    axes[1][1].imshow(cv2.cvtColor(np.float32(merged_im), cv2.COLOR_BGR2RGB));  axes[1][1].axis('off')
    plt.show()
    
    # finally, we show the merged image with full resolution using cv2:
    cv2.imshow('Eye In a Hand', merged_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()