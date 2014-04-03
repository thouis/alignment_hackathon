import sys
import cv2
import structure
import pylab
import scipy.ndimage as nd
import numpy as np
from timer import Timer

def find_corners(im, name):
    coherence, _, _ = structure.orientation_field(im, 11)
    pylab.imshow(im, cmap=pylab.cm.gray)
    pylab.figure()
    pylab.imshow(coherence)
    pylab.title('coherence')
    pylab.colorbar()

    local_mean = nd.filters.gaussian_filter(im.astype(float), 20)
    local_variance = nd.filters.gaussian_filter(im.astype(float) ** 2.0, 20) - local_mean ** 2

    pylab.figure()
    pylab.imshow(np.sqrt(local_variance) / local_mean, cmap=pylab.cm.gray)
    pylab.title('std / mean')
    pylab.colorbar()


    potential_corners = coherence / np.sqrt(local_variance)

    pylab.figure()
    pylab.imshow(potential_corners, cmap=pylab.cm.gray) 

    pylab.colorbar()


    # find local max with a suppression window of radius 11
    pc_max = nd.minimum_filter(potential_corners, (20, 20))
    corners = potential_corners == pc_max
    print np.sum(corners)
    corners = nd.maximum_filter(corners, (5, 5))
    pylab.show()

    imtmp = np.dstack((im, im, im))
    imtmp[:, :, 2][corners] = 255
    cv2.imshow(name, imtmp[::2, ::2])
    

if __name__ == '__main__':
    im1 = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(cv2.imread(sys.argv[2]), cv2.COLOR_BGR2GRAY)

    print "image sizes:", im1.shape, im2.shape
    find_corners(im1, "im1")
    find_corners(im2, "im2")

    cv2.waitKey(0)
