import sys
import cv2
import numpy as np
import pylab
from timer import Timer

def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

if __name__ == '__main__':
    im1 = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(cv2.imread(sys.argv[2]), cv2.COLOR_BGR2GRAY)

    print "image sizes:", im1.shape, im2.shape

    # XXX - paramter - number of desired keypoints
    # Hack: the number of keypoints found in an image seems to be sensisitive
    # to image blur, so estimate the amount or blur and scale the detector
    # threshold equivalently.
    #
    # Another approach would be to binary search for the threshodl that gives
    # the desired number of detected keypoints.
    #
    # Around 

    diff1 = im1[:-2].astype(float) - im1[2:]
    diff2 = im2[:-2].astype(float) - im2[2:]
    thresh1 = int(abs(diff1).mean() * 45.0 / 16.0)
    thresh2 = int(abs(diff2).mean() * 45.0 / 16.0)

    detector = cv2.FastFeatureDetector(threshold=thresh1, nonmaxSuppression=True)
    with Timer("Feature detection"):
        kp1 = detector.detect(im1, None)
        detector.setInt('threshold', thresh2)
        kp2 = detector.detect(im2, None)
    print "    number of keypoints:", len(kp1), len(kp2)

    extractor = cv2.DescriptorExtractor_create('FREAK') 
    # XXXX - parameter - pattern scale
    extractor.setDouble('patternScale', 150.0)
    with Timer("Descriptor extraction"):
        kp1, desc1 = extractor.compute(im1, kp1)
        kp2, desc2 = extractor.compute(im2, kp2)

    # match keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc1, desc2, k=2)

    # XXX - threshold relative to 2nd match
    thresh = 0.8
    good_matches = [m for m in matches if m[0].distance < thresh * m[1].distance]


    # Estimate T, R from Ransac
    print "good", len(good_matches), "matches"
    match_distances = np.array([dist(kp1[m[0].queryIdx].pt, kp2[m[0].trainIdx].pt) for m in good_matches])
    print "    ", np.sum(match_distances < 20), "with image distance < 10"

    from ransac import ransac
    print "RANSAC"
    err, rot, trans, dists, iters = ransac(kp1, kp2, [m[0] for m in good_matches], inlier_pct=10)
    print "   max inlier error", err
    print "   R", rot
    print "   T", trans
    print "   distances:", sorted(dists)
